import datetime
import os
import random
import re

import nltk
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from .metrics import f1_macro
from .stopwords_ua import stopwords

REPLACE_BY_SPACE_RE = re.compile('[/{}\[\]\|@,;.]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-zА-ЩЬЮЯҐЄІЇа-щьюяґєії #+_&]')
HTML_TAGS = re.compile('<[^>]*>')


class UkrainianStemmer():
    def __init__(self, word):
        self.word = word
        self.vowel = r'аеиоуюяіїє'  # http://uk.wikipedia.org/wiki/Голосний_звук
        self.perfectiveground = r'(ив|ивши|ившись|ыв|ывши|ывшись((?<=[ая])(в|вши|вшись)))$'
        # http://uk.wikipedia.org/wiki/Рефлексивне_дієслово
        self.reflexive = r'(с[яьи])$'
        # http://uk.wikipedia.org/wiki/Прикметник + http://wapedia.mobi/uk/Прикметник
        self.adjective = r'(ими|ій|ий|а|е|ова|ове|ів|є|їй|єє|еє|я|ім|ем|им|ім|их|іх|ою|йми|іми|у|ю|ого|ому|ої)$'
        # http://uk.wikipedia.org/wiki/Дієприкметник
        self.participle = r'(ий|ого|ому|им|ім|а|ій|у|ою|ій|і|их|йми|их)$'
        # http://uk.wikipedia.org/wiki/Дієслово
        self.verb = r'(сь|ся|ив|ать|ять|у|ю|ав|али|учи|ячи|вши|ши|е|ме|ати|яти|є)$'
        # http://uk.wikipedia.org/wiki/Іменник
        self.noun = r'(а|ев|ов|е|ями|ами|еи|и|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я|і|ові|ї|ею|єю|ою|є|еві|ем|єм|ів|їв|ю)$'
        self.rvre = r'[аеиоуюяіїє]'
        self.derivational = r'[^аеиоуюяіїє][аеиоуюяіїє]+[^аеиоуюяіїє]+[аеиоуюяіїє].*(?<=о)сть?$'
        self.RV = ''

    def ukstemmer_search_preprocess(self, word):
        word = word.lower()
        word = word.replace("'", "")
        word = word.replace("ё", "е")
        word = word.replace("ъ", "ї")
        return word

    def s(self, st, reg, to):
        orig = st
        self.RV = re.sub(reg, to, st)
        return (orig != self.RV)

    def stem_word(self):
        word = self.ukstemmer_search_preprocess(self.word)
        if not re.search('[аеиоуюяіїє]', word):
            stem = word
        else:
            p = re.search(self.rvre, word)
            start = word[0:p.span()[1]]
            self.RV = word[p.span()[1]:]

            # Step 1
            if not self.s(self.RV, self.perfectiveground, ''):

                self.s(self.RV, self.reflexive, '')
                if self.s(self.RV, self.adjective, ''):
                    self.s(self.RV, self.participle, '')
                else:
                    if not self.s(self.RV, self.verb, ''):
                        self.s(self.RV, self.noun, '')
            # Step 2
            self.s(self.RV, 'и$', '')

            # Step 3
            if re.search(self.derivational, self.RV):
                self.s(self.RV, 'ость$', '')

            # Step 4
            if self.s(self.RV, 'ь$', ''):
                self.s(self.RV, 'ейше?$', '')
                self.s(self.RV, 'нн$', u'н')

            stem = start + self.RV
        return stem


def ua_tokenizer(text, ua_stemmer=True, stop_words=[]):
    """ Tokenizer for Ukrainian language, returns only alphabetic tokens.

    Keyword arguments:
    text -- text for tokenize
    ua_stemmer -- if True use UkrainianStemmer for stemming words (default True)
    stop_words -- list of stop words (default [])
    """
    tokenized_list = []
    text = re.sub(r"""['’"`�]""", '', text)
    text = re.sub(r"""([0-9])([\u0400-\u04FF]|[A-z])""", r"\1 \2", text)
    text = re.sub(r"""([\u0400-\u04FF]|[A-z])([0-9])""", r"\1 \2", text)
    text = re.sub(r"""[\-.,:+*/_]""", ' ', text)

    for word in nltk.word_tokenize(text):
        if word.isalpha():
            word = word.lower()
        if ua_stemmer is True:
            word = UkrainianStemmer(word).stem_word()
        if word not in stop_words:
            tokenized_list.append(word)
    return tokenized_list


def text_cleaning(text):
    text = text.lower()
    text = HTML_TAGS.sub(' ', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(ua_tokenizer(text, stop_words=stopwords))
    return text


def slight_cleaning(text):
    text = text.lower()
    return text


def generate_folds_with_exclussions(folds, original_fold, exc_one, exc_two):
    if not original_fold:
        original_fold = [
            fold for fold in folds
            if fold not in exc_one and fold not in exc_two
        ]

    return original_fold


def load_splits(folder,
                train_folds=None,
                val_folds=None,
                test_folds=None,
                enable_generate_test=False):
    if bool(train_folds) + bool(val_folds) + bool(test_folds) < 2:
        raise ValueError("You should specify min 2 fold types")

    folds = [int(fn.stem.split('_')[-1]) for fn in folder.glob("fold_?.csv")]

    if train_folds is None:
        train_folds = []

    if val_folds is None:
        val_folds = []

    if test_folds is None:
        test_folds = []

    train_folds = generate_folds_with_exclussions(folds, train_folds,
                                                  val_folds, test_folds)

    val_folds = generate_folds_with_exclussions(folds, val_folds, train_folds,
                                                test_folds)

    if enable_generate_test:
        test_folds = generate_folds_with_exclussions(folds, test_folds,
                                                     train_folds, val_folds)

    train_df = pd.concat(
        [pd.read_csv(folder / f"fold_{fold}.csv")
         for fold in train_folds]).reset_index(
             drop=True) if train_folds else pd.DataFrame()

    val_df = pd.concat(
        [pd.read_csv(folder / f"fold_{fold}.csv")
         for fold in val_folds]).reset_index(
             drop=True) if val_folds else pd.DataFrame()

    test_df = pd.concat(
        [pd.read_csv(folder / f"fold_{fold}.csv")
         for fold in test_folds]).reset_index(
             drop=True) if test_folds else pd.DataFrame()

    folds = {"train": train_df, "val": val_df, "test": test_df}
    return folds


def get_tensorboard_writer(logs_dir, model_name):
    writer_dir = logs_dir / model_name
    if not writer_dir.exists():
        writer_dir.mkdir()
    return SummaryWriter(writer_dir)


def generate_checkpoint_name(epoch, model_name=None):
    if model_name:
        return f"epoch-{epoch}-{model_name}-checkpoint.pt"
    return f"epoch-{epoch}-checkpoint.pt"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_metrics():
    metrics = {"f1_macro": f1_macro}

    return metrics


def get_writer(logs_dir, writer_name):
    model_name = f"{writer_name}-{int(datetime.datetime.now().timestamp())}"
    writer = get_tensorboard_writer(logs_dir, model_name)
    return writer
