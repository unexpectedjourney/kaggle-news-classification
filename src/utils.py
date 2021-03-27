import datetime
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from .metrics import f1_macro
from .stopwords_ua import stopwords

REPLACE_BY_SPACE_RE = re.compile('[/{}\[\]\|@,;.]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-zА-ЩЬЮЯҐЄІЇа-щьюяґєії #+_&]')
HTML_TAGS = re.compile('<[^>]*>')


def text_cleaning(text):
    text = text.lower()
    text = HTML_TAGS.sub(' ', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in stopwords)
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

    train_df = pd.concat([
        pd.read_csv(folder / f"fold_{fold}.csv") for fold in train_folds
    ]).reset_index(drop=True) if train_folds else pd.DataFrame()

    val_df = pd.concat([
        pd.read_csv(folder / f"fold_{fold}.csv") for fold in val_folds
    ]).reset_index(drop=True) if val_folds else pd.DataFrame()

    test_df = pd.concat([
        pd.read_csv(folder / f"fold_{fold}.csv") for fold in test_folds
    ]).reset_index(drop=True) if test_folds else pd.DataFrame()

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
    metrics = {
        "f1_macro": f1_macro
    }

    return metrics


def get_writer(logs_dir, writer_name):
    model_name = f"{writer_name}-{int(datetime.datetime.now().timestamp())}"
    writer = get_tensorboard_writer(logs_dir, model_name)
    return writer
