from pathlib import Path
from pprint import pprint

import fire
import torch
import yaml

from cv_preparation import (cv_train_preparation, )
from nlp_preparation import (nlp_train_preparation, nlp_evaluation_preparation)
from src.utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.full_load(file)
    pprint(f"Config: {config}")
    return config


def compute_nlp_part(train, evaluate):
    config_path = Path("configs") / "nlp.yml"
    config = load_config(config_path)

    seed = config.get("seed")
    set_seed(seed)

    batch_size = config.get("batch_size")
    model_name = config.get("model_name")
    max_seq_len = config.get("max_seq_len")
    n_classes = config.get("n_classes")

    if train:
        train_conf = config.get("train", {})
        train_folds = train_conf.get("train_folds")
        val_folds = train_conf.get("val_folds")
        epochs = train_conf.get("epochs")

        nlp_train_preparation(
            batch_size,
            model_name,
            max_seq_len,
            train_folds,
            val_folds,
            epochs,
            n_classes,
            device)
    if evaluate:
        eval_conf = config.get("evaluation", {})
        model_path = eval_conf.get("model_path")
        nlp_evaluation_preparation(
            batch_size,
            model_name,
            model_path,
            max_seq_len,
            n_classes,
            device
        )


def compute_cv_part(train, evaluate):
    config_path = Path("configs") / "cv.yml"
    config = load_config(config_path)

    seed = config.get("seed")
    set_seed(seed)

    batch_size = config.get("batch_size")
    model_name = config.get("model_name")
    n_classes = config.get("n_classes")

    if train:
        train_conf = config.get("train", {})
        train_folds = train_conf.get("train_folds")
        val_folds = train_conf.get("val_folds")
        epochs = train_conf.get("epochs")
        learning_rate = train_conf.get("learning_rate")
        print(learning_rate)

        cv_train_preparation(
            batch_size,
            model_name,
            learning_rate,
            train_folds,
            val_folds,
            epochs,
            n_classes,
            device)
    if evaluate:
        eval_conf = config.get("evaluation", {})
        model_path = eval_conf.get("model_path")
        pass


def main(pipeline_type="cv", train=False, evaluate=False):
    print(
        f"pipeline_type: {pipeline_type}\ntrain: {train}, evaluate: {evaluate}")

    if train and evaluate or (not train and not evaluate):
        print("use either train or evaluate param")
        return
    compute_fn = compute_cv_part if pipeline_type == "cv" else compute_nlp_part
    compute_fn(train, evaluate)


if __name__ == '__main__':
    fire.Fire(main)
