from pathlib import Path
from pprint import pprint

import fire
import torch
import yaml

from combined_preparation import all_evaluation_preparation
from cv_preparation import (cv_train_preparation, cv_evaluation_preparation, )
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
        eval_conf = config.get("evaluate", {})
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
        eval_conf = config.get("evaluate", {})
        model_path = eval_conf.get("model_path")
        cv_evaluation_preparation(
            batch_size,
            model_name,
            model_path,
            n_classes,
            device
        )


def compute_all_parts(train, evaluate):
    config_path = Path("configs") / "all.yml"
    config = load_config(config_path)

    seed = config.get("seed")
    n_classes = config.get("n_classes")

    set_seed(seed)

    cv_conf = config.get("cv", {})
    nlp_conf = config.get("nlp", {})


    if train:
        pass
    if evaluate:
        cv_batch_size = cv_conf.get("batch_size")
        cv_model_name = cv_conf.get("model_name")
        cv_eval_conf = cv_conf.get("evaluate", {})
        cv_model_path = cv_eval_conf.get("model_path")
        cv_coef = cv_eval_conf.get("coef")

        nlp_batch_size = nlp_conf.get("batch_size")
        nlp_model_name = nlp_conf.get("model_name")
        nlp_max_seq_len = nlp_conf.get("max_seq_len")
        nlp_eval_conf = nlp_conf.get("evaluate", {})
        nlp_model_path = nlp_eval_conf.get("model_path")
        nlp_coef = nlp_eval_conf.get("coef")

        all_evaluation_preparation(
            cv_batch_size,
            cv_model_name,
            cv_model_path,
            cv_coef,
            nlp_batch_size,
            nlp_model_name,
            nlp_max_seq_len,
            nlp_model_path,
            nlp_coef,
            n_classes,
            device
        )


def main(pipeline_type="cv", train=False, evaluate=False):
    print(
        f"pipeline_type: {pipeline_type}\ntrain: {train}, evaluate: {evaluate}")

    if train and evaluate or (not train and not evaluate):
        print("use either train or evaluate param")
        return
    if pipeline_type == "cv":
        compute_fn = compute_cv_part
    elif pipeline_type == "nlp":
        compute_fn = compute_nlp_part
    else:
        compute_fn = compute_all_parts

    compute_fn(train, evaluate)


if __name__ == '__main__':
    fire.Fire(main)
