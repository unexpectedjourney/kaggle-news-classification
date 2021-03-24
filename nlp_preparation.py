import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AdamW

from src.data import TextClassificationDataset
from src.eval import eval_nlp
from src.model import TextClassifier
from src.train import fit
from src.utils import (load_splits, get_metrics, get_writer)


def get_dataloaders(
        tokenizer,
        max_seq_len,
        batch_size,
        splits=None,
        test_df=None
):
    dataloaders = {}

    if splits is not None:
        train_dataset = TextClassificationDataset(
            texts=splits["train"]["text"].values.tolist(),
            tokenizer=tokenizer,
            labels=splits["train"]["source"].values.tolist(),
            max_seq_length=max_seq_len
        )

        valid_dataset = TextClassificationDataset(
            texts=splits["val"]["text"].values.tolist(),
            tokenizer=tokenizer,
            labels=splits["val"]["source"].values.tolist(),
            max_seq_length=max_seq_len
        )

        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        val_data_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        dataloaders["train"] = train_data_loader
        dataloaders["val"] = val_data_loader

    if test_df is not None:
        test_dataset = TextClassificationDataset(
            texts=test_df["text"].values.tolist(),
            tokenizer=tokenizer,
            max_seq_length=max_seq_len
        )

        test_data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        dataloaders["test"] = test_data_loader
    return dataloaders


def get_tokenizer(model_name):
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer


def get_test_df(test_df_path, tokenizer):
    test_df = pd.read_csv(test_df_path)
    test_df.text = test_df.title + " " + \
                   tokenizer.sep_token + " " + \
                   test_df.text
    return test_df


def nlp_train_preparation(
        batch_size,
        model_name,
        max_seq_len,
        train_folds,
        val_folds,
        epochs,
        n_classes,
        device
):
    logs_dir = Path("logs")
    states_dir = Path("states") / "nlp"
    states_dir = Path("/media/antonbabenko/hard/news/states") / "nlp"
    folds = Path("folds") / "nlp"

    splits = load_splits(folds, val_folds=val_folds, train_folds=train_folds)

    tokenizer = get_tokenizer(model_name)

    for stage in ["train", "val", "test"]:
        if not stage in splits or splits[stage].empty:
            continue
        splits[stage].text = splits[stage].title + " " + \
                             tokenizer.sep_token + " " + \
                             splits[stage].text

    dataloaders = get_dataloaders(
        tokenizer,
        max_seq_len,
        batch_size,
        splits,
    )

    metrics = get_metrics()
    writer = get_writer(logs_dir, "youscan-ukr-roberta-base")

    model = TextClassifier(
        n_classes=n_classes,
        pretrained_model_name=model_name,
        dropout=0.4
    ).to(device)

    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": 2e-6},
        {"params": model.out.parameters(), "lr": 2e-3}],
        correct_bias=False
    )
    total_steps = len(dataloaders["train"]) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss().to(device)

    fit(
        model,
        criterion,
        optimizer,
        dataloaders,
        device,
        scheduler=scheduler,
        metrics=metrics,
        epochs=epochs,
        model_name=model_name,
        model_folder=states_dir,
        writer=writer
    )


def nlp_evaluation_preparation(
        batch_size,
        model_name,
        model_path,
        max_seq_len,
        n_classes,
        device
):
    states_dir = Path("states") / "nlp"
    states_dir = Path("/media/antonbabenko/hard/news/states") / "nlp"
    data_dir = Path("data")
    submissions = Path("submissions") / "nlp"
    test_df_path = data_dir / "test_without_target.csv"

    checkpoint_path = states_dir / model_path

    tokenizer = get_tokenizer(model_name)
    test_df = get_test_df(test_df_path, tokenizer)

    dataloaders = get_dataloaders(
        tokenizer,
        max_seq_len,
        batch_size,
        test_df=test_df
    )

    model = TextClassifier(
        n_classes=n_classes,
        pretrained_model_name=model_name,
        dropout=0.4
    )

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Models have been loaded")

    model = model.to(device)

    predictions, probs = eval_nlp(model, dataloaders["test"], device)
    probs = probs.cpu().numpy()
    probs = probs.T

    result_dict = {
        key: prob for key, prob in enumerate(probs)
    }
    result_dict["Id"] = test_df.Id.tolist()

    prob_submission_df = pd.DataFrame(result_dict)

    test_df["Predicted"] = predictions

    submission_df = test_df[["Id", "Predicted"]]
    submission_df.to_csv(
        submissions / f"{model_name.replace('/', '-')}-{int(datetime.datetime.now().timestamp())}.csv",
        index=False)

    return submission_df, prob_submission_df
