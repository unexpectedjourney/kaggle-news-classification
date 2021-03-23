import time

import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils import generate_checkpoint_name


def train_nlp(model,
              criterion,
              optimizer,
              scheduler,
              train_loader,
              device,
              pbar_desc="train phase"):
    train_loss = 0.0

    model.train()
    print(train_loader, type(train_loader))
    for element in tqdm(train_loader, desc=pbar_desc):
        num = element["input_ids"].size(0)

        input_ids = element["input_ids"].to(device)
        attention_mask = element["attention_mask"].to(device)
        targets = element["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask)
        loss = criterion(outputs, targets)

        _, preds = torch.max(outputs, dim=1)
        train_loss += loss.item() * num

        model.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    train_loss = train_loss / len(train_loader.sampler)

    return {
        "train_loss": train_loss,
    }


def validate_nlp(model,
                 criterion,
                 metrics,
                 val_loader,
                 device,
                 pbar_desc="validation phase"):
    val_loss = 0.0
    val_metrics = {k: 0.0 for k, v in metrics.items()}

    model.eval()

    for element in tqdm(val_loader, desc=pbar_desc):
        num = element["input_ids"].size(0)

        input_ids = element["input_ids"].to(device)
        attention_mask = element["attention_mask"].to(device)
        targets = element["targets"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, targets)

            val_loss += (loss.item() * num)

            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()

            val_metrics = {
                key: value + metrics[key](preds, targets) * num
                for key, value in val_metrics.items()
            }

    val_loss = val_loss / len(val_loader.sampler)
    val_metrics = {
        key: value / len(val_loader.sampler)
        for key, value in val_metrics.items()
    }

    print(val_metrics)
    return {
        "val_loss": val_loss,
        **val_metrics,
    }


def train_cv(model,
             criterion,
             optimizer,
             scheduler,
             train_loader,
             device,
             pbar_desc="train phase"):
    train_loss = 0.0

    model.train()
    for (idx, images, labels) in tqdm(train_loader, desc=pbar_desc):
        num = images.size(0)

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, dim=1)
        train_loss += loss.item() * num

        model.zero_grad()

        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_loader.sampler)

    return {
        "train_loss": train_loss,
    }


def validate_cv(model,
                criterion,
                metrics,
                val_loader,
                device,
                pbar_desc="validation phase"):
    val_loss = 0.0
    val_metrics = {k: 0.0 for k, v in metrics.items()}

    model.eval()

    for (idx, images, labels) in tqdm(val_loader, desc=pbar_desc):
        num = images.size(0)

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, dim=1)

            val_loss += (loss.item() * num)

            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            val_metrics = {
                key: value + metrics[key](preds, labels) * num
                for key, value in val_metrics.items()
            }

    val_loss = val_loss / len(val_loader.sampler)
    val_metrics = {
        key: value / len(val_loader.sampler)
        for key, value in val_metrics.items()
    }

    print(val_metrics)
    return {
        "val_loss": val_loss,
        **val_metrics,
    }


def fit(model,
        criterion,
        optimizer,
        dataloaders,
        device,
        scheduler=None,
        metrics=None,
        epochs=30,
        model_name=None,
        model_folder=None,
        writer=None,
        fit_type=None):
    if metrics is None:
        metrics = {}
    best_val_score = float("-inf")

    for epoch in range(1, epochs + 1):
        start_point = time.time()

        train_pbar_desc = f"Epoch: {epoch}/{epochs}, train phase"
        train_fn = train_cv if fit_type == "cv" else train_nlp
        train_log = train_fn(
            model,
            criterion,
            optimizer,
            scheduler,
            dataloaders["train"],
            device,
            train_pbar_desc
        )
        train_loss = train_log["train_loss"]

        val_pbar_desc = f"Epoch: {epoch}/{epochs}, validation phase"
        val_fn = validate_cv if fit_type == "cv" else validate_nlp
        val_log = val_fn(
            model,
            criterion,
            metrics,
            dataloaders["val"],
            device,
            val_pbar_desc
        )
        val_loss = val_log["val_loss"]
        val_f1 = val_log["f1_macro"]

        if fit_type == "cv":
            scheduler.step(val_loss)

        end_point = time.time()
        spended_time = end_point - start_point

        report = f"Epoch: {epoch}/{epochs}, time: {spended_time}" \
                 f" train loss: {train_loss}, val loss: {val_loss}"
        print(report)

        if writer is not None:
            writer.add_scalar("time", spended_time, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            for key, value in val_log.items():
                writer.add_scalar(key, value, epoch)

        if val_f1 > best_val_score:
            best_val_score = val_f1
            checkpoint_name = generate_checkpoint_name(epoch, model_name)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, model_folder / checkpoint_name)
            print("Checkpoint was saved")
