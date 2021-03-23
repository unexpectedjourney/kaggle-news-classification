import datetime
from pathlib import Path

import albumentations as A
import pandas as pd
import torch
import ttach as tta
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from pytorch_toolbelt import losses as L
from torch.utils.data import DataLoader

from src.data import ImageClassificationDataset
from src.eval import eval_cv, summarize_cv_predictions
from src.train import fit
from src.utils import load_splits, get_metrics, get_writer


def get_test_df(test_df_path):
    test_df = pd.read_csv(test_df_path)
    test_df = test_df[["Id", "images"]]

    idx_list = []
    image_pathes = []
    inner_df = test_df.copy()
    inner_df = inner_df.dropna()
    for i in range(inner_df.shape[0]):
        idx = inner_df.iloc[i, 0]
        images = inner_df.iloc[i, 1]

        for image in images.split(","):
            idx_list.append(idx)
            image_pathes.append(image)
    test_df = pd.DataFrame({
        "id": idx_list,
        "image": image_pathes
    })
    return test_df


def get_augmentations():
    transform = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(),
        A.RandomRotate90(0.5),
        A.ColorJitter(),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    return transform, test_transform


def get_dataloaders(image_path, batch_size, splits=None, test_df=None):
    transform, test_transform = get_augmentations()

    dataloaders = {}

    if splits is not None:
        train_dataset = ImageClassificationDataset(
            df=splits["train"],
            folder=image_path,
            mode="train",
            transform=transform
        )

        valid_dataset = ImageClassificationDataset(
            df=splits["val"],
            folder=image_path,
            mode="val",
            transform=transform
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
        test_dataset = ImageClassificationDataset(
            df=test_df,
            folder=image_path,
            mode="test",
            transform=test_transform
        )

        test_data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        dataloaders["test"] = test_data_loader

    return dataloaders


def get_model(model_name, n_classes, device):
    model = EfficientNet.from_pretrained(model_name)
    num_ftrs = model._fc.in_features
    model._fc = torch.nn.Linear(num_ftrs, n_classes)
    model = model.to(device)
    return model


def cv_train_preparation(
        batch_size,
        model_name,
        learning_rate,
        train_folds,
        val_folds,
        epochs,
        n_classes,
        device
):
    logs_dir = Path("logs")
    states_dir = Path("states") / "cv"
    folds = Path("folds") / "cv"
    data_dir = Path("data")
    image_path = data_dir / "images" / "images"

    splits = load_splits(folds, val_folds=val_folds, train_folds=train_folds)

    dataloaders = get_dataloaders(
        image_path, batch_size, splits=splits,
    )

    metrics = get_metrics()
    writer = get_writer(logs_dir, model_name)

    model = get_model(model_name, n_classes, device)

    criterion = L.FocalLoss().to(device)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3
    )

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
        writer=writer,
        fit_type="cv"
    )


def cv_evaluation_preparation(
        batch_size,
        model_name,
        model_path,
        n_classes,
        device
):
    states_dir = Path("states") / "cv"
    data_dir = Path("data")
    submissions = Path("submissions") / "cv"
    image_path = data_dir / "images" / "images"
    test_df_path = data_dir / "test_without_target.csv"

    checkpoint_path = states_dir / model_path

    test_df = get_test_df(test_df_path)

    dataloaders = get_dataloaders(
        image_path,
        batch_size,
        test_df=test_df
    )

    model = get_model(model_name, n_classes, device)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Models have been loaded")

    # todo add TTA
    model = model.to(device)
    model = tta.ClassificationTTAWrapper(
        model,
        tta.aliases.vlip_transform()
    )

    submission_df = eval_cv(model, dataloaders["test"], device)
    submission_df = summarize_cv_predictions(submission_df)

    submission_df.to_csv(
        submissions / f"{model_name}-{int(datetime.datetime.now().timestamp())}.csv",
        index=False)
