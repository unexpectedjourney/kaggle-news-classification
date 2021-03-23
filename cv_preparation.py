from pathlib import Path

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from pytorch_toolbelt import losses as L
from torch.utils.data import DataLoader

from src.data import ImageClassificationDataset
from src.train import fit
from src.utils import load_splits, get_metrics, get_writer


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


def get_dataloaders(splits, image_path, batch_size):
    transform, test_transform = get_augmentations()

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
    # test_dataset = ImageClassificationDataset(
    #     df=splits["test"],
    #     folder=image_path,
    #     mode="test",
    #     transform=test_transform
    # )

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

    # test_data_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    # )

    dataloaders = {
        "train": train_data_loader,
        "val": val_data_loader,
        #     "test": test_data_loader
    }
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
        splits, image_path, batch_size,
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
