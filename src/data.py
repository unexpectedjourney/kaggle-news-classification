import cv2
import torch
from torch.utils.data import Dataset

from src.utils import (text_cleaning, slight_cleaning)
from src.constants import TransformerTitleSeparator


def separate_title(row):
    return " ".join([
        row.title,
        TransformerTitleSeparator.END.value,
        row.text,
    ])


class TextClassificationDataset(Dataset):
    def __init__(
            self,
            df,
            tokenizer,
            mode,
            max_seq_length=512,
    ):
        inner_df = df.copy()

        self.labels = None
        if mode != "test":
            self.labels = df.source.tolist()

        inner_df.title = inner_df.title.apply(slight_cleaning)
        inner_df.text = inner_df.text.apply(slight_cleaning)
        # inner_df.title = inner_df.title.apply(text_cleaning)
        # inner_df.text = inner_df.text.apply(text_cleaning)

        self.texts = inner_df.apply(separate_title, axis=1)
        # self.texts = df.text.tolist()


        self.label_dict = None
        self.max_seq_length = max_seq_length

        if self.label_dict is None and self.labels is not None:
            self.label_dict = dict(
                zip(sorted(set(self.labels)), range(len(set(self.labels))))
            )

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        x = self.texts[index]

        output_dict = self.tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        output_dict["input_ids"] = output_dict["input_ids"].flatten()
        output_dict["attention_mask"] = output_dict["attention_mask"].flatten()

        if self.labels is not None:
            y = self.labels[index]
            y_encoded = (
                torch.Tensor([self.label_dict.get(y, -1)]).long().squeeze(0)
            )
            output_dict["targets"] = y_encoded

        # output_dict['text'] = x

        return output_dict


class ImageClassificationDataset(Dataset):
    def __init__(self, df, folder, mode, transform=None):
        self.df = df
        self.folder = folder
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        item = self.df.iloc[index]
        id_ = item["id"]
        name = item.image
        label = -1
        if self.mode != "test":
            label = item.source

        image_path = self.folder / name
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return id_, image, label

    def __len__(self):
        return self.df.shape[0]
