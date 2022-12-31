import os

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

from hatememe.config import CFG

# HOME = os.environ.get("HOME")
# images_path = f"{HOME}/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/img/"
# annotations_path = f"{HOME}/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations/"

class HMDataset(Dataset):
    def __init__(
        self,
        images_path: str,
        annotation_path: str,
        image_transform=None,
        text_transform=None,
        eager_transform=False,
        add_memotion=False

    ) -> None:

        assert annotation_path.endswith(
            ".jsonl"
        ), f"Invalid annotation file format. Format should be '.jsonl', not {annotation_path.split('.')[0]}"

        self.annotation: pd.DataFrame = pd.read_json(annotation_path, lines=True)

        if add_memotion:
            memotion_path = os.path.join(os.path.dirname(annotation_path), "label_memotion.jsonl")
            memotion_annotation: pd.DataFrame = \
                pd.read_json(memotion_path, lines=True)
            self.annotation = pd.concat([self.annotation, memotion_annotation], axis=0).reset_index(drop=True)
            self.annotation["img"] = self.annotation["img"].str.replace("img/","")
        self.images_paths = [os.path.join(images_path, self.annotation.loc[index, "img"]) for index in range(self.__len__())]

        
        self.image_transform = image_transform or (lambda image: image)
        self.text_transform = text_transform or (lambda text: text)

        self.eager_transform = eager_transform
        if self.eager_transform:
            print("Image and text transformations are being applied and this might take a while."
            " Relax... maybe go grab a cup of coffee or catchup with some friends while it runs.")

        if self.eager_transform:
            self.transformed_images = [self.image_transform(Image.open(img_path).convert("RGB")) for img_path in self.images_paths]
            self.transformed_texts = [self.text_transform(self.annotation.loc[index, "text"]) for index in range(self.__len__())]
            self.transformed_labels = [torch.tensor(self.annotation.loc[index, "label"], dtype=float) for index in range(self.__len__())]
        

    def __len__(self):

        return self.annotation.shape[0]

    def __getitem__(self, index):
        if not self.eager_transform:
            return (self.image_transform(Image.open(self.images_paths[index]).convert("RGB")),
                self.text_transform(self.annotation.loc[index, "text"]),
                torch.tensor(self.annotation.loc[index, "label"], dtype=float)
                )
        
        return self.transformed_images[index], self.transformed_texts[index], self.transformed_labels[index]

