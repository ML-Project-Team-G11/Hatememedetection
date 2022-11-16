import os

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

from config import CFG

HOME = os.environ.get("HOME")
images_path = f"{HOME}/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/img/"
annotations_path = f"{HOME}/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations/"

class HMDataset(Dataset):
    def __init__(self, images_path: str, annotation_path: str, image_transform=None, text_transform=None) -> None:
        self.images_path = images_path
        self.annotation_path = annotation_path
        self.image_transform = image_transform
        self.text_transform = text_transform
        assert self.annotation_path.endswith(".jsonl"), f"Invalid annotation file format. Format should be '.jsonl', not {self.annotation_path.split('.')[0]}"
        self.annotation: pd.DataFrame = pd.read_json(self.annotation_path, lines=True)      

    def __len__(self):
        return self.annotation.shape[0]

    def __getitem__(self, index):
        img_path = os.path.join(self.images_path, self.annotation.loc[index,"img"])
        image = Image.open(img_path).convert("RGB")
        text = self.annotation.loc[index,"text"]
        label = self.annotation.loc[index,"label"]
        if self.image_transform:
            image = self.image_transform(image)
        if self.text_transform:
            text = self.text_transform(text)
        return image, text, torch.tensor(label)



def create_dataloader(
        images_path: str = CFG.images_path,
        annotation_path: str = f"{CFG.annotations_path}/train_updated.jsonl",
        batch_size: int = CFG.batch_size,
        image_transform = None,
        text_transform = None,
        shuffle = True,
    ):
    """Function to create dataloader given image path and absolute path for one of
    train.jsonl, dev_unseen.jsonl and test_unseen.jsonl

    images_path (str): path to the folder containing all the images
    annotation_path (str): path to a jsonl file containing the data corresponding 
        to train, test or dev
    batch_size (int): batch size for loading the data
    image_transform (function): A function to use to transform the image data
    text_transform (function): A function to use to transform the text data
    shuffle (bool): To shuffle the data or not

    
    """

    hm_dataset = HMDataset(
        images_path, 
        annotation_path, 
        image_transform=image_transform, 
        text_transform=text_transform
    )

    hm_dataloader = DataLoader(
        hm_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return hm_dataloader