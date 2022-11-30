import os

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

from hatememe.config import CFG

HOME = os.environ.get("HOME")
images_path = f"{HOME}/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/img/"
annotations_path = f"{HOME}/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations/"

class HMDataset(Dataset):
    def __init__(
        self,
        images_path: str,
        annotation_path: str,
        image_transform=None,
        text_transform=None,
        eager_transform=False,

    ) -> None:

        assert annotation_path.endswith(
            ".jsonl"
        ), f"Invalid annotation file format. Format should be '.jsonl', not {annotation_path.split('.')[0]}"

        self.annotation: pd.DataFrame = pd.read_json(annotation_path, lines=True)

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

if __name__=="__main__":
    CFG = CFG()
    # # print(CFG.text_transform("Hello, world!"))
    # train_dataset = HMDataset(
    #     images_path,
    #     os.path.join(annotations_path,'train_updated.jsonl'),
    #     image_transform=CFG.image_transform,
    #     text_transform=CFG.text_transform,
    #     eager_transform=True
    # )
    # train_dataloader = DataLoader(train_dataset, 4, shuffle=True)
    # import time
    # start = time.time()
    # for n, data in enumerate(train_dataloader):
    #     print(n)
    #     i, t, l = data
    #     end = time.time()
    #     print("Duration:", end - start)
    #     start = end

    #     if n > 5:
    #         break