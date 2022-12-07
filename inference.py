import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image
from tqdm import tqdm
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryAccuracy
import wandb
from pytorch_lightning import seed_everything
import clip

from hatememe.config import CFG
from hatememe.dataset import HMDataset
from hatememe.architecture import HMMLP
# from logging import log, basicConfig
from hatememe.logger import log

cfg = CFG()

device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

def get_data_split_file(data_split):
    map = {
        "train": ["train_updated.jsonl"],
        "test-seen": ["test_seen.jsonl"],
        "test-unseen": ["test_unseen.jsonl"],
        "dev-seen": ["dev_seen.jsonl"],
        "dev-unseen": ["dev_unseen.jsonl"],
    }
    
    if data_split=="test-dev-all":
        paths = []
        for key, val in map.items():
            if key.find("test")>=0 or key.find("dev")>=0:
                paths.extend(val)

        return paths
    
    if map.get(data_split) is None: 
        raise ValueError("Invalid `data_split` provided.")

    return map[data_split]

def get_dataloader(data_split_path):
    dataset = HMDataset(
            cfg.images_path,
            data_split_path,
            image_transform=cfg.image_transform,
            text_transform=cfg.text_transform,
            eager_transform=cfg.eager_transform
        )
    
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    return dataloader


def run_loop(net, dataloader, device, split):
    net = net.to(device)
    preds_all = torch.tensor([]).cuda()
    labels_all = torch.tensor([]).cuda()

    for i, data in enumerate(tqdm(dataloader), 0):

        images, texts, labels = data
        images = images.to(device)
        texts = texts.to(device)
        labels = labels.squeeze().to(device)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = net(images, texts)

        preds_all = torch.cat((preds_all, torch.sigmoid(output).squeeze()))
        labels_all = torch.cat((labels_all, labels))
        
    auroc = BinaryAUROC().to(device)
    auroc_score = auroc(preds_all, labels_all.int())
    accuracy = BinaryAccuracy().to(device)
    accuracy_score = accuracy(preds_all, labels_all)
    f1 = BinaryF1Score().to(device)
    f1_score = f1(preds_all, labels_all.int())

    log({f"{split}_f1_score":f1_score, f"{split}_accuracy":accuracy_score, f"{split}_auroc":auroc_score})

    
if __name__=="__main__":
    net = HMMLP()
    net.load_state_dict(torch.load(cfg.model_path))
    net.eval()

    data_split_files = get_data_split_file(cfg.data_split)

    for data_split_file in data_split_files:
        data_split_path = os.path.join(cfg.annotations_path, data_split_file)
        dataloader = get_dataloader(data_split_path)
        run_loop(net, dataloader, device, data_split_file.replace(".jsonl",""))