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
from logging import log
# from hatememe.logger import log


seed_everything(CFG.seed, workers=True)

cfg = CFG()


print("Torch version:", torch.__version__)

from hatememe.dataset import HMDataset

eager_transform = False

train_dataset = HMDataset(
        cfg.images_path,
        os.path.join(cfg.annotations_path,'train_updated.jsonl'),
        image_transform=cfg.image_transform,
        text_transform=cfg.text_transform,
        eager_transform=eager_transform
    )
    
val_dataset = HMDataset(
        cfg.images_path,
        os.path.join(cfg.annotations_path,'dev_unseen.jsonl'),
        image_transform=cfg.image_transform,
        text_transform=cfg.text_transform,
        eager_transform=eager_transform
    )
test_dataset = HMDataset(
        cfg.images_path,
        os.path.join(cfg.annotations_path,'test_unseen.jsonl'),
        image_transform=cfg.image_transform,
        text_transform=cfg.text_transform,
        eager_transform=eager_transform
    )


def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:
            p.grad.data = p.grad.data.float() 

device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
net = HMMLP()
print(net)
convert_models_to_fp32(net)



num_workers=30

train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=num_workers)

net = net.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*cfg.epochs)

scaler = torch.cuda.amp.GradScaler()

for epoch in range(cfg.epochs):
    running_loss = 0
    for i, data in enumerate(tqdm(train_dataloader)):
        net.train()
        images, texts, labels = data
        images, texts, labels = images.to(device), texts.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = net(images, texts)
            loss = criterion(output.squeeze(), labels.squeeze())

        scaler.scale(loss).backward()
        # convert_models_to_fp32(net.base_model)
        scaler.step(optimizer)
        scaler.update()
        # clip.model.convert_weights(net.base_model)
        
        running_loss+=loss.item()

        # torch.cuda.empty_cache()
        # gc.collect()
        if i % cfg.log_every == (cfg.log_every - 1):
            print(
                f"[Epoch {epoch + 1}, step {i+1:3d}] loss: {running_loss/cfg.log_every:.5f}"
            )
            log(level=20, msg={"phase": "train", "epoch":epoch, "step":i+1, "loss":running_loss/cfg.log_every})
            running_loss = 0.0

            # wandb.watch(net)

    scheduler.step()

    net.eval()
    running_loss = 0.0
    total_preds = 0
    preds_all_val = torch.tensor([]).cuda()
    labels_all_val = torch.tensor([]).cuda()

    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers)
    for i, data in enumerate(tqdm(val_dataloader), 0):

        images, texts, labels = data
        images = images.to(device)
        texts = texts.to(device)
        labels = labels.squeeze().to(device)

        with torch.cuda.amp.autocast():
            output = net(images, texts)
            loss = criterion(output.squeeze(), labels.squeeze())

        # loss = criterion(output.squeeze(), labels)

        running_loss += loss.item()

        # Accumulate all predictions and labels
        preds_all_val = torch.cat((preds_all_val, torch.sigmoid(output).squeeze()))
        labels_all_val = torch.cat((labels_all_val, labels))
        
    auroc = BinaryAUROC().to(device)
    auroc_score = auroc(preds_all_val, labels_all_val.int())
    accuracy = BinaryAccuracy().to(device)
    accuracy_score = accuracy(preds_all_val, labels_all_val)
    f1 = BinaryF1Score().to(device)
    f1_score = f1(preds_all_val, labels_all_val.int())
    print(
        f"[Epoch {epoch +1}, step {i+1:3d}] val loss: {running_loss/i+1:.5f} accuracy: "
        f"{accuracy_score} auroc: {auroc_score} f1_score: {f1_score}"
    )
    
    log(level=20, msg={"phase":"val", "epoch":epoch, "step":i+1, "loss":running_loss/i+1, "f1_score":f1_score, "accuracy":accuracy_score})
    preds_all_val = preds_all_val.round()

    if preds_all_val.int().sum()>0:
        print(sum(preds_all_val.int()))

