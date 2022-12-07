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

# basicConfig(level=20)

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
        eager_transform=eager_transform,
        add_memotion=cfg.add_memotion,
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
val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers)

net = net.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*cfg.epochs)

scaler = torch.cuda.amp.GradScaler()

best_auroc = 0
best_acc = 0

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
            log({"epoch":epoch, "train_loss":running_loss/cfg.log_every})
            running_loss = 0.0

            wandb.watch(net)

    scheduler.step()

    net.eval()
    running_loss = 0.0
    total_preds = 0
    preds_all_val = torch.tensor([]).cuda()
    labels_all_val = torch.tensor([]).cuda()

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
    if auroc_score > best_auroc:
        torch.save(net.state_dict(), cfg.model_path)
        best_auroc = auroc_score
        best_acc = accuracy_score
        best_f1_score = f1_score
    print(
        f"[Epoch {epoch +1}, step {i+1:3d}] val loss: {running_loss/i+1:.5f} accuracy: "
        f"{accuracy_score} auroc: {auroc_score} f1_score: {f1_score}"
    )
    
    log({"val_loss":running_loss/i+1, "val_f1_score":f1_score, "val_accuracy":accuracy_score, "val_auroc":auroc_score})

log({"val_loss":running_loss/i+1, "val_f1_score":best_f1_score, "val_accuracy":best_acc, "val_auroc":best_auroc})

torch.cuda.empty_cache()
gc.collect()
del net

###################
# Testing loop 
###################

# Load the best model
net = HMMLP()
net.load_state_dict(torch.load(cfg.model_path))
net = net.cuda()
net.eval()

preds_all_val = torch.tensor([]).cuda()
labels_all_val = torch.tensor([]).cuda()

for i, data in enumerate(tqdm(test_dataloader), 0):

    images, texts, labels = data
    images = images.to(device)
    texts = texts.to(device)
    labels = labels.squeeze().to(device)

    with torch.cuda.amp.autocast():
        output = net(images, texts)
        # loss = criterion(output.squeeze(), labels.squeeze())

    preds_all_val = torch.cat((preds_all_val, torch.sigmoid(output).squeeze()))
    labels_all_val = torch.cat((labels_all_val, labels))
    
wandb.watch(net)
auroc = BinaryAUROC().to(device)
auroc_score = auroc(preds_all_val, labels_all_val.int())
accuracy = BinaryAccuracy().to(device)
accuracy_score = accuracy(preds_all_val, labels_all_val)
f1 = BinaryF1Score().to(device)
f1_score = f1(preds_all_val, labels_all_val.int())

log({"test_f1_score":f1_score, "test_accuracy":accuracy_score, "test_auroc":auroc_score})