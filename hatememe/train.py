import torch
import torch.nn as nn
from config import CFG
import wandb
from architecture import HMMLP

wandb.init(project="hatememe", entity="team-g11")

wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}

# wandb.log({"loss": loss})

# # Optional
# wandb.watch(model)

net = HMMLP()
net = net.to(CFG.device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

epochs = 10
print_every = 50

def train_fn(
    model,
    criterion,
    optimizer,
    data_loader=None,
    epochs = CFG.epochs,
    image_feature_extractor = CFG.image_feature_extractor,
    text_feature_extractor = CFG.text_feature_extractor,
):
    pass

for epoch in range(epochs):
    
    running_loss = 0
    for i, data in enumerate(train_dataloader, 0):
        images, texts, labels = data
        images = images.to(CFG.device)
        texts = texts.to(CFG.device)
        labels = labels.float().squeeze().to(CFG.device)

        with torch.no_grad():
            images = model.encode_image(images) # input_dim: batch_size x 3 x H x W; output_dim: batch_size x 512
            texts = model.encode_text(texts.squeeze()) # input_dim: batch_size x 77; output_dim: batch_size x 512
        
        fused_images_texts = torch.hstack((images,texts))
        fused_images_texts.requires_grad_()
        fused_images_texts = fused_images_texts.float()

        optimizer.zero_grad()

        # Forward pass on the fused data

        output = net(fused_images_texts)

        loss = criterion(output.squeeze(), labels)

        # Compute gradient
        loss.backward()
        # Update weight
        optimizer.step()

        running_loss += loss.item()

        if i%print_every==(print_every - 1): 
            
            print(f"[Epoch {epoch + 1}, step {i+1:3d}] loss: {running_loss/print_every:.5f}")
            running_loss = 0.0

    ## Switch to eval mode
    net.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    for i, data in enumerate(val_dataloader, 0):
        
        images, texts, labels = data
        images = images.to(CFG.device)
        texts = texts.to(CFG.device)
        labels = labels.float().squeeze().to(CFG.device)

        with torch.no_grad():
            images = model.encode_image(images) # input_dim: batch_size x 3 x H x W; output_dim: batch_size x 512
            texts = model.encode_text(texts.squeeze()) # input_dim: batch_size x 77; output_dim: batch_size x 512
        
        fused_images_texts = torch.hstack((images,texts))
        fused_images_texts.requires_grad_()
        fused_images_texts = fused_images_texts.float()

        with torch.no_grad():
            output = net(fused_images_texts)

        loss = criterion(output.squeeze(), labels)

        running_loss += loss.item()

        correct_preds += sum(F.sigmoid(output).squeeze().round()==labels)
        total_preds += len(labels)

    print(f"\n[Epoch {epoch +1}, step {i+1:3d}] val loss: {running_loss/i+1:.5f} accuracy: {correct_preds/total_preds}\n")
    net.train()

    
print("Finished Training!")
        