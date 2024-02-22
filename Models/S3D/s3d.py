import torch.nn as nn
import torch

from torchvision import models
from torch.utils.data.dataloader import default_collate
import torch.optim as optim

def build_model(fine_tune=True, num_classes=100):
    model = models.video.s3d(weights='DEFAULT')
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    if not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    model.classifier[1] = nn.Conv3d(1024, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    return model

def collate_fn(batch):
    batch = [(d[0], d[1]) for d in batch]
    return default_collate(batch)


def train_one_epoch(model, train_set, optimizer, loss_function):
    model.train()
 
    optimizer.zero_grad()
    # Forward pass.
    outputs = model(inputs)
    # Calculate the loss.
    loss = loss_function(outputs, labels)
    train_running_loss += loss.item()

    # Backpropagation.
    loss.backward()
    # Update the weights.
    optimizer.step()
    return loss
# Training loop.
def train(model, num_epochs, train_set):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
           print(f"[INFO]: Epoch {epoch+1} of {num_epochs}")
           train_one_epoch(model, train_set, optimizer, loss_function)