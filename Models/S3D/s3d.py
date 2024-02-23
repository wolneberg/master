import torch.nn as nn
import torch
from tqdm.auto import tqdm

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


def train_one_epoch(model, trainloader, optimizer, criterion, device):
    # det er også en metode innebygd for dette I think...
    # https://github.com/pytorch/vision/blob/main/references/video_classification/train.py 

    # model.train()
 
    # optimizer.zero_grad()
    # # Forward pass.
    # outputs = model(inputs)
    # # Calculate the loss.
    # loss = loss_function(outputs, labels)
    # train_running_loss += loss.item()

    # # Backpropagation.
    # loss.backward()
    # # Update the weights.
    # optimizer.step()
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    bs_accumuator = 0
    counter = 0
    prog_bar = tqdm(
        trainloader, 
        total=len(trainloader), 
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    for i, data in enumerate(prog_bar):
        counter += 1
        image, labels = data
        print(image)
        print('labels: ', labels)
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        bs_accumuator += outputs.shape[0]
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / bs_accumuator)
    return epoch_loss, epoch_acc

def validate(model, testloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    bs_accumuator = 0
    counter = 0
    prog_bar = tqdm(
        testloader, 
        total=len(testloader), 
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    with torch.no_grad():
        for i, data in enumerate(prog_bar):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            bs_accumuator += outputs.shape[0]
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / bs_accumuator)
    return epoch_loss, epoch_acc
    
# Training loop.
def train(num_epochs, num_classes, train_loader, valid_loader, device):
    model = build_model(fine_tune=True, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    for epoch in range(num_epochs):
        print(f"[INFO]: Epoch {epoch+1} of {num_epochs}")
        # train_one_epoch(model, train_set, optimizer, loss_function)
        train_epoch_loss, train_epoch_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, valid_loader, criterion, device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")