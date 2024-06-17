import torch.nn as nn
import torch
from matplotlib import pyplot as plt

from torchvision import models
from torch.utils.data.dataloader import default_collate
import torch.optim as optim

def build_model(fine_tune=True, num_classes=100, trainable_layers=0):
    model = models.video.s3d(weights='DEFAULT')
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    if not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    if trainable_layers > 0:
        for params in model.features[-trainable_layers:].parameters():
            params.requires_grad = True
    model.classifier[1] = nn.Conv3d(1024, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    return model

def collate_fn(batch):
    batch = [(d[0], d[1]) for d in batch]
    return default_collate(batch)


def train_one_epoch(model, trainloader, optimizer, criterion, device):
    # https://github.com/pytorch/vision/blob/main/references/video_classification/train.py 
    model = model.to(device)
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0.0
    bs_accumuator = 0
    counter = 0
    for data in trainloader:
        counter += 1
        image, labels = data
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
    epoch_acc = (train_running_correct / bs_accumuator)
    return epoch_loss, epoch_acc

def validate(model, testloader, criterion, device):
    model = model.to(device)
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct = 0.0
    bs_accumuator = 0
    counter = 0
    with torch.no_grad():
        for data in testloader:
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
    epoch_acc = (valid_running_correct / bs_accumuator)
    return epoch_loss, epoch_acc
    
#Plots
def plot_results(train_loss, valid_loss, train_acc, valid_acc, name):
    fig, (ax1, ax2) = plt.subplots(2)

    fig.set_size_inches(18.5, 10.5)

    # Plot loss
    ax1.set_title('Loss')
    ax1.plot(train_loss, label = 'train')
    ax1.plot(valid_loss, label = 'test')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation']) 

    # Plot accuracy
    ax2.set_title('Accuracy')
    ax2.plot(train_acc,  label = 'train')
    ax2.plot(valid_acc, label = 'test')
    ax2.set_ylim(0,1)
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])

    plt.savefig(f'Models/S3D/results/{name}.png')


# Training loop.
def train(num_epochs, num_classes, train_loader, valid_loader, device, fine_tune, name, trainable_layers=0):
    model = build_model(fine_tune=fine_tune, num_classes=num_classes, trainable_layers=trainable_layers)
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    print(f"trainable layers: {trainable_layers}")
    print(f"Optimizer: {optimizer}")
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    for epoch in range(num_epochs):
        print(f"[INFO]: Epoch {epoch+1} of {num_epochs}")
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
    
    plot_results(train_loss, valid_loss, train_acc, valid_acc, name)

    #Save model
    torch_input = torch.randn(epoch, 3, 172, 172)
    onnx_program = torch.onnx.dynamo_export(model, torch_input)
    onnx_program.save(f"s3d-{name}.onnx")