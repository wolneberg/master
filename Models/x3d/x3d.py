import torch.nn as nn
import torch
from matplotlib import pyplot as plt

from torchvision import models
from torch.utils.data.dataloader import default_collate
import torch.optim as optim

def train_one_epoch(model, trainloader, optimizer, criterion, device):
    # det er også en metode innebygd for dette I think...
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
    epoch_acc = (train_running_correct / bs_accumuator) #regner den ut accuracy i prosent though?
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

def plot_results(train_loss, valid_loss, train_acc, valid_acc, name):
    fig, (ax1, ax2) = plt.subplots(2)

    fig.set_size_inches(18.5, 10.5)

    # Plot loss
    ax1.set_title('Loss')
    ax1.plot(train_loss, label = 'train')
    ax1.plot(valid_loss, label = 'test')
    ax1.set_ylabel('Loss')

    # Determine upper bound of y-axis
    # max_loss = max(train_loss + history.history['val_loss'])

    # ax1.set_ylim([0, np.ceil(max_loss)])
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

    plt.savefig(f'Models/x3d/results/{name}.png')

def train(device, model, num_epochs, train_loader, valid_loader, name, model_name):
    model = model.to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

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
    input = torch.randn(1, 3, 20, 256, 256)
    model.cuda()
    input = input.cuda()
    model.eval()
    torch.onnx.export(model,               # model being run
                  input,                     # model input (or a tuple for multiple inputs)
                  f"Models/x3d/model/export-{model_name}-{name}.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  verbose=True,
                #   opset_version=10,          # the ONNX version to export the model to
                #   do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

    # onnx_program = torch.onnx.dynamo_export(model, input)
    # onnx_program.save(f"Models/x3d/model/{model_name}-{name}.onnx")