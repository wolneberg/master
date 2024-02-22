from torchvision import models

import torch.nn as nn

def build_model(fine_tune=True, num_classes=10):
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

if __name__ == '__main__':
    model = build_model()
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
