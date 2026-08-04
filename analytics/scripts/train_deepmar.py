from typing import Tuple
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from PIL import Image
import os
import albumentations as alb
import numpy as np

from tqdm import tqdm
from level1.common_classifier.model.resnet import resnet18, resnet34, resnet50, resnet101
#from torchvision.models import resnet18, resnet34, resnet50, resnet101

from general import proj


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.annotations = self.read_annotations()
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path, labels = self.annotations[idx]
        image = Image.open(os.path.join(self.image_dir, img_path)).convert("RGB")
        image = np.asarray(image)
        
        if self.transform:
            image = self.transform(image=image)["image"]

        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)  # Convert image to tensor
        labels = torch.tensor(labels, dtype=torch.float32)  # Convert labels to tensor
        return image, labels

    def read_annotations(self):
        # itreate over all files in subfolders, the class is the dir name
        annotations = []
        for c in os.listdir(self.image_dir):
            #ann [visible_hand, has_glove, transparent=0/blue=1]
            if c == 'no_hand':
                ann = [0, 0, 0]
            elif c == 'hand_no_glove':
                ann = [1, 0, 0]
            elif c == 'hand_transparent_glove':
                ann = [1, 1, 0]
            elif c == 'hand_blue_glove':
                ann = [1, 1, 1]
            else:
                raise ValueError(f"Unknown class: {c}")
            
            for f in os.listdir(os.path.join(self.image_dir, c)):
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    annotations.append((os.path.join(c, f), ann))
        return annotations

class DeepMar(nn.Module):
    def __init__(self, num_att: int, im_size: Tuple[int, int], has_calibration: bool = False, weights: str = "resnet18"):
        super(DeepMar, self).__init__()
        # init the necessary parameter for network structure
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.drop_pool5: bool = True
        self.drop_pool5_rate: float = 0.5
        self.use_calibration: bool = True
        self.last_conv_stride: int = 2
        self.image_size = im_size
        last_layer = 0
        if "resnet18" in weights:
            self.base = resnet18(pretrained=False, last_conv_stride=self.last_conv_stride)
            last_layer = 512
        elif "resnet34" in weights:
            self.base = resnet34(pretrained=False, last_conv_stride=self.last_conv_stride)
            last_layer = 512
        elif "resnet101" in weights:
            self.base = resnet101(pretrained=False, last_conv_stride=self.last_conv_stride)
        elif "resnext50_32x4d" in weights:
            from torchvision.models import resnext50_32x4d

            self.base = resnext50_32x4d(pretrained=False)
            self.base = torch.nn.Sequential(*list(self.base.children())[:-1])
        elif "efficientnet_b4" in weights:
            from torchvision.models import efficientnet_b4

            self.base = efficientnet_b4(pretrained=False)
            self.base = torch.nn.Sequential(*list(self.base.children())[:-1])
            last_layer = 1792
        else:  # default assume resnet50
            self.base = resnet50(pretrained=False, last_conv_stride=self.last_conv_stride)
            last_layer = 2048

        self.classifier = nn.Linear(last_layer, num_att)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.constant_(self.classifier.bias, 0)

        if has_calibration:
            self.calib_w = nn.Parameter(torch.ones((1, num_att), requires_grad=False), requires_grad=False)
            self.calib_b = nn.Parameter(torch.zeros((1, num_att), requires_grad=False), requires_grad=False)
            self.calibrated = nn.Parameter(torch.tensor(False), requires_grad=False)
        else:
            self.calibrated = False

    def warmup(self, device, half: bool = False):
        b_size = 2  # batch size warmup
        runs = 5  # warmup runs

        print("Warming up...")
        warmup_image = torch.zeros(b_size, 3, self.image_size[0], self.image_size[1]).to(device)
        if half:
            warmup_image = warmup_image.half()
        for i in range(runs):
            self.forward(warmup_image)

    def forward(self, x):
        x = self.base(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        if self.drop_pool5:
            x = nn.functional.dropout(x, p=self.drop_pool5_rate, training=self.training)
        x = self.classifier(x)

        if self.calibrated and self.use_calibration:
            x = x * self.calib_w + self.calib_b
        return x
    
def train_epoch(model, dataloader, criterion, optimizer, epoch, device, writer):
    model.train()  # Set model to training mode

    losses_all, losses_hands, losses_gloves, losses_colors = 0, 0, 0, 0
    acc_hands = 0
    acc_gloves = 0
    acc_colors = 0
    pbar = tqdm(dataloader, total=len(dataloader))
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        #calculate the hand loss
        loss_hands = criterion(outputs[:, 0], labels[:, 0])
        # calculate the glove loss only on images with hands
        hands_mask = labels[:, 0] == 1
        loss_gloves = criterion(outputs[hands_mask, 1], labels[hands_mask, 1])
        # calculate the color loss only on images with gloves
        gloves_maks = labels[:, 1] == 1
        loss_colors = criterion(outputs[gloves_maks, 2], labels[gloves_maks, 2])

        loss = loss_hands + loss_gloves + loss_colors

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_all += loss.item() * inputs.size(0)
        losses_hands += loss_hands.item() * inputs.size(0)
        losses_gloves += loss_gloves.item() * inputs.size(0)
        losses_colors += loss_colors.item() * inputs.size(0)
        acc_hands += ((outputs[:, 0]>0) == labels[:, 0]).sum().item()
        acc_gloves += ((outputs[:, 1]>0) == labels[:, 1]).sum().item()
        acc_colors += ((outputs[:, 2]>0) == labels[:, 2]).sum().item()

    epoch_loss = losses_all / len(dataloader.dataset)
    loss_hands = losses_hands / len(dataloader.dataset)
    loss_gloves = losses_gloves / len(dataloader.dataset)
    loss_colors = losses_colors / len(dataloader.dataset)
    acc_hands /= len(dataloader.dataset)
    acc_gloves /= len(dataloader.dataset)
    acc_colors /= len(dataloader.dataset)
    writer.add_scalar('Loss/train_all', epoch_loss, epoch)
    writer.add_scalar('Loss/train_hands', loss_hands, epoch)
    writer.add_scalar('Loss/train_gloves', loss_gloves, epoch)
    writer.add_scalar('Loss/train_colors', loss_colors, epoch)
    writer.add_scalar('Accuracy_train/hands', acc_hands, epoch)
    writer.add_scalar('Accuracy_train/gloves', acc_gloves, epoch)
    writer.add_scalar('Accuracy_train/colors', acc_colors, epoch)
    return epoch_loss

@torch.no_grad()
def val_epoch(model, dataloader, criterion, epoch, device, writer=None, test=False):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    acc_hands = 0
    acc_gloves = 0
    acc_colors = 0

    pbar = tqdm(dataloader, total=len(dataloader))
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        acc_hands += ((outputs[:, 0]>0) == labels[:, 0]).sum().item()
        acc_gloves += ((outputs[:, 1]>0) == labels[:, 1]).sum().item()
        acc_colors += ((outputs[:, 2]>0) == labels[:, 2]).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    acc_hands /= len(dataloader.dataset)
    acc_gloves /= len(dataloader.dataset)
    acc_colors /= len(dataloader.dataset)
    if writer is not None:  
        split = 'test' if test else 'val'
        writer.add_scalar(f'Loss/{split}', epoch_loss, epoch)
        writer.add_scalar(f'Accuracy_{split}/hands', acc_hands, epoch)
        writer.add_scalar(f'Accuracy_{split}/gloves', acc_gloves, epoch)
        writer.add_scalar(f'Accuracy_{split}/colors', acc_colors, epoch)
    if test:
        print(f"Test Loss: {epoch_loss:.4f}")
        print(f"Test Accuracy Hands: {acc_hands:.4f}")
        print(f"Test Accuracy Gloves: {acc_gloves:.4f}")
        print(f"Test Accuracy Colors: {acc_colors:.4f}")
    return epoch_loss

# Data transforms
image_size = 192
data_transforms2 = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.Resize((image_size, image_size)),
        # ResizeOrPad((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([ 
        transforms.Resize((image_size, image_size)),
        # ResizeOrPad((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

interp_mode = cv2.INTER_AREA 
val_transforms = alb.Compose(
    [
        alb.Resize(image_size, image_size, interpolation=interp_mode),
        alb.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
train_transforms = alb.Compose(
    [
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.3),
        alb.Rotate(limit=30, p=0.5),
        alb.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        alb.GaussianBlur(blur_limit=(5, 9), p=0.5),
    ]
    + val_transforms.transforms
)
data_transforms = {
    'train': train_transforms,
    'val': val_transforms,
}

# Main function
def main(only_test=False):
    image_dir = '/mnt/d/hands1'
    model_name = 'resnet18'  # Choose between 'resnet34', 'resnet50', 'resnet101', 'efficientnet_b0', etc.
    batch_size = 64
    num_workers = 16
    num_epochs = 60
    learning_rate = 1e-4
    wd = 0
    exp_dir = proj.local_weights_path("hands")
    #create tensorboard writer

    # Create dataset and dataloaders
    datasets = {
#        'train': CustomImageDataset(os.path.join(image_dir, 'train'), transform=data_transforms['train']),
#        'val': CustomImageDataset(os.path.join(image_dir, 'val'), transform=data_transforms['val']),
        'test': CustomImageDataset(image_dir, transform=data_transforms['val']),
    }

    dataloaders = {
#        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
#        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepMar(num_att=3, im_size=(image_size, image_size), has_calibration=False, weights=model_name)

    model = model.to(device)
    if only_test:
        model.load_state_dict(torch.load(os.path.join(exp_dir, 'hands_resnet18_0_1.pt')))
        model.eval()
        test_loss = val_epoch(model, dataloaders['test'], nn.BCEWithLogitsLoss(), 0, device, None, test=True)
        return
    if not only_test:
        writer = SummaryWriter(exp_dir)
    else:
        writer = None
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

    # Train the model
    best_val_loss = 1e9
    best_model = None
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, dataloaders['train'], criterion, optimizer, epoch, device, writer)
        val_loss = val_epoch(model, dataloaders['val'], criterion, epoch, device, writer)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            torch.save(best_model, f'{exp_dir}/best_model.pth')
        if epoch == 20 or epoch == 40: # reduce lr by 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
    
    model.load_state_dict(best_model)
    print(f'Starting validation on {exp_dir}')
    test_loss = val_epoch(model, dataloaders['test'], criterion, 0, device, writer, test=True)
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == '__main__':
    main(True)
