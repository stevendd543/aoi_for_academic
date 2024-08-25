import os
import time
import argparse

import torch
import torch.nn as nn
import torchvision.models as models

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
writer = SummaryWriter('./runs/test3')
def create_network(pretrained = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet34()
    model.train()
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 2)  
    )

    return model,device
class aoi(Dataset):
    def __init__(self, image_dir, label_list, transform = None):
        self.image_dir = image_dir
        self.labels = label_list
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_name, label = self.labels[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('L')

        if(self.transform):
            image = self.transform(image)
        return image,label

train_loss =[]
val_loss = []   
best_train_loss = 9999
best_val_loss = 9999

def train(epoch):
    loss = 0
    total_loss = 0
    acc = 0
    n_data = 0
    model.to(device)
    model.train()

    for batch_idx,(images, labels) in enumerate(tqdm(train_dataloader)):
        if args.gpu:
            images = images.to(device)
            labels = labels.to(device)

        labels = labels.squeeze()
        n_data += labels.size(0)
        optimizer.zero_grad()
        outputs = model(images)
        _, preds = torch.max(outputs, 1) 
        loss = criterion(outputs, labels)

        acc += (preds == labels).sum().item()
        total_loss += loss.item() 

        writer.add_scalar('Loss/batch', loss.item(), epoch * len(train_dataloader) + batch_idx)        
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    train_acc = 100 * acc / n_data
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    torch.save(model,"weights/all3.pth") # SAVE MODEL
    print(f'Epoch [{epoch+1}/{num_epochs}], TRAIN Loss: {total_loss/len(train_dataloader)}," Accuracy: {100* acc/n_data}')


def validate(epoch):
    acc = 0
    total_loss = 0
    n_data = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_dataloader)):
            images = images.to(device)
            images = images.view(images.shape[0], 1, 200, 200)
            labels = labels.to(device)
            n_data += images.shape[0]

            outputs = model(images)
            _, preds = torch.max(outputs, 1)  
            acc += (preds == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            writer.add_scalar('Loss/val_batch', loss.item(), epoch * len(val_dataloader) + batch_idx)
    
    avg_loss = total_loss / len(val_dataloader)
    val_acc = 100 * acc / n_data
    writer.add_scalar('Loss/val_epoch', avg_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {total_loss/len(val_dataloader)} Accuracy: {100* acc/n_data}')

def split_label(file, test_rate = 0.1):
    n = []
    labels = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            image_name, label = line.split() if(len(line.strip().split())<3) else (line.strip().split()[0]+' '+line.strip().split()[1],line.strip().split()[2])
            # names.append(image_name)
            labels.append(int(label))
            n.append((image_name, int(label)))

    train_list, test_list = train_test_split(n,stratify=labels, test_size=test_rate, random_state=1111)
    return train_list, test_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label',
                        type=str,
                        default="C:label\\label_no_aug.txt")
    parser.add_argument('--dataset',
                        type=str,
                        default="C:dataset\\train")
    parser.add_argument('--batch_size',
                        type=int,
                        default=128)
    parser.add_argument('--test_size',
                        type=int,
                        default=0.05)
    parser.add_argument('--epochs',
                        type=int,
                        default=200)
    parser.add_argument('--gpu',
                        type=bool,
                        default=True)
    parser.add_argument('--resume',
                        type=bool,
                        default=False)
    args = parser.parse_args()
    class GaussianNoise(object):
        def __init__(self, mean=0., std=0.1):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
    v_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.41], [0.31])
    ])
    t_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(40),  
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  
        transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),  # 顏色抖動
        transforms.Normalize([0.41],[0.31])
    ])
    image_dir = args.dataset
    label_file = args.label
    batch_size = args.batch_size
    test_size = args.test_size
    num_epochs = args.epochs
    resume = args.resume

    train_labels, test_labels = split_label(file = label_file, test_rate = 0.01)
    train_dataset = aoi(image_dir, train_labels, transform=t_transforms)  
    val_dataset = aoi(image_dir, test_labels, transform=v_transform)  
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model,device = create_network(pretrained=resume)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,160,180], gamma=0.1)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=num_epochs)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                         mode='min',
    #                                                         factor=0.5,
    #                                                         patience=8,
    #                                                         verbose=True,
    #                                                         min_lr=0.00001)

    for epoch in range(num_epochs):
        train(epoch)
        validate(epoch)
        scheduler.step()
    writer.close()
