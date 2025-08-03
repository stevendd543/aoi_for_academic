import os
import time
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import random 
import matplotlib.pyplot as plt

def create_network(model_type='mobilenet_v2', pretrained=False):
    """
    model_type options:
    - 'mobilenet_v2': MobileNetV2 
    - 'mobilenet_v3_small': MobileNetV3 Small  
    - 'mobilenet_v3_large': MobileNetV3 Large 
    - 'efficientnet_b0': EfficientNet-B0 
    - 'shufflenet_v2_x0_5': ShuffleNetV2 0.5x 
    - 'shufflenet_v2_x1_0': ShuffleNetV2 1.0x 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )
    
    elif model_type == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=pretrained)
        model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    elif model_type == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=pretrained)
        model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier = nn.Sequential(
            nn.Linear(960, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )
    
    elif model_type == 'shufflenet_v2_x0_5':
        model = models.shufflenet_v2_x0_5(pretrained=pretrained)
        model.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        model.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )
    
    elif model_type == 'shufflenet_v2_x1_0':
        model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        model.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        model.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    elif model_type == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model_type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    return model, device

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
        return image, label

def calculate_metrics(outputs, labels, loss):
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    probas = torch.softmax(outputs, dim=1).detach().cpu().numpy()[:, 1]
    fpr, tpr, _ = roc_curve(labels, probas)
    roc_auc = auc(fpr, tpr)
    
    return {
        'loss': loss,
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': {
            'TP': tp, 'TN': tn,
            'FP': fp, 'FN': fn
        }
    }

def train(model, train_dataloader, device, optimizer, criterion, epoch, num_epochs):
    total_loss = 0
    all_outputs = []
    all_labels = []
    model.to(device)
    model.train()

    for batch_idx, (images, labels) in enumerate(tqdm(train_dataloader)):
        if args.gpu:
            images = images.to(device)
            labels = labels.to(device)

        labels = labels.squeeze()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        all_outputs.append(outputs)
        all_labels.append(labels)
        
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_outputs, all_labels, avg_loss)
    
    print(f'\nEpoch [{epoch+1}/{num_epochs}] Training Metrics:')
    print(f'Loss: {metrics["loss"]:.4f}')
    print(f'Accuracy: {metrics["accuracy"]*100:.2f}%')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall: {metrics["recall"]:.4f}')
    print(f'Specificity: {metrics["specificity"]:.4f}')
    print(f'F1-Score: {metrics["f1"]:.4f}')
    print(f'ROC AUC: {metrics["roc_auc"]:.4f}')
    print('\nConfusion Matrix:')
    print(f'TP: {metrics["confusion_matrix"]["TP"]} TN: {metrics["confusion_matrix"]["TN"]}')
    print(f'FP: {metrics["confusion_matrix"]["FP"]} FN: {metrics["confusion_matrix"]["FN"]}')
    
    return metrics

def validate(model, val_dataloader, device, criterion):
    total_loss = 0
    all_outputs = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_dataloader)):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            all_outputs.append(outputs)
            all_labels.append(labels)
    
    avg_loss = total_loss / len(val_dataloader)
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_outputs, all_labels, avg_loss)
    
    print('\nValidation Metrics:')
    print(f'Loss: {metrics["loss"]:.4f}')
    print(f'Accuracy: {metrics["accuracy"]*100:.2f}%')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall: {metrics["recall"]:.4f}')
    print(f'Specificity: {metrics["specificity"]:.4f}')
    print(f'F1-Score: {metrics["f1"]:.4f}')
    print(f'ROC AUC: {metrics["roc_auc"]:.4f}')
    print('\nConfusion Matrix:')
    print(f'TP: {metrics["confusion_matrix"]["TP"]} TN: {metrics["confusion_matrix"]["TN"]}')
    print(f'FP: {metrics["confusion_matrix"]["FP"]} FN: {metrics["confusion_matrix"]["FN"]}')
    
    return metrics

def plot_training_history(train_history, val_history, save_dir='plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    epochs = range(1, len(train_history['loss']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, val_history['loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(epochs, train_history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, val_history['accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(epochs, train_history['f1'], 'b-', label='Training F1')
    plt.plot(epochs, val_history['f1'], 'r-', label='Validation F1')
    plt.title('Training and Validation F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.legend()
    
    plt.subplot(2, 3, 4)
    plt.plot(epochs, train_history['precision'], 'b-', label='Training Precision')
    plt.plot(epochs, val_history['precision'], 'r-', label='Validation Precision')
    plt.title('Training and Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot(epochs, train_history['recall'], 'b-', label='Training Recall')
    plt.plot(epochs, val_history['recall'], 'r-', label='Validation Recall')
    plt.title('Training and Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    plt.plot(epochs, train_history['roc_auc'], 'b-', label='Training ROC AUC')
    plt.plot(epochs, val_history['roc_auc'], 'r-', label='Validation ROC AUC')
    plt.title('Training and Validation ROC AUC')
    plt.xlabel('Epochs')
    plt.ylabel('ROC AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def load_labels(file):
    n = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            image_name, label = line.split() if(len(line.strip().split())<3) else (line.strip().split()[0]+' '+line.strip().split()[1],line.strip().split()[2])
            n.append((image_name, int(label)))
    return n

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, default="label\\label_no_aug.txt")
    parser.add_argument('--dataset', type=str, default="dataset\\train")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--model', type=str, default='mobilenet_v2', 
                        choices=['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 
                                'efficientnet_b0', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'custom_lightweight'],
                        help='Model architecture to use')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    args = parser.parse_args()

    v_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.41], [0.31])
    ])
    
    class RandomRotate(object):
        def __call__(self, img):
            angle = random.choice([0, 90, 180, 270])
            return transforms.functional.rotate(img, angle)
        
    t_transforms = transforms.Compose([
        transforms.ToTensor(),
        RandomRotate(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  
        transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
        transforms.Normalize([0.41],[0.31])
    ])

    all_labels = load_labels(args.label)
    train_labels, val_labels = train_test_split(all_labels, test_size=args.val_split, random_state=42, stratify=[label for _, label in all_labels])
    
    print(f"Total samples: {len(all_labels)}")
    print(f"Training samples: {len(train_labels)}")
    print(f"Validation samples: {len(val_labels)}")
    
    train_dataset = aoi(args.dataset, train_labels, transform=t_transforms)
    val_dataset = aoi(args.dataset, val_labels, transform=v_transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    model, device = create_network(model_type=args.model, pretrained=args.resume)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,160,180], gamma=0.1)
    
    if not os.path.exists('weights'):
        os.makedirs('weights')
    
    train_history = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'specificity': [], 'f1': [], 'roc_auc': []}
    val_history = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'specificity': [], 'f1': [], 'roc_auc': []}
    
    best_val_f1 = 0
    best_metrics = None
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        train_metrics = train(model, train_dataloader, device, optimizer, criterion, epoch, args.epochs)
        val_metrics = validate(model, val_dataloader, device, criterion)
        
        for key in train_history.keys():
            train_history[key].append(train_metrics[key])
            val_history[key].append(val_metrics[key])
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_metrics = val_metrics
            torch.save(model.state_dict(), f'weights/{args.model}_best_model.pth')
            print(f"New best model saved with F1-Score: {best_val_f1:.4f}")
        
        scheduler.step()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print("\nBest Validation Metrics:")
    print(f"Accuracy: {best_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"Specificity: {best_metrics['specificity']:.4f}")
    print(f"F1-Score: {best_metrics['f1']:.4f}")
    print(f"ROC AUC: {best_metrics['roc_auc']:.4f}")
    print("\nBest Confusion Matrix:")
    cm = best_metrics['confusion_matrix']
    print(f"TP: {cm['TP']}, TN: {cm['TN']}, FP: {cm['FP']}, FN: {cm['FN']}")
    
    plot_training_history(train_history, val_history)
    
    with open(f'results_{args.model}.txt', 'w') as f:
        f.write("Training Results:\n")
        f.write(f"Total samples: {len(all_labels)}\n")
        f.write(f"Training samples: {len(train_labels)}\n")
        f.write(f"Validation samples: {len(val_labels)}\n\n")
        
        f.write("Best Validation Metrics:\n")
        for metric, value in best_metrics.items():
            if metric != 'confusion_matrix':
                f.write(f"{metric}: {value}\n")
        
        f.write("\nBest Confusion Matrix:\n")
        cm = best_metrics['confusion_matrix']
        f.write(f"TP: {cm['TP']}, TN: {cm['TN']}, FP: {cm['FP']}, FN: {cm['FN']}\n")
        
        f.write("\nTraining History:\n")
        for epoch in range(len(train_history['loss'])):
            f.write(f"Epoch {epoch+1}:\n")
            f.write(f"  Train - Loss: {train_history['loss'][epoch]:.4f}, Acc: {train_history['accuracy'][epoch]:.4f}, F1: {train_history['f1'][epoch]:.4f}\n")
            f.write(f"  Val   - Loss: {val_history['loss'][epoch]:.4f}, Acc: {val_history['accuracy'][epoch]:.4f}, F1: {val_history['f1'][epoch]:.4f}\n")