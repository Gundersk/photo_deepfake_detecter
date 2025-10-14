import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import os
from PIL import Image, ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SIZE = 300  

def safe_loader(path: str) -> Image.Image:
    """Безопасная загрузка изображений с обработкой ошибок"""
    try:
        img = Image.open(path).convert('RGB')
        img.load()
        return img
    except Exception as e:
        print(f"Удален поврежденный файл: {path}")
        return Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))

def main():
    device = torch_directml.device()
    print(f"Используемое устройство: {torch_directml.device_name(0)}")

    config = {
        'batch_size': 12,          
        'num_epochs': 40,
        'learning_rate': 2e-5,
        'weight_decay': 1e-5,
        'patience': 5
    }

    train_transform = transforms.Compose([
        transforms.Resize(380),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(380),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def load_dataset(path):
        return ImageFolder(
            root=path,
            transform=train_transform if 'Train' in path else val_transform,
            loader=safe_loader
    )

    train_dataset = load_dataset('Train')
    val_dataset = load_dataset('Val')
    test_dataset = load_dataset('Test')  

    train_path = 'Train'

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Папка {train_path} не найдена!")

    for class_name in ['real', 'fake']:
        class_dir = os.path.join(train_path, class_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Папка класса {class_dir} не существует!")
        if len(os.listdir(class_dir)) == 0:
            raise ValueError(f"Папка {class_dir} пуста!")

    class_counts = np.array([
        len(os.listdir(os.path.join(train_path, 'real'))),
        len(os.listdir(os.path.join(train_path, 'fake')))
    ], dtype=np.int64)

    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32, device=device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=WeightedRandomSampler(
            weights=class_weights[train_dataset.targets],
            num_samples=len(train_dataset),
            replacement=True
        ),
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features, 2)
    )
    model = model.to(device)

    def unfreeze_layers(epoch):
        layers = list(model.children())[0]
        if epoch >= 10:
            for param in layers[5:].parameters():
                param.requires_grad = True
        if epoch >= 20:
            for param in layers[3:].parameters():
                param.requires_grad = True

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), 
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=2,
        factor=0.5
    )

    best_acc = 0.0
    no_improve = 0
    
    for epoch in range(config['num_epochs']):
        unfreeze_layers(epoch)
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_loss += loss.item()


        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()

 
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / len(val_dataset)
        scheduler.step(val_acc)

        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f} | Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Сохранение модели
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Модель сохранена с точностью {val_acc:.2f}%")
        else:
            no_improve += 1
            if no_improve >= config['patience']:
                print(f"Ранняя остановка на эпохе {epoch+1}")
                break

    print(f"\nЛучшая точность: {best_acc:.2f}%")

if __name__ == '__main__':
    main()