import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import os
from sklearn.metrics import f1_score, average_precision_score, classification_report  # Добавлены новые метрики

# Конфигурация
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
DATA_PATH = '.'  # Путь к корневой папке с данными
MODEL_PATH = 'best_model2.pth'  # Путь к сохранённой модели
IMAGE_SIZE = 300

# Преобразования для тестовых данных
test_transform = transforms.Compose([
        transforms.Resize(380),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Загрузка тестового набора (финальная проверка)
validation_dataset = ImageFolder(root=os.path.join(DATA_PATH, 'Test'), transform=test_transform)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Загрузка модели
model = models.efficientnet_b3(pretrained=False)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=num_ftrs, out_features=2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Инициализация переменных для метрик
all_preds = []
all_labels = []
all_probs = []  # Для вычисления mAP

# Запуск тестирования
with torch.no_grad():
    for images, labels in tqdm(validation_loader, desc='Final Testing'):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)  # Преобразуем в вероятности
        _, predicted = torch.max(outputs.data, 1)
        
        # Сохраняем данные для метрик
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probabilities[:, 1].cpu().numpy())  # Вероятности класса 1 (fake)

# Вычисляем метрики
final_acc = 100 * sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
f1 = f1_score(all_labels, all_preds, average='binary')  # F1 для бинарной классификации
ap = average_precision_score(all_labels, all_probs)     # Average Precision (AP)

# Вывод результатов
print(f'\nFinal Test Accuracy: {final_acc:.2f}%')
print(f'F1 Score: {f1:.4f}')
print(f'Average Precision (AP): {ap:.4f}')

# Детальный отчёт с precision, recall для каждого класса
print('\nClassification Report:')
print(classification_report(all_labels, all_preds, target_names=validation_dataset.classes))