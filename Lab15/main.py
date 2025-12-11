import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from create_sample_dataset import create_sample_dataset
from tokenize_function import tokenize_function, tokenizer
from test_classification_dataset import TextClassificationDataset
from bert_classifier import BERTClassifier
from compute_metrics import compute_metrics
from predict_text import predict_text

# Задание 1. Настройка окужения
# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Задание 2. Подготовка датасета
# Создаем и просматриваем датасет
df = create_sample_dataset()
print(f"Размер датасета: {len(df)} примеров")
print("\nПримеры данных:")
print(df.head())
print("\nРаспределение по классам:")
print(df['label'].value_counts())

# Задание 3. Предобработка данных и токенизация
# Пример токенизации
sample_texts = df['text'].head(3)
tokenized = tokenize_function(sample_texts)

print("Пример токенизации:")
print(f"Исходный текст: {sample_texts.iloc[0]}")
print(f"Токены: {tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0])}")
print(f"Attention mask: {tokenized['attention_mask'][0]}")

# Задание 4. Создание класса Dataset
# Разделение данных на обучающую и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Создание объектов Dataset
train_dataset = TextClassificationDataset(
    train_texts.tolist(), train_labels.tolist(), tokenizer
)
test_dataset = TextClassificationDataset(
    test_texts.tolist(), test_labels.tolist(), tokenizer
)

print(f"Обучающая выборка: {len(train_dataset)} примеров")
print(f"Тестовая выборка: {len(test_dataset)} примеров")
print(f"Классы: {train_dataset.id_to_label}")

# Задание 5. Создание модели BERT для классификации
# Инициализация модели
num_classes = len(set(df['label']))
model = BERTClassifier(num_classes=num_classes)
model.to(device)

print(f"Модель загружена на {device}")
print(f"Количество классов: {num_classes}")

# Задание 6. Обучение модели
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
2
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Обучение"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(data_loader)

def eval_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Оценка"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(data_loader), all_predictions, all_labels

# Создание DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Настройка оптимизатора и планировщика
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * 20  # 5 эпох

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Обучение модели
epochs = 20
train_losses = []
test_losses = []

for epoch in range(epochs):
    print(f"\nЭпоха {epoch + 1}/{epochs}")
    
    # Обучение
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    train_losses.append(train_loss)
    
    # Оценка
    test_loss, predictions, labels = eval_epoch(model, test_loader, device)
    test_losses.append(test_loss)
    
    print(f"Потери на обучении: {train_loss:.4f}")
    print(f"Потери на тесте: {test_loss:.4f}")

# Задание 7. Оценка модели
# Вычисление метрик
metrics, pred_labels, true_labels = compute_metrics(predictions, labels, test_dataset.id_to_label)

print("\nМетрики классификации:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Матрица ошибок
def plot_confusion_matrix(true_labels, pred_labels, classes):
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Матрица ошибок')
    plt.ylabel('Истинные метки')
    plt.xlabel('Предсказанные метки')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(true_labels, pred_labels, list(test_dataset.id_to_label.values()))

# Отчет по классификации
print("\nОтчет по классификации:")
print(classification_report(true_labels, pred_labels))

# Задание 8. Использование модели для предсказаний
# Тестирование на новых текстах
# test_texts = [
#     "Баскетбольный матч закончился со счетом 98:95",
#     "Правительство объявило о новых экономических реформах",
#     "Ученые открыли новый вид динозавров в Аргентине",
#     "Вышел новый процессор с рекордной производительностью",
#     "Фильм получил награду за лучшие визуальные эффекты"
# ]

def load_texts_from_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# Загрузка текстов из файлов
economy_texts = load_texts_from_file("Lab15/economy.txt")
politics_texts = load_texts_from_file("Lab15/politics.txt")
culture_texts  = load_texts_from_file("Lab15/culture.txt")
science_texts  = load_texts_from_file("Lab15/technology.txt")
sport_texts    = load_texts_from_file("Lab15/sports.txt")

# Формируем тестовую выборку
test_texts = (
    economy_texts +
    politics_texts +
    culture_texts +
    science_texts +
    sport_texts
)

print("Предсказания для тестовых текстов:")
for text in test_texts:
    label, probs = predict_text(text, model, tokenizer, test_dataset.id_to_label, device)
    print(f"\nТекст: {text[:50]}")
    print(f"Предсказанная тема: {label}")
    print("Вероятности по классам:")
    for idx, prob in enumerate(probs[0]):
        print(f"  {test_dataset.id_to_label[idx]}: {prob:.4f}")

# Задание 9. Визуализация обучения
# Визуализация кривых обучения
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Потери на обучении', marker='o')
plt.plot(range(1, epochs + 1), test_losses, label='Потери на тесте', marker='s')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.title('Кривые обучения')
plt.legend()
plt.grid(True)
plt.show()

# Визуализация распределения предсказаний
plt.figure(figsize=(12, 5))

# Распределение истинных меток
plt.subplot(1, 2, 1)
true_counts = pd.Series(true_labels).value_counts()
plt.bar(true_counts.index, true_counts.values)
plt.title('Распределение истинных меток')
plt.xticks(rotation=45)

# Распределение предсказанных меток
plt.subplot(1, 2, 2)
pred_counts = pd.Series(pred_labels).value_counts()
plt.bar(pred_counts.index, pred_counts.values)
plt.title('Распределение предсказанных меток')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
