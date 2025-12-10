import torch

def predict_text(text, model, tokenizer, id_to_label, device, max_length=128):
    """Функция для предсказания темы текста"""
    model.eval()
    
    # Токенизация текста
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Предсказание
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1)
    
    # Получение метки
    predicted_label = id_to_label[prediction.item()]
    
    # Получение вероятностей
    probabilities = torch.softmax(outputs.logits, dim=1)
    
    return predicted_label, probabilities.cpu().numpy()