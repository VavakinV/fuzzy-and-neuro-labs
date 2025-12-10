from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Функция для вычисления метрик
def compute_metrics(predictions, labels, id_to_label):
    # Преобразуем ID обратно в метки
    pred_labels = [id_to_label[pred] for pred in predictions]
    true_labels = [id_to_label[label] for label in labels]
    
    # Вычисляем метрики
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }, pred_labels, true_labels