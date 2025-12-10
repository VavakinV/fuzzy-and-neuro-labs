from transformers import BertTokenizer
from transformers import DistilBertTokenizer
from transformers import XLMRobertaTokenizer

# Загрузка токенизатора BERT
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

def tokenize_function(texts, max_length=128):
    """Функция для токенизации текстов"""
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )