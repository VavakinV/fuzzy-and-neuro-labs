from transformers import BertForSequenceClassification
import torch.nn as nn

from transformers import DistilBertForSequenceClassification
from transformers import XLMRobertaForSequenceClassification


## BERT
# class BERTClassifier(nn.Module):
#     def __init__(self, num_classes, model_name='bert-base-multilingual-cased'):
#         super(BERTClassifier, self).__init__()
#         self.bert = BertForSequenceClassification.from_pretrained(
#             model_name,
#             num_labels=num_classes
#         )
    
#     def forward(self, input_ids, attention_mask, labels=None):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels
#         )
#         return outputs

## DistilBERT
# class BERTClassifier(nn.Module):
#     def __init__(self, num_classes, model_name='distilbert-base-multilingual-cased'):
#         super().__init__()
#         self.model = DistilBertForSequenceClassification.from_pretrained(
#             model_name,
#             num_labels=num_classes
#         )

#     def forward(self, input_ids, attention_mask, labels=None):
#         return self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels
#         )

# XLM-RoBERTa  
class BERTClassifier(nn.Module):
    def __init__(self, num_classes, model_name='xlm-roberta-base'):
        super().__init__()
        self.model = XLMRobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

