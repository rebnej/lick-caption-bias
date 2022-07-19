import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.utils
import numpy as np

import transformers as tf
from transformers import BertTokenizer
from transformers import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from transformers import AdamW, get_linear_schedule_with_warmup

import torch.utils.data as data
from transformers import BertModel
from transformers import BertPreTrainedModel




class BERT_Classifier(nn.Module):

    def __init__(self, args, cls_hid_dim):

        super(BERT_Classifier, self).__init__()
        hid_size = args.hidden_dim

        mlp = []
        mlp.append(nn.BatchNorm1d(cls_hid_dim))
        mlp.append(nn.Linear(cls_hid_dim, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, 2, bias=True))

        self.mlp = nn.Sequential(*mlp)

    def forward(self, input):
        return self.mlp(input)



class BERT_GenderClassifier(nn.Module):

    def __init__(self, args, tokenizer):
        super(BERT_GenderClassifier, self).__init__()

        self.lang_model = BertModel.from_pretrained('bert-base-uncased') 
        if args.freeze_bert:
            print("***Freeze BERT***")
            for name, param in self.lang_model.named_parameters():
                param.requires_grad = False

        self.classifier = BERT_Classifier(args, 768)
        ##self.classifier = nn.Linear(768, 2)
        #self.drop = nn.Dropout(.5)
        #self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax()

        self.tokenizer = tokenizer


    def forward(self, input_ids, attention_mask, token_type_ids=None):

        """Forward

        return: logits, not probs
        """
        #inputs_embeds = self.embedding(input_ids)
        outputs = self.lang_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        #outputs = self.lang_model(attention_mask=attention_mask, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, output_hidden_states=True)

        last_hidden_states = outputs.last_hidden_state
        cls_hidden_state = last_hidden_states[:, 0, :] #(batchsize, hid_size)

        logits = self.classifier(cls_hidden_state)
        ##logits = self.fc(cls_hidden_state)

        return logits

