import argparse
import pickle
import nltk
import numpy as np
import json
import os
import pprint
#from pycocotools.coco import COCO
#import pylab
from nltk.tokenize import word_tokenize
import random
from io import open
import sys
import torch
from torch import nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange



        
class BERT_ANN_leak_data(data.Dataset):
    def __init__(self, d_train, d_test, args, gender_task_mw_entries, gender_words, tokenizer, max_seq_length, split, caption_ind=None):
        self.task = args.task
        self.gender_task_mw_entries = gender_task_mw_entries
        self.cap_ind = caption_ind
        self.split = split
        self.gender_words = gender_words

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.d_train, self.d_test = d_train, d_test 

        self.align_vocab = args.align_vocab
        if self.align_vocab:
            self.model_vocab = pickle.load(open('./bias_data/model_vocab/%s_vocab.pkl' %args.cap_model, 'rb'))
            print('len(self.model_vocab):', len(self.model_vocab))

    def __len__(self):
        if self.split == 'train':
            return len(self.d_train)
        else:
            return len(self.d_test)

    def __getitem__(self, index):
        if self.split == 'train':
            entries = self.d_train
        else:
            entries = self.d_test
        entry = entries[index]
        img_id = entry['img_id']

        gender = entry['bb_gender']
        if gender == 'Male':
            gender_target = torch.tensor(0)
        elif gender == 'Female':
            gender_target = torch.tensor(1)

        if self.task == 'captioning':
            ctokens = word_tokenize(entry['caption_list'][self.cap_ind].lower())
            new_list = []
            for t in ctokens:
                if t in self.gender_words:
                    new_list.append('[MASK]')
                elif self.align_vocab:
                    if t not in self.model_vocab:
                        new_list.append('[UNK]')
                    else:
                        new_list.append(t)
                else:
                    new_list.append(t)
            new_sent = ' '.join([c for c in new_list])

            encoded_dict = self.tokenizer.encode_plus(new_sent, add_special_tokens=True, truncation=True, max_length=self.max_seq_length, 
                                                    padding='max_length', return_attention_mask=True, return_tensors='pt')


        input_ids = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        token_type_ids = encoded_dict['token_type_ids']
        token_type_ids = token_type_ids.view(self.max_seq_length)

        input_ids = input_ids.view(self.max_seq_length)
        attention_mask = attention_mask.view(self.max_seq_length)

        return input_ids, attention_mask, token_type_ids, gender_target, img_id



class BERT_MODEL_leak_data(data.Dataset):
    def __init__(self, d_train, d_test, args, gender_task_mw_entries, gender_words, tokenizer, max_seq_length, split):
        self.task = args.task
        self.gender_task_mw_entries = gender_task_mw_entries
        self.split = split
        self.gender_words = gender_words

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.d_train, self.d_test = d_train, d_test 

    def __len__(self):
        if self.split == 'train':
            return len(self.d_train)
        else:
            return len(self.d_test)

    def __getitem__(self, index):
        if self.split == 'train':
            entries = self.d_train
        else:
            entries = self.d_test
        entry = entries[index]
        img_id = entry['img_id']

        gender = entry['bb_gender']
        if gender == 'Male':
            gender_target = torch.tensor(0)
        elif gender == 'Female':
            gender_target = torch.tensor(1)

        if self.task == 'captioning':
            c_pred_tokens = word_tokenize(entry['pred'].lower())
            new_list = []
            for t in c_pred_tokens:
                if t in self.gender_words:
                    new_list.append('[MASK]')
                else:
                    new_list.append(t)
            new_sent = ' '.join([c for c in new_list])

            encoded_dict = self.tokenizer.encode_plus(new_sent, add_special_tokens=True, truncation=True, max_length=self.max_seq_length,
                                                    padding='max_length', return_attention_mask=True, return_tensors='pt')


        input_ids = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        token_type_ids = encoded_dict['token_type_ids']
        token_type_ids = token_type_ids.view(self.max_seq_length)

        input_ids = input_ids.view(self.max_seq_length)
        attention_mask = attention_mask.view(self.max_seq_length)

        return input_ids, attention_mask, token_type_ids, gender_target, img_id

