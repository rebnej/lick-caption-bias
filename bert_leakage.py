import torch
import csv
import spacy
import re
import pickle
import random
import csv
from nltk import word_tokenize
import nltk
nltk.download('punkt')
import time

import argparse
import os
import pprint
import numpy as np
from nltk.tokenize import word_tokenize
from io import open
import sys
import json
import pickle
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import transformers as tf
from transformers import BertTokenizer
from transformers import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from transformers import AdamW, get_linear_schedule_with_warmup

import torch.utils.data as data
from transformers import BertModel
from transformers import BertPreTrainedModel

from model import BERT_GenderClassifier
from bias_dataset import BERT_ANN_leak_data, BERT_MODEL_leak_data

from string import punctuation

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='captioning', type=str)
    parser.add_argument("--cap_model", default='sat', type=str)
    parser.add_argument("--gender_or_race", default='gender', type=str)
    parser.add_argument("--calc_ann_leak", default=False, type=bool)
    parser.add_argument("--calc_model_leak", default=False, type=bool)
    parser.add_argument("--calc_mw_acc", default=True, type=bool)
    parser.add_argument("--test_ratio", default=0.1, type=float)
    parser.add_argument("--balanced_data", default=True, type=bool)
    parser.add_argument("--mask_gender_words", default=True, type=bool)
    parser.add_argument("--freeze_bert", default=False, type=bool)
    parser.add_argument("--store_topk_gender_pred", default=False, type=bool)
    parser.add_argument("--topk_gender_pred", default=50, type=int)
    parser.add_argument("--calc_score", default=True, type=bool)
    parser.add_argument("--align_vocab", default=True, type=bool)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--optimizer", default='adamw', type=str, help="adamw or adam")
    parser.add_argument("--adam_correct_bias", default=True, type=bool)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.98, type=float, help="0.999:huggingface, 0.98:RoBERTa paper")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer. 1e-8:first, 1e-6:RoBERTa paper")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight deay if we apply some. 0.001:first, 0.01:RoBERTa")
    parser.add_argument("--coco_lk_model_dir", default='/Bias/leakage/', type=str)
    parser.add_argument("--workers", default=1, type=int)

    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)

    return parser


def make_train_test_split(args, gender_task_mw_entries):
    if args.balanced_data:
        male_entries, female_entries = [], []
        for entry in gender_task_mw_entries:
            if entry['bb_gender'] == 'Female':
                female_entries.append(entry)
            else:
                male_entries.append(entry)
        #print(len(female_entries))
        each_test_sample_num = round(len(female_entries) * args.test_ratio)
        each_train_sample_num = len(female_entries) - each_test_sample_num

        male_train_entries = [male_entries.pop(random.randrange(len(male_entries))) for _ in range(each_train_sample_num)]
        female_train_entries = [female_entries.pop(random.randrange(len(female_entries))) for _ in range(each_train_sample_num)]
        male_test_entries = [male_entries.pop(random.randrange(len(male_entries))) for _ in range(each_test_sample_num)]
        female_test_entries = [female_entries.pop(random.randrange(len(female_entries))) for _ in range(each_test_sample_num)]
        d_train = male_train_entries + female_train_entries
        d_test = male_test_entries + female_test_entries
        random.shuffle(d_train)
        random.shuffle(d_test)
        print('#train : #test = ', len(d_train), len(d_test))
    else:
        d_train, d_test = train_test_split(gender_task_mw_entries, test_size=args.test_ratio, random_state=args.seed,
                                   stratify=[entry['bb_gender'] for entry in gender_obj_cap_mw_entries])

    return d_train, d_test



def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def calc_random_acc_score(args, model, test_dataloader):
    print("--- Random guess --")
    model = model.cuda()
    optimizer = None
    epoch = None
    val_loss, val_acc, val_male_acc, val_female_acc, avg_score = calc_leak_epoch_pass(epoch, test_dataloader, model, optimizer, False, print_every=500)

    return val_acc, val_loss, val_male_acc, val_female_acc, avg_score



def calc_leak(args, model, train_dataloader, test_dataloader):
    model = model.cuda()
    print("Num of Trainable Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = 1e-5)
    elif args.optimizer == 'adamw':
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        #no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            #{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.beta1, args.beta2), correct_bias=args.adam_correct_bias, eps=args.adam_epsilon)

    train_loss_arr = list()
    train_acc_arr = list()

    # training
    for epoch in range(args.num_epochs):
        # train
        train_loss, train_acc, _, _, _ = calc_leak_epoch_pass(epoch, train_dataloader, model, optimizer, True, print_every=500)
        train_loss_arr.append(train_loss)
        train_acc_arr.append(train_acc)
        if epoch % 5 == 0:
            print('train, {0}, train loss: {1:.2f}, train acc: {2:.2f}'.format(epoch, \
                train_loss*100, train_acc*100))

    print("Finish training")
    print('{0}: train acc: {1:2f}'.format(epoch, train_acc))

    # validation
    val_loss, val_acc, val_male_acc, val_female_acc, avg_score = calc_leak_epoch_pass(epoch, test_dataloader, model, optimizer, False, print_every=500)
    print('val, {0}, val loss: {1:.2f}, val acc: {2:.2f}'.format(epoch, val_loss*100, val_acc *100))
    if args.calc_mw_acc:
        print('val, {0}, val loss: {1:.2f}, Male val acc: {2:.2f}'.format(epoch, val_loss*100, val_male_acc *100))
        print('val, {0}, val loss: {1:.2f}, Feale val acc: {2:.2f}'.format(epoch, val_loss*100, val_female_acc *100))

    return val_acc, val_loss, val_male_acc, val_female_acc, avg_score


def calc_leak_epoch_pass(epoch, data_loader, model, optimizer, training, print_every):
    t_loss = 0.0
    n_processed = 0
    preds = list()
    truth = list()
    male_preds_all, female_preds_all = list(), list()
    male_truth_all, female_truth_all = list(), list()

    if training:
        model.train()
    else:
        model.eval()

    if args.store_topk_gender_pred:
        all_male_pred_values, all_female_pred_values = [], []
        all_male_inputs, all_female_inputs = [], []

    total_score = 0 # for calculate scores

    cnt_data = 0
    for ind, (input_ids, attention_mask, token_type_ids, gender_target, img_id) in tqdm(enumerate(data_loader), leave=False): # images are not provided
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_type_ids = token_type_ids.cuda()

        gender_target = torch.squeeze(gender_target).cuda()
        predictions = model(input_ids, attention_mask, token_type_ids)
        cnt_data += predictions.size(0)

        loss = F.cross_entropy(predictions, gender_target, reduction='mean')

        if not training and args.store_topk_gender_pred:
            pred_values = np.amax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1).tolist()
            pred_genders = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)

            for pv, pg, imid, ids in zip(pred_values, pred_genders, img_id, input_ids):
                tokens = model.tokenizer.convert_ids_to_tokens(ids)
                text = model.tokenizer.convert_tokens_to_string(tokens)
                text = text.replace('[PAD]', '')
                if pg == 0:
                    all_male_pred_values.append(pv)
                    all_male_inputs.append({'img_id': imid, 'text': text})
                else:
                    all_female_pred_values.append(pv)
                    all_female_inputs.append({'img_id': imid, 'text': text})

        if not training and args.calc_score:
            pred_genders = np.argmax(F.softmax(predictions, dim=1).cpu().detach(), axis=1)
            gender_target = gender_target.cpu().detach()
            correct = torch.eq(pred_genders, gender_target)
            #if ind == 0:
            #    print('correct:', correct, correct.shape)

            pred_score_tensor = torch.zeros_like(correct, dtype=float)
            for i in range(pred_score_tensor.size(0)):
                male_score = F.softmax(predictions, dim=1).cpu().detach()[i,0]
                female_score = F.softmax(predictions, dim=1).cpu().detach()[i,1]
                if male_score >= female_score:
                    pred_score = male_score
                else:
                    pred_score = female_score

                pred_score_tensor[i] = pred_score

            scores_tensor = correct.int() * pred_score_tensor
            correct_score_sum = torch.sum(scores_tensor)
            total_score += correct_score_sum.item()

        predictions = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)
        preds += predictions.tolist()
        truth += gender_target.cpu().numpy().tolist()

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t_loss += loss.item()
        n_processed += len(gender_target)

        if (ind + 1) % print_every == 0 and training:
            print('{0}: task loss: {1:4f}'.format(ind + 1, t_loss / n_processed))

        if args.calc_mw_acc and not training:
            male_target_ind = [i for i, x in enumerate(gender_target.cpu().numpy().tolist()) if x == 0]
            female_target_ind = [i for i, x in enumerate(gender_target.cpu().numpy().tolist()) if x == 1]
            male_pred = [*itemgetter(*male_target_ind)(predictions.tolist())]
            female_pred = [*itemgetter(*female_target_ind)(predictions.tolist())]
            male_target = [*itemgetter(*male_target_ind)(gender_target.cpu().numpy().tolist())]
            female_target = [*itemgetter(*female_target_ind)(gender_target.cpu().numpy().tolist())]
            male_preds_all += male_pred
            male_truth_all += male_target
            female_preds_all += female_pred
            female_truth_all += female_target

    acc = accuracy_score(truth, preds)

    if args.calc_mw_acc and not training:
        male_acc = accuracy_score(male_truth_all, male_preds_all)
        female_acc = accuracy_score(female_truth_all, female_preds_all)
    else:
        male_acc, female_acc = None, None

    return t_loss / n_processed, acc, male_acc, female_acc, total_score / cnt_data



def main(args):
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device: {} n_gpu: {}".format(device, n_gpu))
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    gender_obj_cap_mw_entries = pickle.load(open('bias_data/Human_Ann/gender_obj_cap_mw_entries.pkl', 'rb')) # Human captions

    #Select captioning model
    if args.cap_model == 'nic':
        selected_cap_gender_entries = pickle.load(open('bias_data/Show-Tell/gender_val_st10_cap_mw_entries.pkl', 'rb'))
    elif args.cap_model == 'sat':
        selected_cap_gender_entries = pickle.load(open('bias_data/Show-Attend-Tell/gender_val_sat_cap_mw_entries.pkl', 'rb'))
    elif args.cap_model == 'fc':
        selected_cap_gender_entries = pickle.load(open('bias_data/Att2in_FC/gender_val_fc_cap_mw_entries.pkl', 'rb'))
    elif args.cap_model == 'att2in':
        selected_cap_gender_entries = pickle.load(open('bias_data/Att2in_FC/gender_val_att2in_cap_mw_entries.pkl', 'rb'))
    elif args.cap_model == 'updn':
        selected_cap_gender_entries = pickle.load(open('bias_data/UpDn/gender_val_updn_cap_mw_entries.pkl', 'rb'))
    elif args.cap_model == 'transformer':
        selected_cap_gender_entries = pickle.load(open('bias_data/Transformer/gender_val_transformer_cap_mw_entries.pkl', 'rb'))
    elif args.cap_model == 'oscar':
        selected_cap_gender_entries = pickle.load(open('bias_data/Oscar/gender_val_cider_oscar_cap_mw_entries.pkl', 'rb'))
    elif args.cap_model == 'nic_equalizer':
        selected_cap_gender_entries = pickle.load(open('bias_data/Woman-Snowboard/gender_val_snowboard_cap_mw_entries.pkl', 'rb'))
    elif args.cap_model == 'nic_plus':
        selected_cap_gender_entries = pickle.load(open('bias_data/Woman-Snowboard/gender_val_baselineft_cap_mw_entries.pkl', 'rb'))


    masculine = ['man','men','male','father','gentleman','gentlemen','boy','boys','uncle','husband','actor',
                'prince','waiter','son','he','his','him','himself','brother','brothers', 'guy', 'guys', 'emperor','emperors','dude','dudes','cowboy']
    feminine = ['woman','women','female','lady','ladies','mother','girl', 'girls','aunt','wife','actress',
                'princess','waitress','daughter','she','her','hers','herself','sister','sisters', 'queen','queens','pregnant']
    gender_words = masculine + feminine

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ##################### ANN LIC score #######################
    if args.calc_ann_leak:
        print('--- calc ANN LIC score ---')
        ## Captioning ##
        if args.task == 'captioning':
            print('-- Task is Captioning --')
            d_train, d_test = make_train_test_split(args, gender_obj_cap_mw_entries)
            val_acc_list = []
            score_list = []
            male_acc_list, female_acc_list = [], []
            rand_acc_list = []
            rand_score_list = []
            for caption_ind in range(5):
                trainANNCAPobject = BERT_ANN_leak_data(d_train, d_test, args, gender_obj_cap_mw_entries, gender_words, tokenizer,
                                                args.max_seq_length, split='train', caption_ind=caption_ind)
                testANNCAPobject = BERT_ANN_leak_data(d_train, d_test, args, gender_obj_cap_mw_entries, gender_words, tokenizer,
                                                args.max_seq_length, split='test', caption_ind=caption_ind)
                train_dataloader = torch.utils.data.DataLoader(trainANNCAPobject, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
                test_dataloader = torch.utils.data.DataLoader(testANNCAPobject, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
                # initialize gender classifier
                model = BERT_GenderClassifier(args, tokenizer)
                # calculate random predictions
                val_acc, val_loss, val_male_acc, val_female_acc, avg_score = calc_random_acc_score(args, model, test_dataloader)
                rand_acc_list.append(val_acc)
                rand_score_list.append(avg_score)
                # train and test
                val_acc, val_loss, val_male_acc, val_female_acc, avg_score = calc_leak(args, model, train_dataloader, test_dataloader)
                val_acc_list.append(val_acc)
                male_acc_list.append(val_male_acc)
                female_acc_list.append(val_female_acc)
                score_list.append(avg_score)

            female_avg_acc = sum(female_acc_list) / len(female_acc_list)
            male_avg_acc = sum(male_acc_list) / len(male_acc_list)
            avg_score = sum(score_list) / len(score_list)
            print('########### Reluts ##########')
            print(f"LIC score (LIC_D): {avg_score*100:.2f}%")
            #print(f"\t Female Accuracy: {female_avg_acc*100:.2f}%")
            #print(f"\t Male Accuracy: {male_avg_acc*100:.2f}%")
            print('#############################')



    ##################### MODEL LIC score #######################
    if args.calc_model_leak:
        print('--- calc MODEL LIC score---')
        ## Captioning ##
        if args.task == 'captioning':
            print('-- Task is Captioning --')
            d_train, d_test = make_train_test_split(args, selected_cap_gender_entries)
            trainMODELCAPobject = BERT_MODEL_leak_data(d_train, d_test, args, selected_cap_gender_entries, gender_words, tokenizer,
                                                args.max_seq_length, split='train')
            testMODELCAPobject = BERT_MODEL_leak_data(d_train, d_test, args, selected_cap_gender_entries, gender_words, tokenizer,
                                                args.max_seq_length, split='test')
            train_dataloader = torch.utils.data.DataLoader(trainMODELCAPobject, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
            test_dataloader = torch.utils.data.DataLoader(testMODELCAPobject, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
            # initialize gender classifier
            model = BERT_GenderClassifier(args, tokenizer)
            # calculate random predictions
            rand_val_acc, rand_val_loss, rand_val_male_acc, rand_val_female_acc, rand_avg_score = calc_random_acc_score(args, model, test_dataloader)
            # train and test
            val_acc, val_loss, val_male_acc, val_female_acc, avg_score = calc_leak(args, model, train_dataloader, test_dataloader)

            print('########### Reluts ##########')
            print(f'LIC score (LIC_M): {avg_score*100:.2f}%')
            #print(f'\t Male. Acc: {val_male_acc*100:.2f}%')
            #print(f'\t Female. Acc: {val_female_acc*100:.2f}%')
            print('#############################')




if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    print()
    print("---Start---")
    print('Seed:', args.seed)
    print("Epoch:", args.num_epochs)
    print("Freeze BERT:", args.freeze_bert)
    print("Learning rate:", args.learning_rate)
    print("Batch size:", args.batch_size)
    print("Calculate score:", args.calc_score)
    print("Task:", args.task)
    if args.task == 'captioning' and args.calc_model_leak:
        print("Captioning model:", args.cap_model)
    print("Gender or Race:", args.gender_or_race)

    if args.calc_ann_leak:
        print('Align vocab:', args.align_vocab)
        if args.align_vocab:
            print('Vocab of ', args.cap_model)

    print()

    main(args)

