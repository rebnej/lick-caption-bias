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
from race_dataset import BERT_ANN_leak_data, BERT_MODEL_leak_data

from string import punctuation

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='captioning', type=str)
    parser.add_argument("--cap_model", default='sat', type=str)
    parser.add_argument("--gender_or_race", default='race', type=str)
    parser.add_argument("--calc_ann_leak", default=False, type=bool)
    parser.add_argument("--calc_model_leak", default=False, type=bool)
    parser.add_argument("--calc_race_acc", default=True, type=bool)
    parser.add_argument("--test_ratio", default=0.1, type=float)
    parser.add_argument("--balanced_data", default=True, type=bool)
    parser.add_argument("--mask_race_words", default=False, type=bool)
    parser.add_argument("--freeze_bert", default=False, type=bool)
    parser.add_argument("--store_topk_race_pred", default=False, type=bool)
    parser.add_argument("--topk_race_pred", default=50, type=int)
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
    parser.add_argument("--save_model", default=False, type=bool)
    parser.add_argument("--save_model_path", default='/bias-vl/bert.pt', type=str)
    parser.add_argument("--workers", default=1, type=int)

    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)

    return parser


def make_train_test_split(args, gender_task_race_entries):
    if args.balanced_data:
        light_entries, dark_entries = [], []
        for entry in gender_task_race_entries:
            if entry['bb_skin'] == 'Light':
                light_entries.append(entry)
            elif entry['bb_skin'] == 'Dark':
                dark_entries.append(entry)
        #print(len(female_entries))
        each_test_sample_num = round(len(dark_entries) * args.test_ratio)
        each_train_sample_num = len(dark_entries) - each_test_sample_num

        light_train_entries = [light_entries.pop(random.randrange(len(light_entries))) for _ in range(each_train_sample_num)]
        dark_train_entries = [dark_entries.pop(random.randrange(len(dark_entries))) for _ in range(each_train_sample_num)]
        light_test_entries = [light_entries.pop(random.randrange(len(light_entries))) for _ in range(each_test_sample_num)]
        dark_test_entries = [dark_entries.pop(random.randrange(len(dark_entries))) for _ in range(each_test_sample_num)]
        d_train = light_train_entries + dark_train_entries
        d_test = light_test_entries + dark_test_entries
        random.shuffle(d_train)
        random.shuffle(d_test)
        print(len(d_train), len(d_test))
    else:
        print('Balance data')

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
    val_loss, val_acc, val_light_acc, val_dark_acc, avg_score = calc_leak_epoch_pass(epoch, test_dataloader, model, optimizer, False, print_every=500)
    print('val, {0}, val loss: {1:.2f}, val acc: {2:.2f}'.format(epoch, val_loss*100, val_acc *100))
    if args.calc_race_acc:
        print('val, {0}, val loss: {1:.2f}, Light val acc: {2:.2f}'.format(epoch, val_loss*100, val_light_acc *100))
        print('val, {0}, val loss: {1:.2f}, Dark val acc: {2:.2f}'.format(epoch, val_loss*100, val_dark_acc *100))

    return val_acc, val_loss, val_light_acc, val_dark_acc, avg_score


def calc_leak_epoch_pass(epoch, data_loader, model, optimizer, training, print_every):
    t_loss = 0.0
    n_processed = 0
    preds = list()
    truth = list()
    light_preds_all, dark_preds_all = list(), list()
    light_truth_all, dark_truth_all = list(), list()

    if training:
        model.train()
    else:
        model.eval()

    if args.store_topk_race_pred:
        all_light_pred_values, all_dark_pred_values = [], []
        all_light_inputs, all_dark_inputs = [], []

    total_score = 0 # for calculate scores

    cnt_data = 0
    for ind, (input_ids, attention_mask, token_type_ids, race_target, img_id) in tqdm(enumerate(data_loader), leave=False): # images are not provided
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_type_ids = token_type_ids.cuda()

        race_target = torch.squeeze(race_target).cuda()
        #if ind == 0:
        #    for j in range(30):
        #        print(model.tokenizer.convert_ids_to_tokens(input_ids[j]))
        #    print(input_vec.shape) #[batch, num_obj]
        #    print(gender_target.shape) #[batch, 1] or [batch]
        predictions = model(input_ids, attention_mask, token_type_ids)
        #if ind == 0:
        #    print(predictions.shape) #[batch, 2]
        #    print(predictions)
        #    print(F.softmax(predictions, dim=1).cpu().detach().numpy())
        cnt_data += predictions.size(0)

        loss = F.cross_entropy(predictions, race_target, reduction='mean')

        if not training and args.store_topk_race_pred:
            pred_values = np.amax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1).tolist()
            pred_races = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)

            for pv, pr, imid, ids in zip(pred_values, pred_races, img_id, input_ids):
                tokens = model.tokenizer.convert_ids_to_tokens(ids)
                text = model.tokenizer.convert_tokens_to_string(tokens)
                text = text.replace('[PAD]', '')
                if pr == 0:
                    all_light_pred_values.append(pv)
                    all_light_inputs.append({'img_id': imid, 'text': text})
                else:
                    all_dark_pred_values.append(pv)
                    all_dark_inputs.append({'img_id': imid, 'text': text})

        if not training and args.calc_score:
            pred_races = np.argmax(F.softmax(predictions, dim=1).cpu().detach(), axis=1)
            race_target = race_target.cpu().detach()
            correct = torch.eq(pred_races, race_target)
            #if ind == 0:
            #    print('correct:', correct, correct.shape)

            pred_score_tensor = torch.zeros_like(correct, dtype=float)
            for i in range(pred_score_tensor.size(0)):
                #if ind == 0:
                #    print(F.softmax(predictions, dim=1).cpu().detach().shape)
                light_score = F.softmax(predictions, dim=1).cpu().detach()[i,0]
                dark_score = F.softmax(predictions, dim=1).cpu().detach()[i,1]
                if light_score >= dark_score:
                    pred_score = light_score
                else:
                    pred_score = dark_score
                #if ind == 0:
                #    print(male_score)
                #    print(female_score)
                pred_score_tensor[i] = pred_score
            #if ind == 0:
            #    print('pred_score_tensor:',pred_score_tensor, pred_score_tensor.shape)

            scores_tensor = correct.int() * pred_score_tensor
            correct_score_sum = torch.sum(scores_tensor)
            total_score += correct_score_sum.item()
             

        predictions = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)
        #if ind == 0 and epoch % 30 == 0:
        #    print(predictions)
        preds += predictions.tolist()
        truth += race_target.cpu().numpy().tolist()
        #if ind == 0 and epoch % 30 == 0:
        #    print('preds:', preds)
        #    print('truth:', truth)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t_loss += loss.item()
        n_processed += len(race_target)

        if (ind + 1) % print_every == 0 and training:
            print('{0}: task loss: {1:4f}'.format(ind + 1, t_loss / n_processed))

        if args.calc_race_acc and not training:
            light_target_ind = [i for i, x in enumerate(race_target.cpu().numpy().tolist()) if x == 0]
            dark_target_ind = [i for i, x in enumerate(race_target.cpu().numpy().tolist()) if x == 1]
            light_pred = [*itemgetter(*light_target_ind)(predictions.tolist())]
            dark_pred = [*itemgetter(*dark_target_ind)(predictions.tolist())]
            light_target = [*itemgetter(*light_target_ind)(race_target.cpu().numpy().tolist())]
            dark_target = [*itemgetter(*dark_target_ind)(race_target.cpu().numpy().tolist())]
            light_preds_all += light_pred
            light_truth_all += light_target
            dark_preds_all += dark_pred
            dark_truth_all += dark_target

    acc = accuracy_score(truth, preds)

    if args.calc_race_acc and not training:
        light_acc = accuracy_score(light_truth_all, light_preds_all)
        dark_acc = accuracy_score(dark_truth_all, dark_preds_all)
    else:
        light_acc, dark_acc = None, None

    if args.store_topk_race_pred and not training:
        all_light_pred_values = np.array(all_light_pred_values)
        all_dark_pred_values = np.array(all_dark_pred_values)
        #Light
        light_ind = all_light_pred_values.argsort()[-args.topk_race_pred:][::-1]
        light_topk_inputs = np.array(all_light_inputs)[light_ind]
        light_topk_scores = all_light_pred_values[light_ind]
        print("topk inputs (Light)")
        print(light_topk_inputs)
        print(light_topk_scores)
        print(light_ind)
        print()
        #Dark
        dark_ind = all_dark_pred_values.argsort()[-args.topk_race_pred:][::-1]
        dark_topk_inputs = np.array(all_dark_inputs)[dark_ind]
        dark_topk_scores = all_dark_pred_values[dark_ind]
        print("topk inputs (Dark)")
        print(dark_topk_inputs)
        print(dark_topk_scores)
        print(dark_ind)

    #if args.calc_score and not training:
    #    print("### AVG SCORE ###")
    #    print(total_score / (args.batch_size * len(data_loader)))
    #    print("#################")

    ###return t_loss / n_processed, acc, light_acc, dark_acc, total_score / (args.batch_size * len(data_loader))
    return t_loss / n_processed, acc, light_acc, dark_acc, total_score / cnt_data



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

    race_val_obj_cap_entries = pickle.load(open('bias_data/Human_Ann/race_val_obj_cap_entries.pkl', 'rb')) # Human captions

    #Select captioning model
    if args.cap_model == 'nic':
        selected_cap_race_entries = pickle.load(open('bias_data/Show-Tell/race_val_st10_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'sat':
        selected_cap_race_entries = pickle.load(open('bias_data/Show-Attend-Tell/race_val_sat_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'fc':
        selected_cap_race_entries = pickle.load(open('bias_data/Att2in_FC/race_val_fc_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'att2in':
        selected_cap_race_entries = pickle.load(open('bias_data/Att2in_FC/race_val_att2in_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'updn':
        selected_cap_race_entries = pickle.load(open('bias_data/UpDn/race_val_updn_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'transformer':
        selected_cap_race_entries = pickle.load(open('bias_data/Transformer/race_val_transformer_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'oscar':
        selected_cap_race_entries = pickle.load(open('bias_data/Oscar/race_val_cider_oscar_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'nic_equalizer':
        selected_cap_race_entries = pickle.load(open('bias_data/Woman-Snowboard/race_val_snowboard_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'nic_plus':
        selected_cap_race_entries = pickle.load(open('bias_data/Woman-Snowboard/race_val_baselineft_cap_entries.pkl', 'rb'))


    masculine = ['man','men','male','father','gentleman','gentlemen','boy','boys','uncle','husband','actor',
                'prince','waiter','son','he','his','him','himself','brother','brothers', 'guy', 'guys', 'emperor','emperors','dude','dedes','cowboy']
    feminine = ['woman','women','female','lady','ladies','mother','girl', 'girls','aunt','wife','actress',
                'princess','waitress','daughter','she','her','hers','herself','sister','sisters', 'queen','queens','pregnant']
    gender_words = masculine + feminine

    if args.mask_race_words:
        race_words = ['white', 'caucasian','black', 'african', 'asian', 'latino', 'latina', 'latinx','hispanic', 'native', 'indigenous']
    else:
        race_words = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ##################### ANN LIC score #######################
    if args.calc_ann_leak:
        print('--- calc ANN LIC score ---')
        ## Captioning ##
        if args.task == 'captioning':
            print('-- Task is Captioning --')
            d_train, d_test = make_train_test_split(args, race_val_obj_cap_entries)
            acc_list = []
            score_list = []
            light_acc_list, dark_acc_list = [], []
            rand_acc_list = []
            rand_score_list = []
            for caption_ind in range(5):
                trainANNCAPobject = BERT_ANN_leak_data(d_train, d_test, args, race_val_obj_cap_entries, race_words, tokenizer,
                                                args.max_seq_length, split='train', caption_ind=caption_ind)
                testANNCAPobject = BERT_ANN_leak_data(d_train, d_test, args, race_val_obj_cap_entries, race_words, tokenizer,
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
                val_acc, val_loss, val_light_acc, val_dark_acc, avg_score = calc_leak(args, model, train_dataloader, test_dataloader)
                acc_list.append(val_acc)
                light_acc_list.append(val_light_acc)
                dark_acc_list.append(val_dark_acc)
                score_list.append(avg_score)

            dark_avg_acc = sum(dark_acc_list) / len(dark_acc_list)
            light_avg_acc = sum(light_acc_list) / len(light_acc_list)
            avg_score = sum(score_list) / len(score_list)
            print('########### Reluts ##########')
            print(f"LIC score (LIC_D): {avg_score*100:.2f}%")
            #print(f'\t Light. Acc: {light_avg_acc*100:.2f}%')
            #print(f'\t Dark. Acc: {dark_avg_acc*100:.2f}%')
            print('#############################')


    ##################### MODEL LIC score #######################
    if args.calc_model_leak:
        print('--- calc MODEL LIC score ---')
        ## Captioning ##
        if args.task == 'captioning':
            print('-- Task is Captioning --')
            d_train, d_test = make_train_test_split(args, selected_cap_race_entries)
            trainMODELCAPobject = BERT_MODEL_leak_data(d_train, d_test, args, selected_cap_race_entries, race_words, tokenizer,
                                                args.max_seq_length, split='train')
            testMODELCAPobject = BERT_MODEL_leak_data(d_train, d_test, args, selected_cap_race_entries, race_words, tokenizer,
                                                args.max_seq_length, split='test')
            train_dataloader = torch.utils.data.DataLoader(trainMODELCAPobject, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
            test_dataloader = torch.utils.data.DataLoader(testMODELCAPobject, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
            # initialize gender classifier
            model = BERT_GenderClassifier(args, tokenizer)
            # calculate random predictions
            rand_val_acc, rand_val_loss, rand_val_light_acc, rand_val_dark_acc, rand_avg_score = calc_random_acc_score(args, model, test_dataloader)
            # train and test
            val_acc, val_loss, val_light_acc, val_dark_acc, avg_score = calc_leak(args, model, train_dataloader, test_dataloader)

            print('########### Reluts ##########')
            print(f'LIC score (LIC_M): {avg_score*100:.2f}%')
            #print(f'\t Light. Acc: {val_light_acc*100:.2f}%')
            #print(f'\t Dark. Acc: {val_dark_acc*100:.2f}%')
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
    print("Mask race words:", args.mask_race_words)

    if args.calc_ann_leak:
        print('Align vocab:', args.align_vocab)
        if args.align_vocab:
            print('Vocab of ', args.cap_model)

    print()

    main(args)

