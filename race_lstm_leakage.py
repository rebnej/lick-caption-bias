import torchtext
import torch
import csv
import spacy
import re
from torchtext.legacy import data
import pickle
import random
from nltk import word_tokenize
import nltk
nltk.download('punkt')
import time

import argparse
import numpy as np
import json
import os
import pprint
from nltk.tokenize import word_tokenize
from io import open
import sys
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='captioning', type=str)
    parser.add_argument("--cap_model", default='sat', type=str)
    parser.add_argument("--gender_or_race", default='race', type=str)
    parser.add_argument("--calc_ann_leak", default=False, type=bool)
    parser.add_argument("--calc_model_leak", default=False, type=bool)
    parser.add_argument("--calc_mw_acc", default=False, type=bool)
    parser.add_argument("--topk_grad_words", default=1, type=int)
    parser.add_argument("--test_ratio", default=0.1, type=float)
    parser.add_argument("--balanced_data", default=True, type=bool)
    parser.add_argument("--mask_race_words", default=False, type=bool)
    parser.add_argument("--use_glove", default=False, type=bool)
    parser.add_argument("--save_model_vocab", default=False, type=bool)
    parser.add_argument("--align_vocab", default=True, type=bool)
    parser.add_argument("--grad_cam", default=False, type=bool)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--workers", default=1, type=int)

    parser.add_argument("--embedding_dim", default=100, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--bidirectional", default=True, type=bool)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--pad_idx", default=0, type=int)
    parser.add_argument("--fix_length", default=False, type=bool)

    return parser


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
                
        #hidden = [batch size, hid dim * num directions]
            
        return self.fc(hidden), embedded



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

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
        print('#train : #test =', len(d_train), len(d_test))
    else:
        print('Balance data')

    return d_train, d_test


def train(model, iterator, optimizer, criterion, train_proc):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    for batch in iterator:
        
        optimizer.zero_grad()
        #print(batch)
        
        text, text_lengths = batch.prediction

        predictions, _ = model(text, text_lengths)
        predictions = predictions.squeeze(1)
        loss = criterion(predictions, batch.label.to(torch.float32))
        
        acc = binary_accuracy(predictions, batch.label.to(torch.float32))
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        train_proc.append(loss.item())
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), train_proc


def evaluate(model, iterator, criterion, batch_size):

    calc_score = True
    calc_race_acc = True
    m = nn.Sigmoid()
    total_score = 0

    epoch_loss = 0
    epoch_acc = 0

    light_preds_all, dark_preds_all = list(), list()
    light_truth_all, dark_truth_all = list(), list()

    model.eval()
    
    with torch.no_grad():

        cnt_data = 0
        for i, batch in enumerate(iterator):

            text, text_lengths = batch.prediction

            predictions, _ = model(text, text_lengths)
            predictions = predictions.squeeze(1)
            cnt_data += predictions.size(0)

            loss = criterion(predictions, batch.label.to(torch.float32))

            acc = binary_accuracy(predictions, batch.label.to(torch.float32))

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            if calc_score:
                probs = m(predictions).cpu() #[batch_size]
                pred_races = (probs >= 0.5000).int()
                #if i == 0:
                #    print('probs:', probs) #[batch_size]
                #    print('pred_genders:', pred_genders.shape, pred_genders) #[batch_size]
                #    print('batch.label.to(torch.int32):', batch.label.to(torch.int32).shape, batch.label.to(torch.int32)) #[batch_size]

                correct = torch.eq(pred_races, batch.label.to(torch.int32).cpu())
                #if i == 0:
                    #print(correct)

                pred_score_tensor = torch.zeros_like(correct, dtype=float)
                for i in range(pred_score_tensor.size(0)):
                    dark_score = probs[i]
                    light_score = 1 - dark_score
                    if light_score >= dark_score:
                        pred_score = light_score
                    else:
                        pred_score = dark_score

                    pred_score_tensor[i] = pred_score

                scores_tensor = correct.int() * pred_score_tensor
                correct_score_sum = torch.sum(scores_tensor)
                total_score += correct_score_sum.item()


            if calc_race_acc:
                probs = m(predictions).cpu() #[batch_size]
                pred_races = (probs >= 0.5000).int()
                light_target_ind = [i for i, x in enumerate(batch.label.to(torch.int32).cpu().numpy().tolist()) if x == 0]
                dark_target_ind = [i for i, x in enumerate(batch.label.to(torch.int32).cpu().numpy().tolist()) if x == 1]
                light_pred = [*itemgetter(*light_target_ind)(pred_races.tolist())]
                dark_pred = [*itemgetter(*dark_target_ind)(pred_races.tolist())]
                light_target = [*itemgetter(*light_target_ind)(batch.label.to(torch.int32).cpu().numpy().tolist())]
                dark_target = [*itemgetter(*dark_target_ind)(batch.label.to(torch.int32).cpu().numpy().tolist())]
                light_preds_all += light_pred
                light_truth_all += light_target
                dark_preds_all += dark_pred
                dark_truth_all += dark_target
            

    if calc_race_acc:
        light_acc = accuracy_score(light_truth_all, light_preds_all)
        dark_acc = accuracy_score(dark_truth_all, dark_preds_all)
    else:
        light_acc, dark_acc = None, None


    return epoch_loss / len(iterator), epoch_acc / len(iterator), total_score / cnt_data, light_acc, dark_acc



def main(args):
    if os.path.exists('bias_data/race_train.csv'):
        os.remove('bias_data/race_train.csv')
    if os.path.exists('bias_data/race_val.csv'):
        os.remove('bias_data/race_val.csv')
    if os.path.exists('bias_data/race_test.csv'):
        os.remove('bias_data/race_test.csv')

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device: {} n_gpu: {}".format(device, n_gpu))
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


    TEXT = data.Field(tokenize = 'spacy', tokenizer_language ='en_core_web_sm', include_lengths = True)

    LABEL = data.LabelField(dtype = torch.float)

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
                'prince','waiter','son','he','his','him','himself','brother','brothers']
    feminine = ['woman','women','female','lady','ladies','mother','girl', 'girls','aunt','wife','actress',
                'princess','waitress','daughter','she','her','hers','herself','sister','sisters']
    gender_words = masculine + feminine

    if args.mask_race_words:
        race_words = ['white', 'caucasian','black', 'african', 'asian', 'latino', 'latina', 'latinx','hispanic', 'native', 'indigenous']
    else:
        race_words = []


##################### ANN LIC score #######################
    if args.calc_ann_leak:
        print('--- calc ANN LIC score ---')
        ## Captioning ##
        if args.task == 'captioning':
            print('-- task is Captioning --')
            d_train, d_test = make_train_test_split(args, race_val_obj_cap_entries)
            
            val_acc_list = []
            light_acc_list, dark_acc_list = [], []
            score_list = []
            rand_score_list = []
            
            if args.align_vocab:
                model_vocab = pickle.load(open('./bias_data/model_vocab/%s_vocab.pkl' %args.cap_model, 'rb'))
                print('len(model_vocab):', len(model_vocab))

            for cap_ind in range(5):
                with open('bias_data/race_train.csv', 'w') as f:
                    writer = csv.writer(f)
                    for i, entry in enumerate(d_train):
                        if entry['bb_skin'] == 'Light':
                            race = 0
                        else:
                            race = 1
                        ctokens = word_tokenize(entry['caption_list'][cap_ind].lower())
                        new_list = []
                        for t in ctokens:
                            if t in race_words:
                                new_list.append('raceword')
                            elif args.align_vocab:
                                if t not in model_vocab:
                                    new_list.append('<unk>')
                                else:
                                    new_list.append(t)      
                            else:
                                new_list.append(t)

                        new_sent = ' '.join([c for c in new_list])
                        if i <= 5 and cap_ind == 0 and args.seed == 0:
                            print(new_sent)

                        writer.writerow([new_sent.strip(), race, entry['img_id']])

                with open('bias_data/race_test.csv', 'w') as f:
                    writer = csv.writer(f)
                    for i, entry in enumerate(d_test):
                        if entry['bb_skin'] == 'Light':
                            race = 0
                        else:
                            race = 1
                        ctokens = word_tokenize(entry['caption_list'][cap_ind].lower())
                        new_list = []
                        for t in ctokens:
                            if t in race_words:
                                new_list.append('raceword')
                            elif args.align_vocab:
                                if t not in model_vocab:
                                    new_list.append('<unk>')
                                else:
                                    new_list.append(t)
                            else:
                                new_list.append(t)

                        new_sent = ' '.join([c for c in new_list])

                        writer.writerow([new_sent.strip(), race, entry['img_id']])


                nlp = spacy.load("en_core_web_sm")

                TEXT = data.Field(sequential=True, tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True, use_vocab=True)
                LABEL = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)
                IMID = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

                train_val_fields = [
                    ('prediction', TEXT), # process it as text
                    ('label', LABEL), # process it as label
                    ('imid', IMID) 
                    ]

                train_data, test_data = torchtext.legacy.data.TabularDataset.splits(path='bias_data/',train='race_train.csv', test='race_test.csv',
                                                                            format='csv', fields=train_val_fields)

                MAX_VOCAB_SIZE = 25000

                if args.use_glove:
                    TEXT.build_vocab(train_data, vectors = "glove.6B.100d",  max_size = MAX_VOCAB_SIZE)
                else:
                    TEXT.build_vocab(train_data,  max_size = MAX_VOCAB_SIZE)
                LABEL.build_vocab(train_data)
                print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
                print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

                train_iterator, test_iterator = data.BucketIterator.splits(
                                                            (train_data, test_data),
                                                            batch_size = args.batch_size,
                                                            sort_key=lambda x: len(x.prediction), # on what attribute the text should be sorted
                                                            sort_within_batch = True,
                                                            device = device)
                INPUT_DIM = len(TEXT.vocab)
                EMBEDDING_DIM = 100
                HIDDEN_DIM = 256
                OUTPUT_DIM = 1
                N_LAYERS = 2
                BIDIRECTIONAL = True
                DROPOUT = 0.5
                PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
                #print(PAD_IDX)

                model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

                #print(f'The model has {count_parameters(model):,} trainable parameters')

                if args.use_glove:
                    pretrained_embeddings = TEXT.vocab.vectors
                    print(pretrained_embeddings.shape)
                    model.embedding.weight.data.copy_(pretrained_embeddings)

                UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
                model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
                model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

                # Training #
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
                criterion = nn.BCEWithLogitsLoss()

                model = model.to(device)
                criterion = criterion.to(device)

                N_EPOCHS = args.num_epochs

                best_valid_acc = float(0)

                train_proc = []
                valid_loss, valid_acc, avg_score, light_acc, dark_acc = evaluate(model, test_iterator, criterion, args.batch_size)
                rand_score_list.append(avg_score)

                for epoch in range(N_EPOCHS):

                    train_loss, train_acc, train_proc = train(model, train_iterator, optimizer, criterion, train_proc)

                valid_loss, valid_acc, avg_score, light_acc, dark_acc = evaluate(model, test_iterator, criterion, args.batch_size)
                val_acc_list.append(valid_acc)
                light_acc_list.append(light_acc)
                dark_acc_list.append(dark_acc)
                score_list.append(avg_score)
                #print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
                #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

            dark_avg_acc = sum(dark_acc_list) / len(dark_acc_list)
            light_avg_acc = sum(light_acc_list) / len(light_acc_list)
            avg_score = sum(score_list) / len(score_list)

            print('########## Results ##########')
            print(f"LIC score (LIC_D): {avg_score*100:.2f}%")
            #print(f"\t Dark Accuracy: {dark_avg_acc*100:.2f}%")
            #print(f"\t Light Accuracy: {light_avg_acc*100:.2f}%")
            print('#############################')



####################### MODEL LIC score ##########################
    if args.calc_model_leak:
        print('--- calc MODEL LIC score ---')
        ## Captioning ##
        if args.task == 'captioning':
            print('--- task is Captioning ---')
            d_train, d_test = make_train_test_split(args, selected_cap_race_entries)

            with open('bias_data/race_train.csv', 'w') as f:
                writer = csv.writer(f)
                for i, entry in enumerate(d_train):
                    if entry['bb_skin'] == 'Light':
                        race = 0
                    else:
                        race = 1
                    ctokens = word_tokenize(entry['pred'])
                    new_list = []
                    for t in ctokens:
                        if t in race_words:
                            new_list.append('raceword')
                        else:
                            new_list.append(t)      
                    new_sent = ' '.join([c for c in new_list])
                    if i <= 5 and args.seed == 0:
                        print(new_sent)

                    writer.writerow([new_sent.strip(), race, entry['img_id']])

            with open('bias_data/race_test.csv', 'w') as f:
                writer = csv.writer(f)
                for i, entry in enumerate(d_test):
                    if entry['bb_skin'] == 'Light':
                        race = 0
                    else:
                        race = 1

                    ctokens = word_tokenize(entry['pred'])
                    new_list = []
                    for t in ctokens:
                        if t in race_words:
                            new_list.append('raceword')
                        else:
                            new_list.append(t)      
                    new_sent = ' '.join([c for c in new_list])
      
                    writer.writerow([new_sent.strip(), race, entry['img_id']])

    

        nlp = spacy.load("en_core_web_sm")

        TEXT = data.Field(sequential=True, 
                       tokenize='spacy', 
                       tokenizer_language='en_core_web_sm',
                       include_lengths=True, 
                       use_vocab=True)
        LABEL = data.Field(sequential=False, 
                         use_vocab=False, 
                         pad_token=None, 
                         unk_token=None,
                         )
        IMID = data.Field(sequential=False,
                         use_vocab=False,
                         pad_token=None,
                         unk_token=None,
                         )



        train_val_fields = [
            ('prediction', TEXT), # process it as text
            ('label', LABEL), # process it as label
            ('imid', IMID)
        ]

        train_data, test_data = torchtext.legacy.data.TabularDataset.splits(path='bias_data/',train='race_train.csv', test='race_test.csv',
                                                                            format='csv', fields=train_val_fields)

        #ex = train_data[1]
        #print(ex.prediction, ex.label)

        MAX_VOCAB_SIZE = 25000

        if args.save_model_vocab:
            TEXT.build_vocab(train_data, test_data, max_size = MAX_VOCAB_SIZE)
            vocab_itos_list = TEXT.vocab.itos
            file_name = '/bias-vl/%s_vocab.pkl' %args.cap_model
            pickle.dump(vocab_itos_list, open(file_name, 'wb'))
            print('--- Saved vocab ---')

        if args.use_glove:
            print("-- Use GloVe")
            TEXT.build_vocab(train_data, vectors = "glove.6B.100d",  max_size = MAX_VOCAB_SIZE)
        else:
            TEXT.build_vocab(train_data,  max_size = MAX_VOCAB_SIZE)
        LABEL.build_vocab(train_data)
        print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
        print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
        #print(type(TEXT.vocab.itos))
        #print(LABEL.vocab.stoi)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_iterator, test_iterator = data.BucketIterator.splits(
                                                            (train_data, test_data), 
                                                            batch_size = args.batch_size,
                                                            sort_key=lambda x: len(x.prediction), # on what attribute the text should be sorted
                                                            sort_within_batch = True,
                                                            device = device)
        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256
        OUTPUT_DIM = 1
        N_LAYERS = 2
        BIDIRECTIONAL = True
        DROPOUT = 0.5
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        #print(PAD_IDX)

        model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

        #print(f'The model has {count_parameters(model):,} trainable parameters')

        if args.use_glove:
            pretrained_embeddings = TEXT.vocab.vectors
            print(pretrained_embeddings.shape)
            model.embedding.weight.data.copy_(pretrained_embeddings)

        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        # Training #
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        model = model.to(device)
        criterion = criterion.to(device)

        N_EPOCHS = args.num_epochs

        train_proc = []
        for epoch in range(N_EPOCHS):

            train_loss, train_acc, train_proc = train(model, train_iterator, optimizer, criterion, train_proc)

        valid_loss, valid_acc, avg_score, light_acc, dark_acc = evaluate(model, test_iterator, criterion, args.batch_size)
        print('########## Results ##########')
        print(f'LIC score (LIC_M): {avg_score*100:.2f}%')
        #print(f'\t Light. Acc: {light_acc*100:.2f}%')
        #print(f'\t Dark. Acc: {dark_acc*100:.2f}%')
        print('#############################')
        print()



if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    print("---Start---")
    print('Seed:', args.seed)
    print("Epoch:", args.num_epochs)
    print("Learning rate:", args.learning_rate)
    print("Use GLoVe:", args.use_glove)
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
