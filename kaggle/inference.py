# TODO: 경로 설정해야할 부분: 1. 모델경로, 2. bin파일 경로
import os
import gc
import copy
import time
import random

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For Transformer Models
from transformers import AutoTokenizer, AutoModel

# Utils
from tqdm import tqdm
import pathlib

import re
from bs4 import BeautifulSoup

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

CONFIG = dict(
    seed = 42,
    model_name = '../input/roberta-base',  # 모델 경로
    test_batch_size = 128,
    max_length = 128,
    num_classes = 1,
    dropout = 0.2,
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
)

CONFIG["tokenizer"] = AutoTokenizer.from_pretrained(CONFIG['model_name'])

# bin파일 경로
MODEL_PATHS = sorted(pathlib.Path('../input/jigsaw4-roberta/roberta-merge').glob('*.bin'))

def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.text = df['text'].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
                        text,
                        truncation=True,
                        add_special_tokens=True,
                        max_length=self.max_len,
                        padding='max_length'
                    )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']        
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long)
        }    

    
class JigsawModel(nn.Module):
    def __init__(self, model_name, num_classes, drop_p):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(drop_p)
        self.first_layer = nn.Linear(
            1024 
            if 'large' in model_name
            else 768,
            256)
        self.fc = nn.Sequential(
            self.first_layer,
            nn.LayerNorm(256),
            nn.Dropout(drop_p),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        self.model_name = model_name
        
    def forward(self, ids, mask):        
        out = self.model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=False
        )
        # max_length 차원을 가지는 경우
        if out[0].dim() == 3:
            # 시퀀스 토큰 전체의 평균을 사용
            out = torch.mean(out[0],axis=1)
            out = self.drop(out)
        else:
            out = self.drop(out[1])
        outputs = self.fc(out)
        return outputs
    
@torch.no_grad()
def valid_fn(model, dataloader, device):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    PREDS = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        
        outputs = model(ids, mask)
        PREDS.append(outputs.view(-1).cpu().detach().numpy()) 
    
    PREDS = np.concatenate(PREDS)
    gc.collect()
    
    return PREDS


def inference(model_paths, dataloader, device):
    final_preds = []
    for i, path in enumerate(model_paths):
        model = JigsawModel(CONFIG['model_name'], CONFIG['num_classes'], CONFIG['dropout'])
        model.to(CONFIG['device'])
        model.load_state_dict(torch.load(path))
        
        print(f"Getting predictions for model {i+1}")
        preds = valid_fn(model, dataloader, device)
        final_preds.append(preds)
    
    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds

def clean(data):
    
    data = data.replace(r"what's", "what is ")    
    data = data.replace(r"\'ve", " have ")
    data = data.replace(r"can't", "cannot ")
    data = data.replace(r"n't", " not ")
    data = data.replace(r"i'm", "i am ")
    data = data.replace(r"\'re", " are ")
    data = data.replace(r"\'d", " would ")
    data = data.replace(r"\'ll", " will ")
    data = data.replace(r"\'scuse", " excuse ")
    data = data.replace(r"\'s", " ")
    data = data.replace(r"@USER", "")
    
    # Clean some punctutations
    data = data.replace('\n', ' \n ')
    data = data.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3')
    # Replace repeating characters more than 3 times to length of 3
    data = data.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1')    
    # Add space around repeating characters
    data = data.replace(r'([*!?\']+)',r' \1 ')    
    # patterns with repeating characters 
    data = data.replace(r'([a-zA-Z])\1{2,}\b',r'\1\1')
    data = data.replace(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1')
    data = data.replace(r'[ ]{2,}',' ').strip()   
    data = data.replace(r'[ ]{2,}',' ').strip()   
    
    return data


def text_cleaning(text):
    template = re.compile(r'https?://\S+|www\.\S+')
    text = template.sub(r'', text)
    soup = BeautifulSoup(text, 'lxml')
    only_text = soup.get_text()
    text = only_text
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags = re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r"[^a-zA-Z\d]", " ", text)
    text = re.sub(' +', ' ', text)
    text = text.strip()
    return text

set_seed(CONFIG['seed'])
df = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")
df['text'] = df['text'].apply(lambda x: clean(x))
df['text'] = df['text'].apply(lambda x: text_cleaning(x))


test_dataset = JigsawDataset(df, CONFIG['tokenizer'], max_length=CONFIG['max_length'])
test_loader = DataLoader(test_dataset, batch_size=CONFIG['test_batch_size'],
                         num_workers=2, shuffle=False, pin_memory=True)

preds1 = inference(MODEL_PATHS, test_loader, CONFIG['device'])


preds = (preds1-preds1.min())/(preds1.max()-preds1.min())


sub_df = pd.DataFrame()
sub_df['comment_id'] = df['comment_id']
sub_df['score'] = preds
sub_df['score'] = sub_df['score'].rank(method='first')
sub_df[['comment_id', 'score']].to_csv("submission.csv", index=False)
