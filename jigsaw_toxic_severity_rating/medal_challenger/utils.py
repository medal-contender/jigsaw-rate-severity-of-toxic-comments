import random
import torch
import string
import munch
import yaml
import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import StratifiedKFold, GroupKFold

# For Group K-Fold Strategy
class UnionFind():
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.parents[x] > self.parents[y]:
            x, y = y, x
        self.parents[x] += self.parents[y]
        self.parents[y] = x

def get_group_unionfind(df):

    unique_text = set(df['less_toxic']) | set(df['more_toxic'])
    text2num = {text: i for i, text in enumerate(unique_text)}
    num2text = {num: text for text, num in text2num.items()}
    df['num_less_toxic'] = df['less_toxic'].map(text2num)
    df['num_more_toxic'] = df['more_toxic'].map(text2num)

    uf = UnionFind(len(unique_text))
    for seq1, seq2 in df[['num_less_toxic', 'num_more_toxic']].to_numpy():
        uf.union(seq1, seq2)

    text2group = {num2text[i]: uf.find(i) for i in range(len(unique_text))}
    df['group'] = df['less_toxic'].map(text2group)

    return df

def set_seed(seed = 42):

    '''
        프로그램의 시드를 설정하여 매번 실행 결과가 동일하게 함
    '''

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def id_generator(size=12, chars=string.ascii_lowercase + string.digits):

    '''
        학습 버전을 구분짓기 위한 해시를 생성합니다. 
    '''

    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

def get_dataframe(csv_path):

    return pd.read_csv(csv_path)

def get_folded_dataframe(df,n_splits,random_state,shuffle=True,is_skf=True):

    if is_skf:
        skf = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=shuffle, 
            random_state=random_state
        )
        iterator = skf.split(X=df, y=df.worker)
    else:
        gkf = GroupKFold(n_splits=n_splits)
        df = get_group_unionfind(df)
        iterator = gkf.split(df, df, df['group'])

    for fold, ( _, val_) in enumerate(iterator):
        df.loc[val_ , "kfold"] = fold
        
    df["kfold"] = df["kfold"].astype(int)

    return df

def get_best_model(save_dir):

    model_list = glob(save_dir + '/*.bin')
    best_loss = float("inf")
    best_model = None

    for model in model_list:
        loss = float(model.split('_')[-1][:-4])
        if loss <= best_loss:
            best_loss = loss
            best_model = model
    
    return best_model

class ConfigManager(object):

    def __init__(self, args):

        self.config_file = args.config_file
        self.cfg = self.load_yaml(args.config_file)
        self.cfg = munch.munchify(self.cfg)
        self.cfg.config_file = args.config_file
        if args.train:
            self.cfg.training_keyword = args.training_keyword

    def load_yaml(self,file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.full_load(f)

        return data

