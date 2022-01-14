import random
import torch
import string
import munch
import yaml
import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import StratifiedKFold, KFold

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

def get_folded_dataframe(df,n_splits,random_state,shuffle=True):

    skf = StratifiedKFold(
        n_splits=n_splits, 
        shuffle=shuffle, 
        random_state=random_state
    )

    for fold, ( _, val_) in enumerate(skf.split(X=df, y=df.worker)):
        df.loc[val_ , "kfold"] = int(fold)
        
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
        self.cfg.training_keyword = args.training_keyword


    def load_yaml(self,file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.full_load(f)

        return data

