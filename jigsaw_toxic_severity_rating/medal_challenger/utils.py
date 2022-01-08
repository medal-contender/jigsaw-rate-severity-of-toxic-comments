import numpy as np
import random
import torch
import string
import os
import pandas as pd
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