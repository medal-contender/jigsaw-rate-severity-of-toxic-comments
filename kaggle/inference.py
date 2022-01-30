import os
import gc
import random

# For text manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
from torch.utils.data import Dataset, DataLoader

# For Transformer Models
from transformers import AutoTokenizer

# Utils
from tqdm import tqdm
from glob import glob
import sys
sys.path.insert(0, '../input/medalchallengerrepo/jigsaw-rate-severity-of-toxic-comments-develop/jigsaw_toxic_severity_rating/')

from tqdm import tqdm

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# pt파일 경로
MODEL_WEIGHTS = glob('../input/roberta/roberta/*.pt')
MODEL_DIR = '../input/models/roberta-base'

CONFIG = dict(
    seed = 42,
    test_batch_size = 128,
    max_length = 128,
    device = torch.device("cuda"),
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
)

ensemble_key = {
    'DEBERTA-BASE-DATA': 1.0,
    'roberta': 1.2,
    'muppet': 0.9,
}

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

    
@torch.no_grad()
def valid_fn(model, textloader, device):
    model.eval()
    
    PREDS = []
    
    bar = tqdm(enumerate(textloader), total=len(textloader))
    for _, text in bar:
        ids = text['ids'].to(device, dtype = torch.long)
        mask = text['mask'].to(device, dtype = torch.long)
        
        outputs = model(ids, mask)
        PREDS.append(outputs.view(-1).cpu().detach().numpy()) 
    
    PREDS = np.concatenate(PREDS)
    gc.collect()
    
    return PREDS


def inference(model_paths, textloader, device):
    final_preds = []
    for i, path in enumerate(model_paths):

        model_keyword = os.path.basename(path).split('_')[0][1:-1]
        assert model_keyword in ensemble_key, f"Model File Should Contain A Keyword Predifined In Ensemble Key"
        model_value = ensemble_key[model_keyword]

        model = torch.load(path)
        model.to(CONFIG['device'])
        
        print(f"Getting predictions for model {i+1}")
        preds = valid_fn(model, textloader, device)
        # Weight On Predictions
        preds *= model_value
        final_preds.append(preds)
    
        del model
        _ = gc.collect()
    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds

set_seed(CONFIG['seed'])

df = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")

test_dataset = JigsawDataset(
                    df, 
                    CONFIG['tokenizer'], 
                    max_length=CONFIG['max_length']
                )
test_loader = DataLoader(
                    test_dataset, 
                    batch_size=CONFIG['test_batch_size'],
                    num_workers=2, 
                    shuffle=False, 
                    pin_memory=True
                )

preds1 = inference(MODEL_WEIGHTS, test_loader, CONFIG['device'])

preds = (preds1-preds1.min())/(preds1.max()-preds1.min())

sub_df = pd.DataFrame()
sub_df['comment_id'] = df['comment_id']
sub_df['score'] = preds
sub_df['score'] = sub_df['score'].rank(method='first')
sub_df[['comment_id', 'score']].to_csv("/jigsaw/submission.csv", index=False)
