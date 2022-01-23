import torch
import torch.nn as nn
from transformers import AutoModel
from torch.optim import lr_scheduler
from medal_challenger.configs import SCHEDULER_LIST
from transformers import RobertaModel,RobertaTokenizer
from transformers import MPNetModel,MPNetTokenizer
from transformers import FunnelBaseModel,FunnelTokenizer,FunnelModel
from transformers import ElectraModel, ElectraTokenizer, ElectraForSequenceClassification

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

def fetch_scheduler(optimizer, cfg):
    '''
        Config에 맞는 Solver Scheduler를 반환합니다.
    '''
    if SCHEDULER_LIST[cfg.model_param.scheduler] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train_param.T_max, 
            eta_min=float(cfg.train_param.min_lr)
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.train_param.T_0, 
            eta_min=float(cfg.train_param.min_lr)
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'None':
        return None
        
    return scheduler