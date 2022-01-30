import torch
import torch.nn as nn
from transformers import AutoModel
from torch.optim import lr_scheduler
from medal_challenger.configs import SCHEDULER_LIST
from transformers import AutoConfig

class AttentionBlock(nn.Module):

    def __init__(self, in_features, middle_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, middle_features)
        self.V = nn.Linear(middle_features, out_features)

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector
    
class JigsawModel(nn.Module):
    
    def __init__(self, model_name, num_classes, drop_p, is_extra_attn=False):
        super().__init__()
        self.is_extra_attn = is_extra_attn
        self.model = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(drop_p)
        self.first_layer = nn.Linear(
            1024 
            if 'large' in model_name
            else 768,
            256)
              
        self.attention = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(drop_p),
            AttentionBlock(768, 768, 1)
        )
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
            # 추가 Attention 사용 
            if self.is_extra_attn:
                out = self.attention(out[0])
            # 시퀀스 토큰 전체의 평균을 사용
            else:
                out = torch.mean(out[0],axis=1)
                out = self.drop(out)
        else:
            out = self.drop(out[1])
        outputs = self.fc(out)
        return outputs
        
 
class DeeperJigsawModel(nn.Module):

    def __init__(self, model_name, num_classes, drop_p):
        super().__init__()

        self.num_classes = num_classes
        config = AutoConfig.from_pretrained(model_name)
        config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": drop_p,
                       "layer_norm_eps": 1e-7})

        self.model = AutoModel.from_pretrained(model_name, config=config)

        self.attention1 = self.attention_block(768,256)
        self.attention2 = self.attention_block(768,256)
        self.attention4 = self.attention_block(768,256)
        self.attention8 = self.attention_block(768,256)
        self.attention16 = self.attention_block(768,256)
        self.attention20 = self.attention_block(768,256)
        self.attention_sm = self.attention_block(768,6)

        self.fc = nn.Sequential(
            nn.Linear(768, num_classes)
        )

    def attention_block(self,in_features,out_features):

        return nn.Sequential(
            nn.Linear(in_features,out_features),
            nn.Tanh(),
            nn.Linear(out_features,self.num_classes),
            nn.Softmax(dim=1)
        )

    def get_context(self, attention_block, value):

        weights = attention_block(value)
        context = torch.sum(weights*value,dim=1)

        return context

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask)

        context1 = self.get_context(self.attention1,output.hidden_states[-1])
        context2 = self.get_context(self.attention2,output.hidden_states[-2])
        context4 = self.get_context(self.attention4,output.hidden_states[-4])
        context8 = self.get_context(self.attention8,output.hidden_states[-8])
        context16 = self.get_context(self.attention16,output.hidden_states[-11])
        context20 = self.get_context(self.attention20,output.hidden_states[-13])
        
        context_vectors = torch.stack(
            [
                context1, 
                context2, 
                context4, 
                context8, 
                context16, 
                context20
            ], 
            dim=1
        )

        layer_weights = self.attention_sm(context_vectors)

        context_vector = torch.sum(
            layer_weights * context_vectors, 
            dim=1
        )

        output = self.fc(context_vector)
        return output

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
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'LambdaLR':
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: cfg.train_param.reduce_ratio ** epoch
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'MultiplicativeLR':
        scheduler = lr_scheduler.MultiplicativeLR(
            optimizer,
            lr_lambda=lambda epoch: cfg.train_param.reduce_ratio ** epoch
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.train_param.step_size, gamma=cfg.train_param.gamma
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
             milestones=cfg.train_param.milestones, gamma=cfg.train_param.gamma
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.train_param.gamma
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min', min_lr=cfg.train_param.min_lr
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'CyclicLR':
        scheduler = lr_scheduler.CyclicLR(
            optimizer,
            base_lr=float(cfg.train_param.base_lr), 
            step_size_up=cfg.train_param.step_size_up, 
            max_lr=float(cfg.train_param.lr), 
            gamma=cfg.train_param.gamma, 
            mode='exp_range'
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.train_param.max_lr, 
            steps_per_epoch=cfg.train_param.steps_per_epoch, 
            epochs=cfg.train_param.epochs,
            anneal_strategy='linear'
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'None':
        return None
        
    return scheduler