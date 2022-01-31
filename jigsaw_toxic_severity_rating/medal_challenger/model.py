import torch
import torch.nn as nn
from transformers import AutoModel
from torch.optim import lr_scheduler
from medal_challenger.configs import SCHEDULER_LIST
from transformers import AutoConfig

class AttentionBlock(nn.Module):

    def __init__(self, in_features, middle_features, out_features, drop_p):
        super().__init__()
        self.W = nn.Linear(in_features, middle_features)
        self.V = nn.Linear(middle_features, out_features)
        self.layer_norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(drop_p)

    def forward(self, features):
        # Normalization
        features = self.layer_norm(features)
        features = self.drop(features)
        # Attention Mechanism
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        # Context Vector As An Output
        return context_vector
    
class JigsawModel(nn.Module):
    
    def __init__(
        self, 
        model_name, 
        num_classes, 
        drop_p, 
        is_extra_attn=True,
        is_deeper_attn=True,
        device='cuda'
        ):
        
        super().__init__()

        self.input_hidden_dim = 1024 if 'large' in model_name else 768
        self.is_extra_attn = is_extra_attn
        self.is_deeper_attn = is_deeper_attn

        config = AutoConfig.from_pretrained(model_name)
        config.update({"output_hidden_states": True if self.is_deeper_attn else False,
                       "hidden_dropout_prob": drop_p,
                       "layer_norm_eps": 1e-7})

        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.drop = nn.Dropout(drop_p)
              
        self.dims_level_list = [
            [self.input_hidden_dim,256,num_classes,drop_p],
            [self.input_hidden_dim,256,num_classes,drop_p],
            [self.input_hidden_dim,256,num_classes,drop_p],
            [self.input_hidden_dim,256,num_classes,drop_p],
            [self.input_hidden_dim,256,num_classes,drop_p],
            [self.input_hidden_dim,256,num_classes,drop_p],
            [self.input_hidden_dim,6,num_classes,drop_p],
        ]

        # Selection Of Indices Of Hidden Layers
        self.level_list = [
            -1,-2,-4,-8,-11,-13
        ]
        
        # For The Top Hidden Layer
        self.simple_attention = AttentionBlock(
                                    self.input_hidden_dim, 
                                    self.input_hidden_dim, 
                                    num_classes,
                                    drop_p
                                ).to(device)
        # For Selected Hidden Layers
        self.deep_attention = [AttentionBlock(*level).to(device) for level in self.dims_level_list]
        # For Selected Hidden Layers
        self.simple_fc = nn.Linear(self.input_hidden_dim,num_classes)
        # For The Top Hidden Layer
        self.deep_fc = nn.Sequential(
            nn.Linear(self.input_hidden_dim,256),
            nn.LayerNorm(256),
            nn.Dropout(drop_p),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        
        
    def forward(self, ids, mask):        
        out = self.model(
            input_ids=ids,
            attention_mask=mask,
        )
        # Feature With max_length Dim
        if out[0].dim() == 3:
            # Additional Attention 
            if self.is_extra_attn:
                # Deeper Attention
                if self.is_deeper_attn:
                    out_list = [out.hidden_states[idx] for idx in self.level_list]
                    contexts = [attention(out_list[i]) for i, attention in enumerate(self.deep_attention[:-1])]
                    context = torch.stack(contexts,dim=1)
                    context = self.deep_attention[-1](context)
                    out = self.simple_fc(context)
                # Simple Attention
                else:
                    out = self.simple_attention(out[0])
                    out = self.deep_fc(out)
            # Squeeze Dim By Averaging Out max_length Dim
            else:
                out = torch.mean(out[0],axis=1)
                out = self.drop(out)
                out = self.deep_fc(out)
        # Simple Forward Pass
        else:
            out = self.drop(out[1])
            out = self.deep_fc(out)
        return out


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