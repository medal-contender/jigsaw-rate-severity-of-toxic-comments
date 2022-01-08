import torch.nn as nn
from transformers import AutoModel
from torch.optim import lr_scheduler

class JigsawModel(nn.Module):

    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, num_classes)
        self.model_name = model_name
        
    def forward(self, ids, mask):        
        out = self.model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=False
        )
        # max_length 차원을 가짐
        if (
            "distilbert" in self.model_name 
            or "electra" in self.model_name
            or "deberta" in self.model_name
        ):
            # [CLS] 토큰만 사용 (16,768)
            out = self.drop(out[0][:,0,:])
        else:
            out = self.drop(out[1])
        outputs = self.fc(out)
        return outputs

def fetch_scheduler(optimizer, CONFIG):
    '''
        Config에 맞는 Solver Scheduler를 반환합니다.
    '''
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'None':
        return None
        
    return scheduler