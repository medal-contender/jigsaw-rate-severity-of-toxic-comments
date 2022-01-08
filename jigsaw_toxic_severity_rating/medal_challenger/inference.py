import gc
import torch
import numpy as np
from tqdm import tqdm
from medal_challenger.model import JigsawModel

@torch.no_grad()
def infer_with_one_model(model, dataloader, device):
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

def bert_ensemble(model_paths, dataloader, CONFIG):
    final_preds = []
    for i, path in enumerate(model_paths):
        model = JigsawModel(f"../models/{CONFIG['model_name']}")
        model.to(CONFIG['device'])
        model.load_state_dict(torch.load(path))
        
        print(f"Getting predictions for model {i+1}")
        preds = infer_with_one_model(model, dataloader, CONFIG['device'])
        final_preds.append(preds)
    
    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds