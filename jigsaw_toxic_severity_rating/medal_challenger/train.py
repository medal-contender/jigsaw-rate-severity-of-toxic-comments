import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def criterion(outputs1, outputs2, targets, margin):
    '''
        Pairwise Ranking Loss

        targets는 1로, outputs1이 outputs2보다 더 커야한다는 가정을 둡니다.
        loss(outputs1,outputs2,targets) = max(0,-targets*(outputs1-outputs2)+margin)으로 계산됩니다.
        outputs1과 outputs2의 차이가 크면 loss는 0에 가까워지고
        outputs1과 outputs2의 차이가 적어지면 margin에 가까워집니다.

        loss가 0에 가깝다는 뜻은 두 샘플이 충분히 멀리 있어 Parameters가 업데이트 되지 않는다는 것입니다.
    '''
    return nn.MarginRankingLoss(
        margin=margin # 0.5
    )(outputs1, outputs2, targets)

def train_one_epoch(
        model, 
        optimizer, 
        scheduler, 
        dataloader, 
        epoch,
        cfg,
    ):
    '''
        1 에폭 학습을 위한 함수
    '''
    model.train()

    '''
        Gradient Scaling:
            만약 어떤 특정한 연산에 대한 Forward Pass가 float16 입력을 받으면, 
            해당 연산에 대한 Backward Pass는 float16 Gradient를 생성합니다.
            float16으로 표현이 안되는 매우 작은 Gradient는 0으로 Flush 됩니다.
            따라서 Forward Pass는 float16으로 표현하고,
            Backward Pass에서는 다시 float32로 표현해야합니다(Scaling).
    '''
    scaler = GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        # 기기 변경
        more_toxic_ids = data['more_toxic_ids'].to(cfg.model_param.device, dtype = torch.long)
        more_toxic_mask = data['more_toxic_mask'].to(cfg.model_param.device, dtype = torch.long)
        less_toxic_ids = data['less_toxic_ids'].to(cfg.model_param.device, dtype = torch.long)
        less_toxic_mask = data['less_toxic_mask'].to(cfg.model_param.device, dtype = torch.long)
        targets = data['target'].to(cfg.model_param.device, dtype=torch.long)
        
        batch_size = more_toxic_ids.size(0)

        # Automatic Mixed Precision Within The Region
        with autocast():
            more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
            less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
            
            loss = criterion(
                more_toxic_outputs, 
                less_toxic_outputs, 
                targets, 
                cfg.train_param.ranking_margin
            )
            loss = loss / cfg.train_param.accumulate_grad_batches
        
        '''
            Gradients 저장 및 Upscaling:
                scaler를 사용해 float16의 loss를 float32로 정밀도를 올려 Gradient를 저장합니다.
                이 과정(with autocast() & scale())을 통해
                outputs의 용량은 절반이 되고, 연산되는 loss 크기도 절반이 되지만 Data Accuracy는 유지되어
                연산 시간과 Memory Footprint(사용하는 메모리 양)에서 이득이 있습니다.
                (특히 Linear Layer와 Conv. Layer에서 큰 효과가 있습니다.)
        '''
        scaler.scale(loss).backward()

        # Gradient Accumulation
        if (step + 1) % cfg.train_param.accumulate_grad_batches == 0:
            # Optimizer가 모든 파라미터를 Iterate하면서 모든 파라미터의 Gradients로 파라미터를 업데이트 합니다.
            # Makes The Optimizer Iterate Over All Parameters (Tensors) 
            # It Is Supposed To Update And Use Their Internally Stored Grad To Update Their Values.
            scaler.step(optimizer)
            scaler.update()
            # Parameter Gradients를 0으로 초기화
            optimizer.zero_grad()

            # Scheduler에 따라 Learning Rate 조정
            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(
            Epoch=epoch, 
            Train_Loss=epoch_loss,
            LR=optimizer.param_groups[0]['lr']
        )
    gc.collect()
    
    return epoch_loss


@torch.no_grad()
def valid_one_epoch(
        model, 
        dataloader, 
        device, 
        epoch,
        margin,
    ):
    '''
        1 에폭 평가를 위한 함수
    '''
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        more_toxic_ids = data['more_toxic_ids'].to(device, dtype = torch.long)
        more_toxic_mask = data['more_toxic_mask'].to(device, dtype = torch.long)
        less_toxic_ids = data['less_toxic_ids'].to(device, dtype = torch.long)
        less_toxic_mask = data['less_toxic_mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype=torch.long)
        
        batch_size = more_toxic_ids.size(0)

        more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
        less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
        
        loss = criterion(more_toxic_outputs, less_toxic_outputs, targets, margin)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(
            Epoch=epoch, 
            Valid_Loss=epoch_loss,
        )   
    
    gc.collect()
    
    return epoch_loss