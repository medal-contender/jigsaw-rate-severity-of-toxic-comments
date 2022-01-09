import os
import gc
import warnings
import wandb
import time
import copy
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AdamW
from collections import defaultdict
from medal_challenger.utils import id_generator, set_seed, get_dataframe, get_folded_dataframe
from medal_challenger.configs import BERT_MODEL_LIST, SCHEDULER_LIST
from medal_challenger.dataset import prepare_loaders
from medal_challenger.model import JigsawModel, fetch_scheduler
from medal_challenger.train import train_one_epoch, valid_one_epoch
from colorama import Fore, Style

blue_font = Fore.BLUE
yellow_font = Fore.YELLOW
reset_all = Style.RESET_ALL

# 경고 억제
warnings.filterwarnings("ignore")

# CUDA가 구체적인 에러를 보고하도록 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def run_training(
        model, 
        optimizer, 
        scheduler, 
        device, 
        num_epochs, 
        fold,
        save_dir,
        train_loader,
        valid_loader,
        run,
        CONFIG,
    ):

    # 자동으로 Gradients를 로깅
    wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss = train_one_epoch(
                                model, 
                                optimizer, 
                                scheduler, 
                                dataloader=train_loader, 
                                device=device, 
                                epoch=epoch,
                                CONFIG=CONFIG,
                            )
        
        val_epoch_loss = valid_one_epoch(
                            model, 
                            valid_loader, 
                            device=device, 
                            epoch=epoch,
                            margin=CONFIG['margin']
                        )
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        
        # Loss 로깅
        wandb.log({"Train Loss": train_epoch_loss})
        wandb.log({"Valid Loss": val_epoch_loss})
        
        # 베스트 모델 저장
        if val_epoch_loss <= best_epoch_loss:
            print(f"{blue_font}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            run.summary["Best Loss"] = best_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"{save_dir}/Loss-Fold-{fold}.bin"
            # 모델 저장
            torch.save(model.state_dict(), PATH)
            print(f"Model Saved{reset_all}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))
    
    # 베스트 모델 로드
    model.load_state_dict(best_model_wts)
    
    return model, history


def main(args):

    # TODO: Config로 받을지 Args로 받을지 결정해야 함 

    # Args로 Config 설정
    CONFIG = {
        "seed": args.seed, 
        "epochs": args.epochs, 
        "model_name": BERT_MODEL_LIST[args.model_name],
        "train_batch_size": args.train_batch_size,
        "valid_batch_size": args.valid_batch_size,
        "test_batch_size": args.test_batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "scheduler": SCHEDULER_LIST[args.scheduler],
        "min_lr": args.min_lr,
        "T_max": args.T_max,
        "weight_decay": args.weight_decay,
        "n_fold": args.n_fold, 
        "n_accumulate": args.n_accumulate,
        "num_classes": args.num_classes,
        "margin": args.margin,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "hash_name": args.hash_name
    }
    print(os.getcwd())
    CONFIG["tokenizer"] = AutoTokenizer.from_pretrained(f"../models/{CONFIG['model_name']}")
    CONFIG['group'] = f'{args.hash_name}-{args.model_name}-Baseline'

    if "deberta" in CONFIG['model_name']:
        CONFIG['train_batch_size'] = int(CONFIG['train_batch_size']/2)
        CONFIG['valid_batch_size'] = int(CONFIG['valid_batch_size']/2)
        CONFIG['test_batch_size'] = int(CONFIG['test_batch_size']/2)
    wandb.login(key=args.wandb_key)

    set_seed(CONFIG['seed'])

    HASH_NAME = id_generator(size=12)

    # 모델 저장 경로 
    root_save_dir = './trained'
    save_dir = os.path.join(root_save_dir,CONFIG['model_name'],CONFIG['scheduler'])
    os.makedirs(save_dir,exist_ok=True)

    # 데이터 경로 
    root_data_dir = '../input/jigsaw-toxic-severity-rating'
    train_csv = os.path.join(root_data_dir,'validation_data.csv')   
    test_csv = os.path.join(root_data_dir,'comments_to_score.csv')
    submission_csv = os.path.join(root_data_dir,'sample_submission.csv')
    
    # 데이터프레임
    train_df = get_dataframe(train_csv)
    test_df = get_dataframe(test_csv)
    submission_df = get_dataframe(submission_csv)

    # K Fold
    train_df = get_folded_dataframe(train_df, CONFIG['n_fold'], CONFIG['seed'])

    # 학습 진행

    for fold in range(0, CONFIG['n_fold']):
        print(f"{yellow_font}====== Fold: {fold} ======{reset_all}")
        run = wandb.init(project='Jigsaw', 
                        config=CONFIG,
                        job_type='Train',
                        group=CONFIG['group'],
                        tags=['roberta-base', f'{HASH_NAME}', 'margin-loss'],
                        name=f'{HASH_NAME}-fold-{fold}',
                        anonymous='must')
        
        train_loader, valid_loader = prepare_loaders(train_df,CONFIG,fold,True)
        
        model = JigsawModel(f"../models/{CONFIG['model_name']}",CONFIG['num_classes'])
        model.to(CONFIG['device'])
        
        optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        scheduler = fetch_scheduler(optimizer,CONFIG)
        
        model, history = run_training(
                            model, 
                            optimizer, 
                            scheduler,
                            device=CONFIG['device'],
                            num_epochs=CONFIG['epochs'],
                            fold=fold,
                            save_dir=save_dir,
                            train_loader=train_loader,
                            valid_loader=valid_loader,
                            run=run,
                            CONFIG=CONFIG,
                        )
        
        run.finish()
        
        del model, history, train_loader, valid_loader
        _ = gc.collect()
        print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-key", 
        type=str, 
        default='', 
        help='Type Authentication Key For WANDB.'
    )
    parser.add_argument(
        "--hash-name", 
        type=str, 
        default='jigsaw', 
        help='Type Name Of Your Project For WANDB.'
    )
    parser.add_argument(
        "--model-name", 
        choices=[
            "hatebert","roberta","distilbert","electra","luke",
            "deberta","bigbird_roberta","t5","bert","toxicbert",
        ], 
        default='bert', 
        help='Select Model From ["hatebert","roberta","distilbert","electra","luke","deberta","bigbird_roberta","t5","bert","toxicbert",].'
    )
    parser.add_argument(
        "--scheduler", 
        choices=[
            "cos_ann","cos_ann_warm","none",
        ], 
        default='none', 
        help='Select Scheduler From ["cos_ann","cos_ann_warm","none"].'
    )
    parser.add_argument(
        '--max-length', 
        type=int, 
        default=128,
        help="Type The Maximum Length Of Sequnce."
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10,
        help="Type The Number Of Epochs For Training."
    )
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=1e-4,
        help="Type Learning Rate For Training."
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help="Type Seed For Program."
    )
    parser.add_argument(
        '--min-lr', 
        type=float, 
        default=1e-6,
        help="Type The Minimum Learning Rate For Training."
    )
    parser.add_argument(
        '--T-max', 
        type=int, 
        default=500,
        help="Type The Maximum Number Of Iterations For One Cycle."
    )
    parser.add_argument(
        '--weight-decay', 
        type=float, 
        default=1e-6,
        help="Type Weight Decay Value."
    )
    parser.add_argument(
        '--n-fold', 
        type=int, 
        default=2,
        help="Type The Number Of Folds."
    )
    parser.add_argument(
        '--n-accumulate', 
        type=int, 
        default=1,
        help="Type The Size Of Step Stack."
    )
    parser.add_argument(
        '--num-classes', 
        type=int, 
        default=1,
        help="Type The Number Of Classes."
    )
    parser.add_argument(
        '--margin', 
        type=float, 
        default=0.5,
        help="Type Margin Value For Loss Function."
    )
    parser.add_argument(
        '--train-batch-size', 
        type=int, 
        default=16,
        help="Type Train Batch Size."
    )
    parser.add_argument(
        '--valid-batch-size', 
        type=int, 
        default=64,
        help="Type Validation Batch Size."
    )
    parser.add_argument(
        '--test-batch-size', 
        type=int, 
        default=64,
        help="Type Test Batch Size."
    )

    args = parser.parse_args()

    main(args)