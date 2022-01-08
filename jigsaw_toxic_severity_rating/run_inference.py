import os
import gc
import torch
import wandb
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from transformers import AutoTokenizer, AdamW
from jigsaw_toxic_severity_rating.medal_challenger.inference import bert_ensemble
from medal_challenger.dataset import JigsawDataset, prepare_loaders
from medal_challenger.model import JigsawModel
from medal_challenger.utils import id_generator, set_seed, get_dataframe
from medal_challenger.configs import BERT_MODEL_LIST, SCHEDULER_LIST

# 경고 억제
warnings.filterwarnings("ignore")

# CUDA가 구체적인 에러를 보고하도록 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main(args):

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

    wandb.login(key='27b98c412ec8a5a4e2a7433293569c9122e88fab')

    set_seed(CONFIG['seed'])

    HASH_NAME = id_generator(size=12)

    # 모델 저장 경로 
    root_save_dir = './'
    save_dir = os.path.join(root_save_dir,CONFIG['model_name'],CONFIG['scheduler'])
    assert os.path.isdir(save_dir), f""

    test_csv = "../input/jigsaw-toxic-severity-rating/comments_to_score.csv"
    test_df = get_dataframe(test_csv)
    test_loader = prepare_loaders(test_df,CONFIG,is_train=False)

    MODEL_PATHS = glob(f'{save_dir}/*')

    preds1 = bert_ensemble(MODEL_PATHS, test_loader, CONFIG)

    preds = (preds1-preds1.min())/(preds1.max()-preds1.min())

    sub_df = get_dataframe(test_csv)
    sub_df['score'] = preds
    sub_df[['comment_id', 'score']].to_csv("submission.csv", index=False)

    print(f'preds.shape : {preds.shape}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
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