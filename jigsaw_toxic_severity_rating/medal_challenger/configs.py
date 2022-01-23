# Config 설정
BERT_MODEL_LIST = {
    "hatebert": "hateBERT",
    "roberta": "roberta-base",
    "roberta-large": "roberta-large",
    "funnel": "large",
    "muppet": "muppet-roberta-base",
    "distilbert": "distilbert-base-cased",
    "electra": "electra-base-discriminator",
    "luke": "luke-base",
    "deberta": "deberta-v3-base",
    "bigbird_roberta": "bigbird-roberta-base",
    "t5": "t5-base",
    "bert": "bert-base-cased",
    "toxicbert": "toxic-bert",
    "toxic-roberta": "unbiased-toxic-roberta"
}

SCHEDULER_LIST = {
    "cos_ann": 'CosineAnnealingLR',
    "cos_ann_warm": 'CosineAnnealingWarmRestarts',
    "none": 'None',
}
