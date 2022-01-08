# Config 설정
BERT_MODEL_LIST = {
    "hatebert":"GroNLP/hateBERT",
    "roberta":"roberta-base",
    "distilbert":"distilbert-base-cased",
    "electra":"google/electra-base-discriminator",
    "luke":"studio-ousia/luke-base",
    "deberta":"microsoft/deberta-v3-base",
    "bigbird_roberta":"google/bigbird-roberta-base",
    "t5":"t5-base",
    "bert":"bert-base-cased",
    "toxicbert":"unitary/toxic-bert"
}

SCHEDULER_LIST = {
    "cos_ann":'CosineAnnealingLR',
    "cos_ann_warm":'CosineAnnealingWarmRestarts',
    "none":'None',
}