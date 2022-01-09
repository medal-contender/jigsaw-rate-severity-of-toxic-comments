# 디렉토리 생성 
mkdir models
cd ./models

# 모델 레포지토리 클론 
git lfs install
# HateBERT
git clone https://huggingface.co/GroNLP/hateBERT
# RoBERTa
git clone https://huggingface.co/roberta-base
# DistilBERT
git clone https://huggingface.co/distilbert-base-cased
# Electra
git clone https://huggingface.co/google/electra-base-discriminator
# LUKE
git clone https://huggingface.co/studio-ousia/luke-base
# DeBERTa
git clone https://huggingface.co/microsoft/deberta-v3-base
# Bigbird-RoBERTa
git clone https://huggingface.co/google/bigbird-roberta-base
# T5
git clone https://huggingface.co/t5-base
# BERT
git clone https://huggingface.co/bert-base-cased
# ToxicBERT
git clone https://huggingface.co/unitary/toxic-bert

GIT_LFS_SKIP_SMUDGE=1

