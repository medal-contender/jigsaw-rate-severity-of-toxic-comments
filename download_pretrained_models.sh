# 디렉토리 생성
mkdir models
cd ./models

# 모델 레포지토리 클론
apt-get install git-lfs
# HateBERT
git clone https://huggingface.co/GroNLP/hateBERT
cd ./hateBERT
GIT_LFS_SKIP_SMUDGE=1
cd ..
# RoBERTa

git clone https://huggingface.co/roberta-base
cd ./roberta-base
GIT_LFS_SKIP_SMUDGE=1
cd ..
# DistilBERT
git clone https://huggingface.co/distilbert-base-cased
cd ./distilbert-base-cased
GIT_LFS_SKIP_SMUDGE=1
cd ..
# Electra
git clone https://huggingface.co/google/electra-base-discriminator
cd ./electra-base-discriminator
GIT_LFS_SKIP_SMUDGE=1
cd ..
# LUKE
git clone https://huggingface.co/studio-ousia/luke-base
cd ./luke-base
GIT_LFS_SKIP_SMUDGE=1
cd ..
# DeBERTa
git clone https://huggingface.co/microsoft/deberta-v3-base
cd ./deberta-v3-base
GIT_LFS_SKIP_SMUDGE=1
cd ..
# Bigbird-RoBERTa
git clone https://huggingface.co/google/bigbird-roberta-base
cd ./bigbird-roberta-base
GIT_LFS_SKIP_SMUDGE=1
cd ..
# T5
git clone https://huggingface.co/t5-base
cd ./t5-base
GIT_LFS_SKIP_SMUDGE=1
cd ..
# BERT
git clone https://huggingface.co/bert-base-cased
cd ./bert-base-cased
GIT_LFS_SKIP_SMUDGE=1
cd ..
# ToxicBERT
git clone https://huggingface.co/unitary/toxic-bert
cd ./toxic-bert
GIT_LFS_SKIP_SMUDGE=1
cd ..


