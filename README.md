# Jigsaw Rate Severity Of Toxic Comments

---
<p align="center">
  <img src="./images/jigsaw.jpg" width=550>
</p>

---

## Members

```
- Jeongwon Kim (kimkim031@naver.com)
- Jaewoo Park (jerife@naver.com)
- Youngmin Paik (ympaik@hotmail.com)
- Kyubin Kim (kimkyu1515@naver.com)
```

---

## Competition Overview

In this competition, we will be asking you to score a set of about fourteen thousand comments. Pairs of comments were presented to expert raters, who marked one of two comments more harmful — each according to their own notion of toxicity. In this contest, when you provide scores for comments, they will be compared with several hundred thousand rankings. Your average agreement with the raters will determine your individual score. In this way, we hope to focus on ranking the severity of comment toxicity from innocuous to outrageous, where the middle matters as much as the extremes.

### Dataset

- Ruddit Dataset
- Jigsaw Rate Severity of Toxic Comments
- toxic-task

### Due Date

- Team Merge Deadline - January 31, 2022
- Submission Deadline - February 7, 2022

---

## Program

- Fetch Pretrained Models

```shell
$ sh ./download_pretrained_models.sh
```

- Train

```shell
$ cd /jigsaw-toxic-severity-rating/jigsaw_toxic_severity_rating
$ python3 run_train.py \
          "--config-file", "/jigsaw-toxic-severity-rating/configs/roberta.yaml", \
          "--train", \
          "--training-keyword", "roberta-test"
```          
You can set `launch.json` for vscode as follow:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "run_train.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/jigsaw_toxic_severity_rating",
            "args": ["--config-file", "../configs/roberta.yaml", "--train"]
        }
    ]
}
```

## Requirements

### Environment

- Python 3.7 (To match with Kaggle environment)
- Conda
- git
- git-lfs
- CUDA 11.3 + PyTorch 1.10.1

Pytorch version may vary depanding on your hardware configurations.

### Installation with virtual environment

```bash
git clone https://github.com/medal-contender/nbme-score-clinical-patient-notes.git
conda create -n nbme python=3.7
conda activate nbme
# PyTorch installation process may vary depending on your hardware
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
cd nbme-score-clinical-patient-notes
pip install -r requirements.txt
```

Run this in WSL (or WSL2)

```bash
./download_pretrained_models.sh
```

### To update the code

```bash
$ git pull
```

If you have local changes, and it causes to abort `git pull`, one way to get around this is the following:

```bash
# removing the local changes
$ git stash
# update
$ git pull
# put the local changes back on top of the recent update
$ git stash pop
```
