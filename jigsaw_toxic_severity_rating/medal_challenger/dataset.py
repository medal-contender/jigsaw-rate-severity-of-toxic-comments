import torch
from torch.utils.data import Dataset, DataLoader

class JigsawDataset(Dataset):

    def __init__(self, df, tokenizer, max_length, is_train=True):
        super().__init__()
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.is_train = is_train
        if is_train:
            self.more_toxic = df['more_toxic'].values
            self.less_toxic = df['less_toxic'].values
        else:
            self.text = df['text'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):

        # 학습 시 데이터
        if self.is_train:
            more_toxic = self.more_toxic[index]
            less_toxic = self.less_toxic[index]

            '''
                encode_plus는 문장을 토크나이징 합니다.
                add_special_tokens를 True로 설정하면
                토큰의 시작점에 '[CLS]' 토큰을 붙이고, 
                토큰의 마지막 지점에 '[SEP]' 토큰을 붙입니다.

            '''
            inputs_more_toxic = self.tokenizer.encode_plus(
                                    more_toxic,
                                    truncation=True,
                                    add_special_tokens=True,
                                    max_length=self.max_len,
                                    padding='max_length'
                                )
            inputs_less_toxic = self.tokenizer.encode_plus(
                                    less_toxic,
                                    truncation=True,
                                    add_special_tokens=True,
                                    max_length=self.max_len,
                                    padding='max_length'
                                )

            # MarginRankingLoss의 Target
            target = 1
            
            # Vocab에 대한 인덱스와 어텐션 마스크를 생성
            more_toxic_ids = inputs_more_toxic['input_ids']
            more_toxic_mask = inputs_more_toxic['attention_mask']
            
            less_toxic_ids = inputs_less_toxic['input_ids']
            less_toxic_mask = inputs_less_toxic['attention_mask']
            
            
            return {
                'more_toxic_ids': torch.tensor(more_toxic_ids, dtype=torch.long),
                'more_toxic_mask': torch.tensor(more_toxic_mask, dtype=torch.long),
                'less_toxic_ids': torch.tensor(less_toxic_ids, dtype=torch.long),
                'less_toxic_mask': torch.tensor(less_toxic_mask, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long)
            }
        
        # 테스트 시 데이터
        else:
            text = self.text[index]
            inputs = self.tokenizer.encode_plus(
                            text,
                            truncation=True,
                            add_special_tokens=True,
                            max_length=self.max_len,
                            padding='max_length'
                        )
            
            ids = inputs['input_ids']
            mask = inputs['attention_mask']        
            
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long)
            }    

def prepare_loaders(df, cfg, fold=1, is_train=True):
    '''
        데이터로더를 반환합니다.
        학습 시:
            데이터프레임의 kfold 컬럼을 기준으로 학습셋과 평가셋을 분할합니다.
        테스트 시:
            text 데이터를 불러오는 데이터 로더를 반환합니다.
    '''
    if is_train:
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        
        train_dataset = JigsawDataset(
            df_train, 
            tokenizer=cfg.tokenizer, 
            max_length=cfg.model_param.max_length
        )
        valid_dataset = JigsawDataset(
            df_valid, 
            tokenizer=cfg.tokenizer, 
            max_length=cfg.model_param.max_length
        )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.train_param.batch_size,
            num_workers=cfg.train_param.num_workers, 
            shuffle=cfg.train_param.shuffle, 
            pin_memory=cfg.train_param.pin_memory, 
            drop_last=cfg.train_param.drop_last
        )
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=cfg.valid_param.batch_size, 
            num_workers=cfg.valid_param.num_workers, 
            shuffle=cfg.valid_param.shuffle, 
            pin_memory=cfg.valid_param.pin_memory
        )
        
        return train_loader, valid_loader

    else:
        test_dataset = JigsawDataset(
                        df, 
                        cfg.tokenizer, 
                        max_length=cfg.model_param.max_length, 
                        is_train=False
        )
        test_loader = DataLoader(
                        test_dataset, 
                        batch_size=cfg.infer_param.batch_size,
                        num_workers=cfg.infer_param.num_workers, 
                        shuffle=cfg.infer_param.shuffle, 
                        pin_memory=cfg.infer_param.pin_memory
                    )
        return test_loader
