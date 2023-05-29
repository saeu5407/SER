import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig

from src.dataset import SERDataset, collate_fn, collate_fn_test
from src.utils import make_trainset_n_split
from src.modeling import Wav2Vec2ForSpeechClassification
from src.train import train
from src.models import BaseModel

if __name__ == '__main__':

    default_path = os.getcwd().split(os.path.sep + 'src')[0]
    dataset_path = os.path.join(default_path, 'datasets')

    # split datasets
    make_trainset_n_split(dataset_path)

    # load datasets & dataloader
    train_datasets = SERDataset(data_path=f"{dataset_path}/train_split.csv")
    valid_datasets = SERDataset(data_path=f"{dataset_path}/valid_split.csv")

    train_dataloader = DataLoader(train_datasets, batch_size=8, shuffle=True,
                                  collate_fn=collate_fn, num_workers=2)
    valid_dataloader = DataLoader(valid_datasets, batch_size=8, shuffle=True,
                                  collate_fn=collate_fn, num_workers=2)
    # config
    """
    model_name_or_path = "facebook/wav2vec2-base-960h"  # jonatasgrosman/wav2vec2-large-xlsr-53-english 컴퓨터 성능좋으면 이거하자
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(train_datasets.data.label.unique()),
        finetuning_task="wav2vec2_clf",
    )
    setattr(config, 'pooling_mode', "mean")  # 객체 내부의 속성을 변경해주는 파이썬 내장 함수로 config 파일에 pooling_mode를 추가한 것

    # load model & freeze feature extractor
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )
    model.freeze_feature_extractor()
    """
    model = BaseModel()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f">>> Use {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_model = train(model=model, device=device,
                       train_loader=train_dataloader, valid_loader=valid_dataloader,
                       optimizer=optimizer, scheduler=scheduler, epochs=100,
                       default_path=default_path)

    # predict test data