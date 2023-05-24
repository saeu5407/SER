import os
import pandas as pd
from sklearn.model_selection import train_test_split

def make_trainset_n_split(dataset_path):

    train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    train_df.path = train_df.path.map(lambda x: os.path.join(dataset_path, 'train', x.split('/')[-1]))

    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_df.path = train_df.path.map(lambda x: os.path.join(dataset_path, 'test', x.split('/')[-1]))

    train, valid = train_test_split(train_df, test_size=0.2, random_state=101, stratify=train_df["label"])
    train.to_csv(f"{dataset_path}/train_split.csv", index=False)
    valid.to_csv(f"{dataset_path}/valid_split.csv", index=False)
    test_df.to_csv(f"{dataset_path}/test_split.csv", index=False)

if __name__ == '__main__':

    default_path = os.getcwd().split(os.path.sep + 'src')[0]
    dataset_path = os.path.join(default_path, 'datasets')

    make_trainset_n_split(dataset_path)