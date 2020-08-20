import os
import numpy as np
import torch
import pandas as pd
from collections import Counter
from string import punctuation
from torch.utils.data import DataLoader, TensorDataset
from utils.helpers import *


def main():

    BATCH_SIZE = 50
    TRAIN_PATH = '/home/alexander/airflow/src/data/train.csv'
    TEST_PATH = '/home/alexander/airflow/src/data/test.csv'

    test_data_sub = pd.read_csv(TEST_PATH)
    train_data = pd.read_csv(TRAIN_PATH)

    print(train_data.head(1))

    reviews=train_data['review'].values
    labels=train_data['sentiment'].values
    input_test=test_data_sub['review'].values
    y_test=list()

    all_reviews, all_words=review_formatting(reviews)
    count_words = Counter(all_words)
    total_words=len(all_words)
    sorted_words=count_words.most_common(total_words)
    vocab_to_int={w:i+1 for i,(w,c) in enumerate(sorted_words)}

    features=preprocess(reviews, vocab_to_int)
    train_x=features[:int(0.90*len(features))]
    train_y=labels[:int(0.90*len(features))]
    valid_x=features[int(0.90*len(features)):]
    valid_y=labels[int(0.90*len(features)):]

    #create Tensor Dataset
    train_data=TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data=TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))

    train_loader=DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader=DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)

    print('Return 2 data loaders: train and valid')


    return {'train_iter':train_loader, 'val_iter':valid_loader}

if __name__ == '__main__':
    main()


# def test_me():
#     data = ['WoW' for i in range(3)]
#     for sample in data:
#         print(sample)
#         print()
#     return data
#
# if __name__ == '__main__':
#     test_me()
