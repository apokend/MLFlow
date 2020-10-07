import os
import numpy as np
import torch
import pandas as pd
from collections import Counter
from string import punctuation
from torch.utils.data import DataLoader, TensorDataset
from utils.helpers import *


def main(my_path_to_data, **kwargs):

    BATCH_SIZE = kwargs['batch_size']
    #BATCH_SIZE = 256
    print(BATCH_SIZE)

    train_data = pd.read_csv(os.path.join(my_path_to_data, 'train.csv'))
    print('Train data successfully loaded')

    test_data_sub=pd.read_csv(os.path.join(my_path_to_data, 'test.csv'))

    print('Test data successfully loaded')

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
    my_path_to_data = '/home/alexander/Документы/test_airflow_docker/src/data'
    main(my_path_to_data)
