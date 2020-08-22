import numpy as np
import time
import argparse
import torch
from experiment_utils.logger import MlFlowLogger
import torch.optim as optim
import torch.nn.functional as F
import os
from model import CnnSentimentAnalysis

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, logger):
    print('Train step')
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        inputs, target = batch
        target = target.float()
        logits = model(inputs).squeeze().float()
        loss = F.binary_cross_entropy_with_logits(logits, target)
        acc = accuracy(logits, target)
        loss.backward()
        optimizer.step()
        logger.log_metrics({'accuracy' : acc.item(),
                            'loss'     : loss.item()})


def evaluate(model, iterator, logger):
    print('Eval step')
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            inputs, target = batch
            target = target.float()
            logits = model(inputs).squeeze().float()
            loss = F.binary_cross_entropy_with_logits(logits, target)
            acc = accuracy(logits, target)

            logger.log_metrics({'val_loss' : loss.item()})


def main(args):
    print('Step 0. Creating logger')
    logger = MlFlowLogger(experiment_name = 'MyCoolestExperiments!!')
    logger.start_session()

    print('Step 1. Entry in main')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log_params({'device':device})

    print('Step 2. Load datasets')
    train_iter = torch.load(args.train_iter, map_location=device)
    val_iter = torch.load(args.val_iter, map_location=device)

    print(train_iter)
    print()
    print(val_iter)

    print('Step 3. Load model and components')

    vocab_size = 148794
    output_size = 1
    embedding_dim = 100
    hidden_dim = 64
    kernel_size = [2]
    dropout = 0.5

    cls = CnnSentimentAnalysis(input_dim = vocab_size,
                                 embedding_dim = embedding_dim,
                                 hidden_dim = hidden_dim,
                                 output = output_size,
                                 kernel_size = kernel_size,
                                 dropout = dropout)

    optimizer = optim.Adam(cls.parameters(), lr = 0.001)

    cls = cls.to(device)
    print(cls)
    print()

    print('Step 4. Log parameters')

    logger.log_params({
        'vocab_size':vocab_size,
        'output_size': output_size,
        'embedding_dim': embedding_dim,
        'hidden_dim':hidden_dim,
        'kernel_size':kernel_size,
        'dropout': dropout})

    print('Step 5. Start training')
    N_EPOCHS = 1
    for epoch in range(N_EPOCHS):
        logger.log_metrics({'epoch':epoch})
        train(cls, train_iter, optimizer, logger)
        #evaluate(cls, val_iter, logger)

    print('Step 6. Log model')
    logger.log_model(cls,'models')
    logger.end_session()


if __name__ == '__main__':

    print('Start MlTrainPipeline')
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_iter', default = '/home/alexander/airflow/src/data_loaders/train_iter')
    parser.add_argument('--val_iter', default = '/home/alexander/airflow/src/data_loaders/val_iter')

    args = parser.parse_args()
    main(args)
    print('End MlTrainPipeline')
