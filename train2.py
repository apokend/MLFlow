import numpy as np
import time
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import CnnSentimentAnalysis
import os
import mlflow
from mlflow.models.signature import infer_signature


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, device):
    print('Train step')
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        inputs, target = batch
        inputs = inputs.to(device)
        target = target.float().to(device)
        logits = model(inputs).squeeze().float()
        loss = F.binary_cross_entropy_with_logits(logits, target)
        acc = accuracy(logits, target)
        loss.backward()
        optimizer.step()
        mlflow.log_metric('accuracy',  acc.item())
        mlflow.log_metric('loss',      loss.item())


def main(**kwargs):

#-----------------------------------------------------------
    print('Step 1. Entry in main')

    import mlflow
    import mlflow.pytorch


    mlflow.set_tracking_uri('http://mlflow:5000')
    mlflow.set_experiment(kwargs['experiment_name'])

    with mlflow.start_run(run_name="Demo run"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mlflow.log_param('device', device)
#-----------------------------------------------------------



#-----------------------------------------------------------
        print('Step 1. Load datasets')

        train_iter = kwargs['train_iter']
        val_iter = kwargs['val_iter']

#-----------------------------------------------------------



#-----------------------------------------------------------
        print('Step 2. Load model and components')

        cls = kwargs['model'].to(device)
        optimizer = optim.Adam(cls.parameters(), lr = 0.001)

#-----------------------------------------------------------



#-----------------------------------------------------------
        print('Step 3. Log parameters')

        params = {'input_dim':      kwargs['input_dim'],
                  'output_size':    kwargs['output'],
                  'embedding_dim':  kwargs['embedding_dim'],
                  'hidden_dim':     kwargs['hidden_dim'],
                  'kernel_size':    kwargs['kernel_size'],
                  'dropout':        kwargs['dropout'],
                  'batch_size':     kwargs['batch_size']}

        for elem, value in params.items():
            mlflow.log_param(elem, value)

        mlflow.log_param('batch_size', kwargs['batch_size'])
#-----------------------------------------------------------



#-----------------------------------------------------------
        print('Step 5. Start training')
        N_EPOCHS = 1
        for epoch in range(N_EPOCHS):
            mlflow.log_metric('epoch', epoch)
            train(cls, train_iter, optimizer, device)
#-----------------------------------------------------------



#-----------------------------------------------------------
        print('Step 6. Save artifact')

        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        with open('outputs/test.txt', 'w') as f:
            f.write('Hi my baby!')
        mlflow.log_artifact('outputs')

        # sample, answer = iter(train_iter).next()
        #
        # cls = cls.to('cpu')
        #
        # signature = infer_signature(sample.cpu().detach().numpy(),
        #                             cls(sample.to('cpu')).detach().numpy())
        #
        # sample = {'msg':'Some cool text'}
        # output = {'output':float}
        mlflow.pytorch.log_model(cls,
                                 'models',
                                  registered_model_name = kwargs['model_name'],
                                  code_paths=[os.path.join(os.getcwd(),'src/model.py')])
                                  
#-----------------------------------------------------------
