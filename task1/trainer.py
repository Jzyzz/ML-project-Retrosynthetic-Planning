import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.distributions import normal
from itertools import repeat
from model import MLPClassifier
from dataloader import load_from_csv, RPDataset


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class RetroPredTrainer():
    def __init__(self):
        super().__init__()
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = MLPClassifier(2048, 11858).to(self.device)
        self.train_dataset = RPDataset(load_from_csv('ML/schneider50k/raw_train.csv'))
        self.valid_dataset = RPDataset(load_from_csv('ML/schneider50k/raw_test.csv'))
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=128, shuffle=True)
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=128, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0001)
        
        self.len_epoch = len(self.train_loader)
        # self.train_loader = inf_loop(self.train_loader)
                

    def train(self, num_epoch):
        for epoch in range(num_epoch):
            loss = 0
            acc = 0
            for batch_idx, (data, template) in enumerate(self.train_loader,0):
                # print(template)
                data = data.to(self.device).float()
                template = template.to(self.device)
                # train the model
                self.model.train()

                self.optimizer.zero_grad()
                outputs = self.model(data).to(self.device)
                loss = self.criterion(outputs, template)
                loss.backward()
                self.optimizer.step()
                _,indices = torch.max(outputs, 1)
                correct = indices == template

            # evaluate on training dataset
            train_acc = correct.bool().float().mean().item()
            print("train acc = ", "%.4f" % train_acc, "train loss = ", "%.4f" % loss.item())
            
            # if (epoch+1) % 5 == 0:
            val_acc = self.evaluate()
            print("test acc = ", "%.4f" % val_acc, "test loss = ", "%.4f" % loss.item())
                

    def evaluate(self):
        self.model.eval()
        mean_acc = 0.0
        
        with torch.no_grad():
            for batch_idx, (data, template) in enumerate(self.valid_loader):
                data = data.to(self.device).float()
                template = template.to(self.device)
                output = self.model(data).to(self.device)
                loss = self.criterion(output, template)
                _,indices = torch.max(output, 1)
                correct = indices == template
                
        mean_acc = correct.bool().float().mean().item()
        return mean_acc