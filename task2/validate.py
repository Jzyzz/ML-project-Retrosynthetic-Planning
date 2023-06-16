import torch
import numpy as np

def error(output, label):
    return torch.sum(torch.abs(output - label) / (label + 1e-7))

def validate(test_loader, model):
    loss = 0.0
    num = 0
    for data, label in test_loader:
        data = data.cuda()
        label = label.cuda()
        output = model(data)
        num += len(data)
        err = error(output, label)
    return err / num