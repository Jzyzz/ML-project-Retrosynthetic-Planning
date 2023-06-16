import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from validate import validate
from model import Model, CNNModel, MLP
import numpy as np

def train(args, train_loader, test_loader):
    model = MLP(input_dim=2048 * 10)
#     test_fp = test_fp.cuda()
#     test_values = test_values.cuda()
    criterion = nn.MSELoss()
    criterion = criterion.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model = model.cuda()
    model.train()
    weight_decay = 0.01

    for epoch in range(args.num_epochs):
        running_loss = 0.0

        num = 0
        for inputs, labels in train_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            outputs.reshape(-1, 1)
            # print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)
            
#             l2_reg = torch.tensor(0.).cuda()
#             for param in model.parameters():
#                 l2_reg += torch.norm(param, p=2)
#             loss += weight_decay * l2_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        
        print(f"Epoch {epoch+1}/{args.num_epochs} Loss: {running_loss, avg_loss}")
        if epoch % (args.num_epochs / 100) == 0:
            train_err = validate(train_loader, model)
            test_err = validate(test_loader, model)
            # test_loss = 0.0
            
#             for data, label in test_loader:
#                 data = data.cuda()
#                 label = label.cuda()
#                 output = model(data)

#                 test_loss += criterion(output, label).item()
#             output = model(test_fp)  
#             test_loss = criterion(output, test_values)
            print('%d, train_err: %.6f, test_err: %.6f' % (epoch * 100 / args.num_epochs, train_err, test_err))

    
    torch.save(model.state_dict(), 'save_model_eq1.pth')
    
