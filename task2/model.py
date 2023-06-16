import torch.nn as nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(2048, 2048 * 2048 * 3)
        self.fc = nn.Linear(2048, 1)
        self.resnet = models.resnet50(pretrained=True)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 3, 2048, 2048)
        x = self.resnet(x)
        x = self.fc(x)
        return x

def init_weights(module):  
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
        
class block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential (
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.95, affine=True, track_running_stats=True),
            
        )
        self.apply(init_weights)

    def forward(self, X):
        return self.net(X)
    

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.feature = nn.Sequential(
            block(1, 4),
            block(4, 8),
            nn.AdaptiveAvgPool2d(output_size=(4, 4))
        )
        self.fc = nn.Linear(8 * 4 * 4, 1)
        
        self.apply(init_weights)

    def forward(self, input_data):
        input_data = self.pre(input_data).view(-1, 1, 256, 256)
        feature = self.feature(input_data)
        feature = feature.view(-1, 8 * 4 * 4)
        output = self.fc(feature)
        
        return output
    
class MLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x