import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, x_dim, out_dim):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            # nn.Linear(2048, 1024),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            # nn.Linear(512, 512),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(1024, out_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
    