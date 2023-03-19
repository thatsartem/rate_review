import torch
import torch.nn as nn

pretrained_embeddings = torch.load('embeddings.pth')

class CNNModel1(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self.cnn = nn.Sequential(
            nn.Conv1d(pretrained_embeddings.shape[1], hidden_size, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv1d(hidden_size,hidden_size,kernel_size=3,padding=1, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.clf = nn.Sequential(
            nn.Linear(hidden_size,30),
            nn.ReLU(),
            nn.Linear(30,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embeddings(x)
        x = x.permute(0,2,1)
        x = self.cnn(x)
        predictions = self.clf(x)
        return predictions

    def forward(self, x):
        x = self.embeddings(x)
        x = x.permute(0,2,1)
        x = self.cnn(x)
        predictions = self.clf(x)
        return predictions
    
class CNNModel2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self.cnn = nn.Sequential(
            nn.Conv1d(pretrained_embeddings.shape[1], hidden_size, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv1d(hidden_size,hidden_size,kernel_size=3,padding=1, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.clf = nn.Sequential(
            nn.Linear(hidden_size,10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.embeddings(x)
        x = x.permute(0,2,1)
        x = self.cnn(x)
        predictions = self.clf(x)*torch.tensor(10)
        return predictions