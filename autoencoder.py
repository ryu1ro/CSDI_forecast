import torch.nn as nn

class AutoEncoder_base(nn.Module):
    def __init__(self, d_in, d_enc=64, d1=128, d2=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_in, d1),
            nn.Tanh(),
            nn.Linear(d1, d2),
            nn.ReLU(),
            nn.Linear(d2, d1),
            nn.ReLU(),
            nn.Linear(d1, d_enc)
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_enc, d1),
            nn.ReLU(),
            nn.Linear(d1, d2),
            nn.ReLU(),
            nn.Linear(d2, d1),
            nn.Tanh(),
            nn.Linear(d1, d_in)
        )


    def forward(self, x):
        return self.decoder(self.encoder(x))