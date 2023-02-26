import torch
from torch import nn
from torch.nn import functional as F
torch.backends.cudnn.deterministic = True

# Architecture taken from here: https://arxiv.org/pdf/2110.01534.pdf

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 9, 9) #16,28,28

class AE(nn.Module):
    def __init__(self, z_dim=4096):
        super(AE, self).__init__()

        kernel_size = 3
        stride = 2

        self.encoder_convolutions = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, track_running_stats=False),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=3, padding=1),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(),
            Flatten()
        )

        input = torch.rand(4, 3, 224, 224) # 3,224,224
        h_dim = self.encoder_convolutions(input).shape[-1]

        self.fc_layers_encoder = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.BatchNorm1d(z_dim, track_running_stats=False),
            nn.LeakyReLU()
        )

        self.mu = nn.Linear(z_dim, z_dim)

        self.fc_layers_decoder = nn.Sequential(
            nn.BatchNorm1d(z_dim, track_running_stats=False),
            nn.LeakyReLU(),
            nn.Linear(z_dim, h_dim)
        )
        
        self.decoder_convolutions = nn.Sequential(
            UnFlatten(),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=3, padding=1, output_padding=2),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16, track_running_stats=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid(),
        )
    
    def bottleneck(self, h):
        return self.mu(h)

    def encode(self, x):
        x = self.encoder_convolutions(x)
        h = self.fc_layers_encoder(x)
        mu = self.bottleneck(h)
        return mu

    def decode(self, z):
        z = self.fc_layers_decoder(z)
        z = self.decoder_convolutions(z)
        return z

    def forward(self, x):
        mu = self.encode(x)
        z = self.decode(mu)
        return z, x, mu, None

