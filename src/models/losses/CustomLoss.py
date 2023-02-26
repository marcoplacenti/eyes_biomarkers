import torch
from torch import nn

from torchvision.models import vgg16, VGG16_Weights
from torchvision.models._utils import IntermediateLayerGetter

from torchmetrics import StructuralSimilarityIndexMeasure

from pytorch_metric_learning import losses, miners

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomLoss(nn.Module):
    def __init__(self, loss_name):
        super(CustomLoss, self).__init__()

        assert loss_name in ['PerceptualLoss', 'SSIMLoss',
                            'Arcface', 'Triplet', 'Contrastive']
        self.loss_name = loss_name
        
        if self.loss_name == 'PerceptualLoss':
            self.mse_loss = nn.MSELoss()
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            return_layers = {
                                    '1': 'output1', 
                                    '6': 'output2', 
                                    '11': 'output3'
            }
            self.inter_layers = IntermediateLayerGetter(
                                            vgg.features, 
                                            return_layers=return_layers
            )

        if self.loss_name == 'SSIMLoss':
            self.ssim = StructuralSimilarityIndexMeasure()

        if self.loss_name == 'Arcface':
            self.cross_entropy_loss = nn.CrossEntropyLoss()

        if self.loss_name == 'Triplet':
            self.triplet_margin_loss = losses.TripletMarginLoss(margin=0.4)
            self.mining_func = miners.TripletMarginMiner(margin=0.4)

        if self.loss_name == 'Contrastive':
            self.cross_entropy_loss = nn.CrossEntropyLoss()
            self.T = 1.0

    def forward(self, x, y):
        if self.loss_name == 'PerceptualLoss':
            x = self.inter_layers(x)
            recon_x = self.inter_layers(y)
            assert list(x.keys()) == list(recon_x.keys())
            acc = 0
            for key in list(x.keys()):
                orig = x[key].reshape(x[key].size(0), -1)
                recon = recon_x[key].reshape(recon_x[key].size(0), -1)
                acc += self.mse_loss(orig, recon)
            return acc

        if self.loss_name == 'SSIMLoss':
            # if ssim=1->1-(1+1)/2=1-1=0, else if ssim=-1->1-(1-1)/2=1-0=1
            return 1 - (1 + self.ssim(x, y))/2
        
        if self.loss_name == 'Arcface':
            return self.cross_entropy_loss(x, y)
            
        if self.loss_name == 'Triplet':
            indices_tuple = self.mining_func(x, y)
            loss = self.triplet_margin_loss(x, y, indices_tuple)
            return loss

        if self.loss_name == 'Contrastive':
            # normalize
            q = nn.functional.normalize(x, dim=1)
            k = nn.functional.normalize(y, dim=1)
            # Einstein sum is more intuitive
            logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
            N = logits.shape[0]  # batch size per GPU
            labels = (torch.arange(N, dtype=torch.long) + N * 0)
            #print(labels)
            cel = self.cross_entropy_loss(logits.to(device), labels.to(device))
            loss = cel * (2 * self.T)
            return loss