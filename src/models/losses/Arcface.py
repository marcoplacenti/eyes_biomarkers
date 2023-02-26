import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg

class Arcface(nn.Module):
    def __init__(self, embeddings_size, num_classes, s=8, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.sin_m = torch.sin(torch.tensor(m))
        self.cos_m = torch.cos(torch.tensor(m))
        self.num_classes = num_classes
        self.fc = nn.Linear(embeddings_size, self.num_classes, bias=False)
        torch.nn.init.xavier_uniform(self.fc.weight)
        self.tanh = nn.Tanh()

    def forward(self, x, label=None):
        # Normalize the embeddings
        x = torch.nn.functional.normalize(x, dim=1)
        # Normalize the weights
        weights = torch.nn.functional.normalize(self.fc.weight, dim=1)
        # Calculate cosine similarity between embeddings and weights
        cosine = torch.matmul(x, weights.transpose(1, 0))
        # Calculate the arccos of the cosine similarity
        arc_cosine = torch.acos(cosine)
        print(arc_cosine)
        # Calculate the additivie margin
        additive_margin = self.s * torch.cos(arc_cosine + self.m)
        print(additive_margin)
        # Calculate the logits
        logits = cosine * self.s + additive_margin
        # Calculate the loss
        print(logits)
        print(label)
        return logits



        if not torch.all(torch.isnan(torch.flatten(x)) == False):
            print("NaN is in x")
            exit()
        w_L2 = linalg.norm(self.fc.weight, dim=1, keepdim=True)
        if not torch.all(torch.isnan(torch.flatten(w_L2)) == False):
            print("NaN is in w_L2")
            exit()
        norm_weights = self.fc.weight / w_L2
        if torch.flatten(norm_weights).min() < -1 or torch.flatten(norm_weights).max() > 1:
            print("norm_weights returns values < -1 or > 1")
            exit()
        x_L2 = linalg.norm(x, dim=1, keepdim=True)
        if not torch.all(torch.isnan(torch.flatten(x_L2)) == False):
            print("NaN is in x_L2")
            exit()
        norm_features = x / x_L2
        if torch.flatten(norm_features).min() < -1 or torch.flatten(norm_features).max() > 1:
            print("norm_features returns values < -1 or > 1")
            exit()
        dot_product = torch.einsum('bm,mc->bc', norm_features, norm_weights.T)
        #dot_product = self.tanh(dot_product)
        if torch.flatten(dot_product).min() < -1 or torch.flatten(dot_product).max() > 1:
            print("dot_product returns values < -1 or > 1")
            exit()
        theta = torch.acos(dot_product) # if logit=-1 -> 3.14, if logit=1 -> 0
        if not torch.all(torch.isnan(torch.flatten(theta)) == False):
            print("NaN is in theta")
            exit()
        one_hot = F.one_hot(label, num_classes=self.num_classes)
        m = one_hot * self.m
        cos_theta = self.s * torch.cos(theta + m) # if logit=3.14 -> -1, if logit=0 -> 1
        if not torch.all(torch.isnan(torch.flatten(cos_theta)) == False):
            print("NaN is in cos_theta")
            exit()
        return cos_theta
