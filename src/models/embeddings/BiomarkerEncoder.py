import torch
from torch import nn
from torchmetrics import Accuracy
import pytorch_lightning as pl

import wandb
from itertools import chain

from models.embeddings.AE import AE
from models.embeddings.VAE import VAE
from models.embeddings.inceptionv3 import InceptionV3_299
from models.embeddings.moco import MoCo_ResNet

from models.losses.CustomLoss import CustomLoss
from models.losses.Arcface import Arcface
from models.losses.metrics import AddMarginProduct

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiomarkerEncoder(pl.LightningModule):
    def __init__(self, model, loss_name, num_classes=None):
        super(BiomarkerEncoder, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.loss_name = loss_name

        self.set_opt_and_loss_func_()

        self.save_hyperparameters(ignore=['model'])

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self.model.train()
        metrics_dict = {}
        if isinstance(self.model, AE) or isinstance(self.model, VAE):
            _, inputs, _ = batch
            embeddings = self(inputs)
            recon_x, x, mu, logvar = embeddings
            loss = self.loss_fn(x, recon_x)
            if isinstance(self.model, VAE):
                kld = (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
                loss = loss + kld
                metrics_dict['KLD'] = kld

            metrics_dict['loss'] = loss

        elif isinstance(self.model, InceptionV3_299):
            _, inputs, targets = batch
            if not torch.all(torch.isnan(torch.flatten(inputs)) == False):
                print("NaN is in data")
                exit()
            #for name, param in self.model.named_parameters():
            #    self.print_nan_gradients(self.model)
            embeddings = self(inputs)
            #print(embeddings)
            if not torch.all(torch.isnan(torch.flatten(embeddings)) == False):
                print("NaN is in embeddings")
                exit()
            if self.loss_name == 'Arcface':
                features = self.arcface(embeddings, targets)
                loss = self.loss_fn(features, targets)
                acc = self.acc(targets, torch.argmax(features, dim=1))
            elif self.loss_name == 'Triplet':
                loss = self.loss_fn(embeddings, targets)
                acc = torch.Tensor([-1])  

            metrics_dict['loss'] = loss
            metrics_dict['accuracy'] = acc

        elif isinstance(self.model, MoCo_ResNet):
            #print("epoch", self.current_epoch)
            #print("batch", batch[0][0].size())
            _, q_batch, _ = batch[0]
            _, k_batch, _ = batch[1]
            q1, q2, k1, k2 = self((q_batch, k_batch, 0.99))
            loss = self.loss_fn(q1, k2) + self.loss_fn(q2, k1)
            #print("loss ", loss)

            metrics_dict['loss'] = loss

        if self.global_step % 500  == 0:
            pass
            #    self.trainer.save_checkpoint(f'./models/checkpoints/inception_checkpoint_globalstep={self.global_step}_ownarcface_test.pth')
            #    torch.save(self.model.state_dict(), f'./models/checkpoints/inception_model_globalstep={self.global_step}_ownarcface_test.pth')
            #    torch.save(self.loss_fn.state_dict(), f'./models/checkpoints/inception_arcface_globalstep={self.global_step}_ownarcface_test.pth')

        return metrics_dict

    def print_nan_gradients(self, model) -> None:
        """Iterates over model parameters 
            and prints out parameter + gradient information if NaN.
        """
        for param in model.parameters():
            print(f"{param}, {param.grad}")

    def training_step_end(self, metrics_dict):
        for k in metrics_dict.keys():
            self.log(f'ptl/train/batch_{k}', metrics_dict[k])
        return metrics_dict

    def training_epoch_end(self, list_metrics_dict):
        cum_metrics_dict = {metric: [] for metric in list_metrics_dict[0].keys()}
        for step_dict in list_metrics_dict:
            for metric in step_dict.keys():
                cum_metrics_dict[metric].append(step_dict[metric])

        for metric in cum_metrics_dict.keys():
            avg = torch.mean(torch.stack(cum_metrics_dict[metric]))
            self.log(f'ptl/train/epoch_{metric}', avg)

    def validation_step(self, batch, batch_idx):
        metrics_dict = {}
        if isinstance(self.model, AE) or isinstance(self.model, VAE):
            _, inputs, _ = batch
            embeddings = self(inputs)
            recon_x, x, mu, logvar = embeddings
            loss = self.loss_fn(x, recon_x)
            if isinstance(self.model, VAE):
                kld = (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
                loss = loss + kld
                metrics_dict['KLD'] = kld

            metrics_dict['loss'] = loss
        elif isinstance(self.model, InceptionV3_299):
            _, inputs, targets = batch
            embeddings = self(inputs)
            if self.loss_name == 'Arcface':
                features = self.arcface(embeddings, targets)
                loss = self.loss_fn(features, targets)
                acc = self.acc(targets, torch.argmax(features, dim=1))
            elif self.loss_name == 'Triplet':
                loss = self.loss_fn(embeddings, targets)
                acc = torch.Tensor([-1])         

            metrics_dict['loss'] = loss
            metrics_dict['accuracy'] = acc

        elif isinstance(self.model, MoCo_ResNet):
            _, q_batch, _ = batch[0]
            _, k_batch, _ = batch[1]
            q1, q2, k1, k2 = self((q_batch, k_batch, 0.99))
            loss = self.loss_fn(q1, k2) + self.loss_fn(q2, k1)

            metrics_dict['loss'] = loss

        return metrics_dict

    def validation_step_end(self, metrics_dict):
        for k in metrics_dict.keys():
            self.log(f'ptl/val/batch_{k}', metrics_dict[k])
        return metrics_dict

    def validation_epoch_end(self, list_metrics_dict):
        cum_metrics_dict = {metric: [] for metric in list_metrics_dict[0].keys()}
        for step_dict in list_metrics_dict:
            for metric in step_dict.keys():
                cum_metrics_dict[metric].append(step_dict[metric])

        for metric in cum_metrics_dict.keys():
            avg = torch.mean(torch.stack(cum_metrics_dict[metric]))
            self.log(f'ptl/val/epoch_{metric}', avg)

    def test_step(self, batch, batch_idx):
        metrics_dict = {}
        if isinstance(self.model, AE) or isinstance(self.model, VAE):
            _, inputs, _ = batch
            embeddings = self(inputs)
            recon_x, x, mu, logvar = embeddings
            loss = self.loss_fn(x, recon_x)
            if isinstance(self.model, VAE):
                kld = (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
                loss = loss + kld
                metrics_dict['KLD'] = kld

            metrics_dict['loss'] = loss
        elif isinstance(self.model, InceptionV3_299):
            _, inputs, targets = batch
            embeddings = self(inputs)
            if self.loss_name == 'Arcface':
                features = self.arcface(embeddings, targets)
                loss = self.loss_fn(features, targets)
                acc = self.acc(targets, torch.argmax(features, dim=1))
            elif self.loss_name == 'Triplet':
                loss = self.loss_fn(embeddings, targets)
                acc = torch.Tensor([-1])         

            metrics_dict['loss'] = loss
            metrics_dict['accuracy'] = acc

        elif isinstance(self.model, MoCo_ResNet):
            _, q_batch, _ = batch[0]
            _, k_batch, _ = batch[1]
            q1, q2, k1, k2 = self((q_batch, k_batch, 0.99))
            loss = self.loss_fn(q1, k2) + self.loss_fn(q2, k1)

            metrics_dict['loss'] = loss

        return metrics_dict

    def test_step_end(self, metrics_dict):
        for k in metrics_dict.keys():
            self.log(f'ptl/test/batch_{k}', metrics_dict[k])
        return metrics_dict

    def test_epoch_end(self, list_metrics_dict):
        cum_metrics_dict = {metric: [] for metric in list_metrics_dict[0].keys()}
        for step_dict in list_metrics_dict:
            for metric in step_dict.keys():
                cum_metrics_dict[metric].append(step_dict[metric])

        for metric in cum_metrics_dict.keys():
            avg = torch.mean(torch.stack(cum_metrics_dict[metric]))
            self.log(f'ptl/test/epoch_{metric}', avg)

    def batch_inference(self, inputs):
        self.model.to(device)
        self.model.eval()
        if isinstance(self.model, AE) or isinstance(self.model, VAE):
            embeddings = self.model(inputs)
            recon_x, _, mu, _ = embeddings
            return mu, recon_x
        elif isinstance(self.model, InceptionV3_299):
            embeddings = self(inputs)
            return embeddings

        elif isinstance(self.model, MoCo_ResNet):
            q1, _, _, _ = self((inputs, inputs, 0.99))
            return q1
  
    def configure_optimizers(self):
        sched = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                                    self.optimizers[0], 
                                    patience=3, 
                                    factor=0.1)}
        sched['monitor'] = 'ptl/train/epoch_loss'
        
        return self.optimizers, sched


    def set_opt_and_loss_func_(self):
        if isinstance(self.model, AE) or isinstance(self.model, VAE):
            params = self.model.parameters()
            self.optimizers = [torch.optim.Adam(params, lr=1e-3)]
            self.loss_fn = CustomLoss(self.loss_name)

        elif isinstance(self.model, InceptionV3_299):
            if self.loss_name == 'Arcface':
                self.arcface = Arcface(self.model.latent_size, self.num_classes)
                #self.arcface = AddMarginProduct(self.model.latent_size, self.num_classes, device)
                params = chain(self.model.parameters(), self.arcface.parameters())
            elif self.loss_name == 'Triplet':
                params = self.model.parameters()

            self.loss_fn = CustomLoss(self.loss_name)
            self.optimizers = [torch.optim.Adam(params, lr=1e-4)]
            self.acc = Accuracy()

        elif isinstance(self.model, MoCo_ResNet):
            self.optimizers = [torch.optim.AdamW(self.model.parameters(), 1e-3)]
            self.loss_fn = CustomLoss(self.loss_name)