import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
np.random.seed(0)
import joblib
import argparse
import yaml
import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms

from pytorch_lightning import Trainer, seed_everything
seed_everything(42, workers=True)
from pytorch_lightning.loggers import WandbLogger
os.environ["WANDB_SILENT"] = "true"
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.embeddings.BiomarkerEncoder import BiomarkerEncoder

from models.embeddings.AE import AE
from models.embeddings.VAE import VAE
from models.embeddings.inceptionv3 import InceptionV3_224, InceptionV3_299
from models.embeddings.moco import MoCo_ResNet

from data.FundusDataset import FundusDataset
from data.Sampler import TripletSampler, MocoSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataloader(dataset, batch_size, sampler=None):
    loader = DataLoader(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=(sampler is None),
            sampler=sampler,
            pin_memory=False, 
            num_workers=1,
            drop_last=True)
    return loader
    

def run(config):
    """Pipeline controller

    Args:
        config (args): argument passed as --config when launching the script
    """

    with open(config) as infile:
        config_dict = yaml.load(infile, Loader=yaml.SafeLoader)

    batch_size = config_dict['batch_size']
    max_epochs = config_dict['max_epochs']
    model_name = config_dict['model_name']
    lbls_flag = config_dict['labels_flag']
    emb_size = config_dict['embedding_size']
    print_model_summary = config_dict['print_model_summary']
    experiment_name = config_dict['experiment_name']
    loss_name = config_dict['loss_name']
    dataset_size = config_dict['dataset_size']
    res = config_dict['resolution']
    aug = config_dict['augmented']
    
    norm = (model_name in ['InceptionV3', 'MoCo'])

    train_set = FundusDataset(
                            set='train', 
                            labels=lbls_flag, 
                            size=dataset_size, 
                            normalized=norm, 
                            resolution=res, 
                            augmented=aug)
    inference_set = FundusDataset(
                            set='inference', 
                            labels=lbls_flag, 
                            size=dataset_size, 
                            normalized=norm, 
                            resolution=res, 
                            augmented=aug)
    
    """
    if model_name == 'AE' or model_name == 'VAE':
        items = [train_set.__getitem__(i) for i in range(8)]
        for idx, item in enumerate(items):
            id, item, target = item
            
            if idx < 8:
                filename = train_set.get_part_from_idx(idx)
                img = item.numpy()
                img_plt = img.transpose(1, 2, 0)
                plt.imsave(f'./data/recons/{model_name}_orig/{filename}.png',
                            img_plt)
    """
    if loss_name == 'Triplet':
        train_sampler = TripletSampler(train_set, batch_size)
        train_loader = get_dataloader(train_set, batch_size, train_sampler)
        inference_sampler = TripletSampler(inference_set, batch_size)
        inference_loader = get_dataloader(inference_set, batch_size, 
                                            inference_sampler)
    elif loss_name == 'Contrastive':
        train_sampler = MocoSampler(train_set, batch_size)
        train_loader = get_dataloader(train_set, batch_size, train_sampler)
        inference_sampler = MocoSampler(inference_set, batch_size)
        inference_loader = get_dataloader(inference_set, batch_size, 
                                            inference_sampler)
    else:
        train_loader = get_dataloader(train_set, batch_size)
        inference_loader = get_dataloader(inference_set, batch_size)

    if model_name == 'AE' or model_name == 'VAE':
        if model_name == 'AE':
            model = AE(emb_size)
        elif model_name == 'VAE':
            model = VAE(emb_size)
        if print_model_summary:
            print(summary(model.to(device), (3, res, res)))
        ckpt_monitor_metric = 'ptl/val/epoch_loss'
        num_classes = None
        #early_stopping = EarlyStopping(monitor='val_loss', mode='min')
        
    elif model_name == 'InceptionV3':
        model = InceptionV3_299(emb_size)
        if print_model_summary:
            print(summary(model.to(device), (3, res, res)))
        ckpt_monitor_metric = 'ptl/train/epoch_loss'
        num_classes = train_set.classes_count()

    elif model_name == 'MoCo':
        from functools import partial
        import torchvision.models as torchvision_models
        model = MoCo_ResNet(partial(torchvision_models.__dict__['resnet50'], 
                            zero_init_residual=True))
        if print_model_summary:
            print(summary(model.to(device), (3, res, res)))
        ckpt_monitor_metric = 'ptl/train/epoch_loss'
        num_classes = None

    ckpt_callback = ModelCheckpoint(
                            dirpath=f'./models/checkpoints/{model_name}',
                            save_top_k=3, 
                            monitor=ckpt_monitor_metric)
    biomarker_encoder = BiomarkerEncoder(model, loss_name, num_classes)
    wandb_logger = WandbLogger(project='fundus_biomarkers', 
                                name=model_name+'-'+loss_name) 
                                #experiment=experiment_name)
    wandb_logger.watch(model)
    trainer = Trainer(
                    accelerator='gpu', devices=-1,
                    max_epochs=max_epochs, 
                    logger=wandb_logger, 
                    callbacks=[ckpt_callback],
                    gradient_clip_val=0.5,
                    #check_val_every_n_epoch=1,)
                    #log_every_n_steps=1)
    )
    
    if model_name == 'AE' or model_name == 'VAE':
        trainer.fit(biomarker_encoder, train_loader)
    elif model_name == 'InceptionV3':
        trainer.fit(biomarker_encoder, train_loader)
    elif model_name == 'MoCo':
        trainer.fit(biomarker_encoder, train_loader)

    print('Best model path: ', ckpt_callback.best_model_path)

    print("Running inference...")
    dir_suffix = ('dataaug' if aug else 'basic')
    embs_dir = f'./data/embeddings/{model_name}_{loss_name}_{dir_suffix}_{emb_size}_{res}/'
    recon_dir = f'./data/recons/{model_name}_{loss_name}_{dir_suffix}_{emb_size}_{res}/' 

    if not os.path.exists(embs_dir):
        os.makedirs(embs_dir)
    if not os.path.exists(recon_dir):
        os.makedirs(recon_dir)

    for batch in inference_loader:
        if model_name == 'AE' or model_name == 'VAE':
            ids, inputs, _ = batch
            output, recs = biomarker_encoder.batch_inference(inputs.to(device))
            if len(os.listdir(recon_dir)) < 8:
                for id, rec in enumerate(recs):
                    img = rec.cpu().detach().numpy().transpose(1,2,0)
                    filename = inference_set.get_part_from_idx(ids[id])
                    plt.imsave(f'{recon_dir}recon_{filename}.png', 
                                img)
        elif model_name == 'InceptionV3':
            ids, inputs, _ = batch
            output = biomarker_encoder.batch_inference(inputs.to(device))
        elif model_name == 'MoCo':
            ids, inputs, _ = batch[0]
            output = biomarker_encoder.batch_inference(inputs.to(device))

        for id, emb in enumerate(output):
            joblib.dump(
                emb.cpu().detach().numpy(), 
                embs_dir+inference_set.get_part_from_idx(ids[id])+'.emb', 
                compress=5)
        
    print("Inference done.")
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fundus Scans Biomarkers Pipeline')
    parser.add_argument("--config", help="Provide path to configuration file")
    args = parser.parse_args()

    run(args.config)
