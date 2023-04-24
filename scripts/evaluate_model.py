# %%
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm
import pandas as pd
import click

from src.data.band3_binary_mask_data import Band3BinaryMaskDataset, RandomCropImgAndLabels, ToTensorImgAndLabels, ToTensorImgAndLabelsExtension
from src.path import ProjPaths
from src.metrics import sample_logits_and_labels, logits_to_prediction, classification_cases, prediction_metrics

MODEL_CHOICES = ['UNet', 'UNetFineTuned']
# MODEL_CHOICES = ['UNet_pytorch', 'UNet', 'UNetFineTuned']

@click.command()
@click.option('--output_fname', type=click.Path(), 
              prompt='Please provide the filename where outputs should be stored')
@click.option('--model_name', type=click.Choice(choices=MODEL_CHOICES), 
              default='base', prompt='Which model should be used?',
              help='Defines the model to be used. Model names need \
                to be resolved in the code of the script')
def compute_metrics(model_name, output_fname):

    outpath = ProjPaths.metrics_path / output_fname

    if model_name == 'UNet':

        from src.models.unet_ptl import UNet
        chkpt_path = ProjPaths.model_path / 'unet' / 'unet_ptl_v5' / 'checkpoints' / 'best_model-unet-epoch=15-val_loss=0.09.ckpt'
        model = UNet.load_from_checkpoint(chkpt_path)

    elif model_name == 'UNetFineTuned':

        from src.models.unet_finetune import UNetFineTune
        chkpt_path = ProjPaths.model_path / 'unet' / 'unet_finetuned_v8' / 'checkpoints' / 'best_model-unet-epoch=03-val_loss=0.09.ckpt'
        model = UNetFineTune.load_from_checkpoint(chkpt_path)

    else: 
        raise ValueError(f'Model not implemented yet: {model_name}. Allowed options: {MODEL_CHOICES}')

    model.eval()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)
    # %% create data loader
    
    test_path = ProjPaths.interim_sn1_data_path / "test"
    test_dataset = Band3BinaryMaskDataset(test_path, transform=transforms.Compose([
                                               RandomCropImgAndLabels(384),
                                               ToTensorImgAndLabels()
                                           ]))
    
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=0)

    # %%
    
    all_metrics_list = []

    for this_sample_id in tqdm(range(0, len(test_dataset))):

        sample = test_dataset[this_sample_id]
        labels, logits = sample_logits_and_labels(sample, model, DEVICE)
        pred = logits_to_prediction(logits, pred_threshold=0.5)
        true_pos, true_neg, false_pos, false_neg = classification_cases(labels, pred)
        metrics_df = prediction_metrics(true_pos, true_neg, false_pos, false_neg)
        metrics_df.index = [this_sample_id]

        all_metrics_list.append(metrics_df)

    all_metrics = pd.concat(all_metrics_list)

    # fix NaN metrics for images without buildings
    xx_inds_zero_build = (all_metrics['true_pos'] == 0) & (all_metrics['false_pos'] == 0)
    all_metrics.loc[xx_inds_zero_build, 'jaccard']  = 1
    all_metrics.loc[xx_inds_zero_build, 'dice']  = 1
    all_metrics = all_metrics.reset_index()
    all_metrics.rename({'index': 'sample_id'}, axis=1, inplace=True)

    # %% save metrics to disk

    all_metrics.to_csv(outpath, index=False)


if __name__ == '__main__':

    compute_metrics()

    
    

        
    