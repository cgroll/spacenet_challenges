import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp


from src.config_vars import ConfigVariables

    
class UNetFineTune(pl.LightningModule):
    def __init__(self, lr=0.00008, optimizer='Adam', freeze_resnet=True):
        super(UNetFineTune, self).__init__()

        # from https://www.kaggle.com/code/alexj21/pytorch-eda-unet-from-scratch-finetuning
        model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
        if freeze_resnet == True:
            for name, p in model.named_parameters():
                if "encoder" in name:
                    p.requires_grad = False

        self.model = model
        self.lr = lr
        self.optimizer = optimizer

        IMAGE_PIXEL_SIZE = ConfigVariables.unet_3band_img_pixel_size

        self.example_input_array = torch.rand(1, 3, IMAGE_PIXEL_SIZE, IMAGE_PIXEL_SIZE)

    # def on_train_start(self):
    #     self.logger.log_hyperparams(self.hparams, {'max_epochs': self.trainer.max_epochs, 'hp_metric': -3})

    def forward(self, x):
        return self.model(x)
    
    def BCELoss(self, logits, labels):
        return nn.BCEWithLogitsLoss(logits, labels)
    
    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam([dict(params=self.parameters(), lr=self.lr),])
        else:
            raise ValueError(f'Optimizer {self.optimizer} is not implemented yet')
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        data = train_batch['image']
        target = train_batch['labels']

        logits = self.forward(data)
        loss = F.binary_cross_entropy_with_logits(
            input=logits, target=target
        )
        
        #self.log('train_loss', loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        data = val_batch['image']
        target = val_batch['labels']

        logits = self.forward(data)
        loss = F.binary_cross_entropy_with_logits(
            input=logits, target=target
        )

        # compute metrics
        threshold = 0.5
        normed_logits = torch.sigmoid(logits).cpu().numpy()
        true_labels = target.cpu().numpy()

        normed_logits[normed_logits >= threshold] = 1
        normed_logits[normed_logits < threshold] = 0

        n_correct = (normed_logits == true_labels).sum()
        n_entries = np.prod(normed_logits.shape)
        accuracy = n_correct / n_entries

        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        self.log('hp_metric', loss)
        # self.logger.log_metrics({'val_loss': loss})



if __name__ == '__main__':
    
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from src.data.unet_data_loader import UNetDataModule
    from pathlib import PureWindowsPath
    
    MAX_EPOCHS = 20 # 20
    OPTIMIZER = 'Adam'
    LOSS_FUNC = ''

    data_module = UNetDataModule()

    for this_lr in [0.0002]: #[0.00008, 0.0001]: # [0.00008, 0.0001, 0.00015]: #, 0.0002

        logger = TensorBoardLogger('tb_logs_3', name='UNet Ptl V1', log_graph=True)
        logger.log_hyperparams({"max_epochs": MAX_EPOCHS, "optimizer": OPTIMIZER, 'lr': this_lr})

        model = UNetFineTune(lr=this_lr, optimizer=OPTIMIZER)

        chkpt_path = str(PureWindowsPath("C:\\Users\\cgrol\\OneDrive\\Dokumente\\GitHub\\spacenet_challenges\\tb_logs_2\\UNet-Fine-Tuned\\version_6\\checkpoints\\epoch=19-step=16720.ckpt"))
        model = UNetFineTune.load_from_checkpoint(chkpt_path)
        for name, p in model.named_parameters():
            if "encoder" in name:
                p.requires_grad = True

        #logger.log_graph(model, input_array=)
        # trainer = pl.Trainer(max_epochs=2, precision=16, accelerator="gpu", logger=logger)
        # trainer = pl.Trainer(precision=16, accelerator="gpu", logger=logger)
        # trainer = pl.Trainer(max_epochs=MAX_EPOCHS, precision=16, accelerator="gpu", 
        #                      logger=logger, limit_train_batches=10, log_every_n_steps=2, 
        #                      callbacks=[checkpoint_callback_best, checkpoint_callback_default])

        checkpoint_callback_best = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            #dirpath='model_checkpoints/unet',
            filename='best_model-unet-{epoch:02d}-{val_loss:.2f}') # -lr_{lr:2.8f}

        checkpoint_callback_default = ModelCheckpoint()

        trainer = pl.Trainer(max_epochs=MAX_EPOCHS,
                             logger=logger,
                             callbacks=[checkpoint_callback_best, checkpoint_callback_default, 
                                        EarlyStopping(monitor="val_loss", mode="min", patience=5)
                                        ])
        # trainer = pl.Trainer(max_epochs=2, precision=16, accelerator="gpu", fast_dev_run=True)

        trainer.fit(model, data_module)
        # trainer.fit(model, ckpt_path=chkpt_path)

        logger.finalize("success")
        print(checkpoint_callback_best.best_model_path)



    # TODO:
    # - log images
    # - log into different subfolders (ADAM vs SDG)
    # - resume training -> There was a problem with data_loader; but loading model weights works
    # - fix hp_metric -> Done
    # - try finding optimal learning rate
    # - try mixed precision -> didn't reduce memory size for me. Hence, could not further increase batch size
    # - try different batch sizes
    
    # Other models, data usage:
    # - fine-tuned model
    # - include multi-spectral data
    # - include other data like open maps?