

if __name__ == '__main__':
    
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from src.data.unet_data_loader import UNetDistTransfSamDataModule
    from pathlib import PureWindowsPath
    from src.models.unet_ptl import UNet
    import pytorch_lightning as pl
    import numpy as np

    
    MAX_EPOCHS = 50
    OPTIMIZER = 'Adam'
    LOSS_FUNC = ''

    data_module = UNetDistTransfSamDataModule() # inputs: satellite images + distance transformed labels, and also SAM distance transforms

    # Run learning rate finder
    for this_lr in [0.00008, 0.0001, 0.00015]:

        logger = TensorBoardLogger('tb_logs_2', name='UNet-DistTransf-V1', log_graph=True)
        logger.log_hyperparams({"max_epochs": MAX_EPOCHS, "optimizer": OPTIMIZER, 'lr': this_lr, 'input_channels': 4})

        model = UNet(input_channels=4, optimizer=OPTIMIZER, lr=this_lr)

        checkpoint_callback_best = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            filename='best_model-unet-dist-transf-{epoch:02d}-{val_loss:.2f}') # -lr_{lr:2.8f}

        checkpoint_callback_default = ModelCheckpoint()

        trainer = pl.Trainer(max_epochs=MAX_EPOCHS, # precision='16-mixed', # accelerator="gpu", 
                            logger=logger,
                            callbacks=[checkpoint_callback_best, checkpoint_callback_default, 
                                    EarlyStopping(monitor="val_loss", mode="min", patience=5)
                                    ])
        # Fit model
        trainer.fit(model, data_module)

        # trainer = pl.Trainer(max_epochs=2, precision=16, accelerator="gpu", fast_dev_run=True)

        logger.finalize("success")
        print(checkpoint_callback_best.best_model_path)
