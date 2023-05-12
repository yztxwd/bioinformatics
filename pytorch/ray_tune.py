#!/usr/bin/env python
# coding: utf-8

import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

import ray
from ray import tune, air
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch

# start a fresh new ray runtime
ray.shutdown()
ray.init()
#ray.init(runtime_env={"env_vars": {"PL_DISABLE_FORK": "1"}})   # i think it fixed something, forget what exactly that was
ray.available_resources()

config = {
    'conv1d_filter': 32,
    'dense_aug_feature': 32,
    'num_conv': 2,
    'num_dense': 2,
    'lr': 1e-4,
    'batch_size':  64
}

def train_tune(config, num_epochs=50, num_gpus=1, chroms_channel=12, seqonly=False):
    print(f"current WORKING dir is {os.getcwd()}")
    print(f"trial directory is {session.get_trial_dir()}")
    model = ConvRepeatVGGLike_Ray(**config, chroms_channel, seqonly)
    trainer = pl.Trainer(
        max_epochs = num_epochs,
        accelerator = "gpu",
        devices = num_gpus,
        auto_select_gpus = False,
        logger=TensorBoardLogger(
            save_dir=os.getcwd(), name="", version=".", log_graph=True),
        enable_progress_bar=False,
        callbacks = [
           ModelCheckpoint(filename="checkpoint_{epoch}-{val_loss:.6f}", 
                                                monitor='val_loss', 
                                                save_last=True, 
                                                save_top_k=1, 
                                                mode='min', 
                                                every_n_epochs=1),
            ModelSummary(max_depth=-1),
            TuneReportCallback(
                {
                    "val_loss": "val_loss",
                },
                on="validation_end")
        ]
    )
    trainer.fit(model, datamodule=SeqChromDataModule(data_config='/path/to/dataconfig', pred_bed=None,
    num_workers = 8, batch_size=config['batch_size']))

def tune_train_asha(num_samples=10, num_epochs=50, gpus_per_trial=1):
    config = {
        'conv1d_filter': tune.choice([32, 64]),
        'conv1d_filter_width': tune.choice([3, 5, 7, 9]),
        'dense_aug_feature': tune.choice([64, 128]),
        'num_dense': tune.choice([3, 4]),
        'lr': tune.loguniform(1e-4, 1e-3),
        'gamma': tune.choice([0.9, 1.0]),
        'batch_size':  tune.choice([256])
    }

    reporter = CLIReporter(
        parameter_columns = ['conv1d_filter', 'dense_aug_feature', 'num_dense', 'lr', 'batch_size'],
        metric_columns = ['loss', 'training_iteration']
    )

    train_fn_with_parameters = tune.with_parameters(
        train_tune,
        num_epochs = 50,
        num_gpus = gpus_per_trial,
        chroms_channel = 12,
        seqonly = False
    )
    resources_per_trial = {"cpu": 10, "gpu": gpus_per_trial}
    
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=10,
        reduction_factor=2
    )
    bayesopt = BayesOptSearch()
    
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config = tune.TuneConfig(
            metric='val_loss',
            mode='min',
            scheduler=scheduler,
            search_alg=bayesopt,
            num_samples=num_samples
        ),
        run_config=air.RunConfig(
            name='FoxC1_tune_train_asha_VGGLike_balanced_val_v1',
            progress_reporter=reporter,
            callbacks=[tune.logger.CVSLoggerCallback()],
            local_dir="path/to/dir"
        ),
        param_space=config
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)

    return results

tune_train_asha(num_samples=50, num_epochs=100, gpus_per_trial=1)

