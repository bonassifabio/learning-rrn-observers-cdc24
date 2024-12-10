'''
Copyright (C) 2024 Fabio Bonassi, Carl Andersson, and co-authors

This file is part of learning-rrn-observers-cdc24.

learning-rrn-observers-cdc24 is free software: you can redistribute it 
and/or modify it under the terms of the GNU General Public License as 
published by the Free Software Foundation, either version 3 of the License, 
or (at your option) any later version.

learning-rrn-observers-cdc24  is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU General Public License
along with learning-rrn-observers-cdc24.  
If not, see <http://www.gnu.org/licenses/>.
'''

import json
import logging
import os
import warnings
from datetime import datetime
from typing import Callable, NamedTuple, Optional

import numpy as np
import pytorch_lightning as pl
import scipy.io
import torch
from pytorch_lightning.utilities.model_summary.model_summary import summarize
from torch.optim import Adam
from torch.utils.data import DataLoader

from s5.callbacks import PlotValidationTrajectories
from s5.ssnet.data import MinMaxSequenceScaler, SequenceScaler, tbptt
from s5.subspace import SubspaceEncodedRNN

_InputOutputPairs = tuple[list[torch.Tensor], list[torch.Tensor]]
DatasetOfSequences = tuple[_InputOutputPairs, _InputOutputPairs, _InputOutputPairs, Optional[torch.Tensor], Optional[torch.Tensor]]
TBPTTParams = NamedTuple('TBPTTParams', [('Ts_train', int), ('Ns_train', int), ('Ts_val', int), ('Ns_val', int), ('context_window', int)])
TrainingParams = NamedTuple('TrainingParams', [('batch_size', int), ('lr', float), ('early_stopping_patience', int), ('training_loss', Callable), ('validation_metric', Callable), ('test_metrics', dict[str, Callable] | Callable)])

def _sanitize_callable(d: dict):
    for k, v in d.items():
        if isinstance(v, torch.nn.Module):
            d[k] = v.__class__.__name__
        elif callable(v):
            d[k] = v.__name__
        elif isinstance(v, dict):
            d[k] = _sanitize_callable(v)
    return d

def get_data_sequences(dataset: str):
    if dataset.lower().startswith('ph'):
        dataset_file = 'Datasets/PH/Dataset_6000.mat' if dataset.lower().endswith('10s') else 'Datasets/PH/Dataset_large.mat'
        dataset = scipy.io.loadmat(dataset_file)
        u_train_exp, y_train_exp = dataset['U_train'], dataset['Y_train']
        u_val_exp, y_val_exp = dataset['U_val'], dataset['Y_val']
        u_test_exp, y_test_exp = dataset['U_test'], dataset['Y_test']
        U_min, U_max = dataset['U_min'], dataset['U_max']

        return (u_train_exp, y_train_exp), (u_val_exp, y_val_exp), (u_test_exp, y_test_exp), U_min, U_max

    elif dataset.lower().startswith('toy'):
        dataset_file = 'Datasets/toy_long_context_input.npy' if dataset.lower().endswith('input') else 'Datasets/toy_long_context_lownoise.npy'
        data = np.load(dataset_file)
        ntr = int(data.shape[1] * 0.6) 
        nval = ntr + int(data.shape[1] * 0.2)
        nend = int(data.shape[1] * 1.0)
        u_train_exp, y_train_exp = data[0, :ntr].reshape(-1, 1), data[1, :ntr].reshape(-1, 1)
        u_val_exp, y_val_exp = data[0, ntr:nval].reshape(-1, 1), data[1, ntr:nval].reshape(-1, 1)
        u_test_exp, y_test_exp = data[0, nval:nend].reshape(-1, 1), data[1, nval:nend].reshape(-1, 1)
        U_mean = np.mean(data[0, :])
        U_std = np.std(data[0, :])
        U_min = U_mean - 2 * U_std
        U_max = U_mean + 2 * U_std

        return (u_train_exp, y_train_exp), (u_val_exp, y_val_exp), (u_test_exp, y_test_exp), U_min, U_max
        
def run_training_helper(tbptt_params: TBPTTParams, 
                        training_params: TrainingParams, 
                        observer: torch.nn.Module, 
                        model: torch.nn.Module,
                        datset_name: str = 'ph',
                        parent_dir: str = 'training_output',
                        accelerator: str = 'cpu',
                        max_epochs: int = 5000,
                        suppress_warnings: bool = True,
                        show_progress: bool = True,
                        name: str = None):
    
    if name is None:
        name = f'{observer.__class__.__name__ if observer is not None else "NoObsv"}_{model.__class__.__name__}_{datset_name}_Batch{training_params.batch_size}_Ts{tbptt_params.Ts_train}'
    
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    output_folder = f'{parent_dir}{"/" if not parent_dir.endswith("/") else ""}{name}_{datetime.now().strftime("%y%m%d_%H%M")}'

    training_sequences, validation_sequences, testing_sequences, U_min, U_max = get_data_sequences(datset_name)
    in_scaler = SequenceScaler(bias=(U_min + U_max) / 2, scale=(U_max - U_min) / 2)
    out_scaler = MinMaxSequenceScaler()

    dataset = tbptt(training=training_sequences,
                    validation=validation_sequences, 
                    testing=testing_sequences,
                    **tbptt_params._asdict(),
                    input_scaler=in_scaler,
                    output_scaler=out_scaler)
    
    train_loader = DataLoader(dataset.training, batch_size=training_params.batch_size, shuffle=True)
    val_loader = DataLoader(dataset.validation, batch_size=max(1, tbptt_params.Ns_val), shuffle=False)
    test_loader = DataLoader(dataset.testing, batch_size=1, shuffle=False)

    full = SubspaceEncodedRNN(observer, model)

    full.training_settings(train_loss=training_params.training_loss, 
                       val_metric=training_params.validation_metric, 
                       test_metric=training_params.test_metrics,
                       optimizer=Adam, 
                       optimizer_opts={'lr': training_params.lr})
    
    # Define the checkpoints
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='best-model', monitor='validation_metric', save_top_k=1, mode='min', save_last=True)
    timer = pl.callbacks.Timer(interval='epoch')
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='validation_metric', patience=training_params.early_stopping_patience, mode='min')
    test = PlotValidationTrajectories(test_dataloader=test_loader, hidden_states=False, every_n_epochs=20)
    callbacks = [early_stopping_callback, timer, checkpoint_callback, test]

    if suppress_warnings:
        warnings.filterwarnings("ignore", ".*does not have many workers.*")
        warnings.filterwarnings("ignore", "Missing logger folder.*")
        warnings.filterwarnings("ignore", ".*GPU available but not used.*")
        logging.getLogger('lightning').setLevel(logging.ERROR)
        logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
        logging.getLogger('pytorch.lightning').setLevel(logging.ERROR)

    if observer is not None:    
        num_obsv_parameters = sum(p.numel() for p in observer.parameters() if p.requires_grad)
    else:
        num_obsv_parameters = 0
    
    full.input_scaler = in_scaler
    full.output_scaler = out_scaler

    # # Trainer
    trainer = pl.Trainer(max_epochs=max_epochs, 
                         default_root_dir=output_folder, 
                         callbacks=callbacks, 
                         accelerator=accelerator, 
                         enable_progress_bar=show_progress,
                         log_every_n_steps=1)
    
    trainer.fit(full, train_dataloaders=train_loader, val_dataloaders=val_loader)

    full.eval()
    test_metrics = test.test_trajectories(trainer, full, final=True)
    seconds_per_epoch = timer.time_elapsed('train') / trainer.current_epoch
    
    torch.save(full.state_dict(), f'{output_folder}/trained_weights.pt')

    results = {}
    results['model_summary'] = str(summarize(full, max_depth=4))
    results['final_test_metrics'] = test_metrics
    results['seconds_per_epoch'] = seconds_per_epoch
    results['required_epochs'] = trainer.current_epoch
    results['restored_epochs'] = checkpoint_callback.best_model_score.cpu().numpy().item()
    results['training_params'] = _sanitize_callable(training_params._asdict())
    results['tbptt_params'] = _sanitize_callable(tbptt_params._asdict())
    results['dataset_name'] = datset_name
    results['model_params'] = _sanitize_callable(model.params_str)
    results['observer_params'] = _sanitize_callable(observer.params_str) if observer is not None else {}
    results['observer_num_trainable_params'] = num_obsv_parameters

    trainer.logger.experiment.add_text('Network Summary', results['model_summary'])

    # Export to a json file
    with open(f'{output_folder}/summary.json', 'w') as f:
        json.dump(results, f, indent=4)

    return checkpoint_callback.best_model_score.cpu().numpy().item()