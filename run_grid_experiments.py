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

import itertools
import multiprocessing as mp
from datetime import datetime

import numpy as np

from s5.initializations import HippoDiagonalizedInitializer
from s5.nets import FFNNObserver, S5Model, SingleLayerGRU, SingleLayerLSTM
from s5.util import match_metric_or_loss
from training_helper import TBPTTParams, TrainingParams, run_training_helper

# Number of parallel instances of the training_helper to run at the same time
NUM_PAR_INST = 7
N_REP = 3

#### HYPERPARAMETERS GRID SEARCH ####
context_window_exps = [10, 25, 50, 100, 150] * N_REP
Ts_exps = [100]
batch_size_exps = [25, 50]
observer_exps = ['S5Model-small', 'SingleLayerLSTM', 'SingleLayerGRU', 'FFNNObserver', 'None']


combinations = list(itertools.product(Ts_exps, batch_size_exps, context_window_exps, observer_exps))
idxs = np.random.permutation(np.arange(len(combinations)))
idx_split = np.array_split(idxs, NUM_PAR_INST)

### MODEL FIXED HYPERPARAMETERS ###
gamma_final = 0.6
patience = 70

obsv_N_S5 = 16
obsv_H = 4
obsv_init = HippoDiagonalizedInitializer(scale=0.1)
model_N = 8

obsv_params_s5 = {'H': obsv_H,
                  'in_features': 2,
                  'out_features': model_N,
                  'initialization': obsv_init,
                  'default_Δ': 10.0,
                  'return_next_y': False,
                  'skip_connection': True,
                  'skip_identity': True,
                  'learnable_Λ': True,
                  'learnable_eig_scale': True,
                  'learnable_eig_rot': False, 
                  'use_parallel_scan': False,
                  'batch_norm': 'post-noaffine',
                  'return_sequence': False,
                  'initial_eig_scale': 0.55 }

obsv_params_gru = {'hidden_states': 2*model_N,
                     'in_features': 2,
                     'out_features': model_N }

obsv_params_lstm = {'hidden_states': 2*model_N,
                    'in_features': 2,
                    'out_features': model_N }

obsv_params_ffnn = {'in_features': 2,
                    'out_features': model_N }
                                        
model_params = {'hidden_states': model_N,
                'in_features': 1,
                'out_features': 1 }


# Create NUM_PAR_INST instances of the training_helper and run them in parallel
def run_batch_experiments(exps: np.ndarray, id: int):
    output_dir = f'training_output_{datetime.now().strftime("%y%m%d_%H%M")}_w{id}'

    for i, exp_id in enumerate(exps):
        ts, bs, cw, obsv_class = combinations[exp_id]

        if cw == 0:
            observer = None
        else:
            match obsv_class:
                case 'S5Model':
                    observer = S5Model(N=[obsv_N_S5, obsv_N_S5],
                                       **obsv_params_s5)
                case 'S5Model-small':
                    observer = S5Model(N=[obsv_N_S5],
                                       **obsv_params_s5)
                case 'S5Model-Large':
                    observer = S5Model(N=[obsv_N_S5 * 2, obsv_N_S5 * 2],
                                       **obsv_params_s5)
                case 'SingleLayerGRU':
                    observer = SingleLayerGRU(**obsv_params_gru)
                case 'SingleLayerLSTM':
                    observer = SingleLayerLSTM(**obsv_params_lstm)
                case 'FFNNObserver':
                    observer = FFNNObserver(**obsv_params_ffnn, context_window=cw)
                case 'None':
                    observer = None
    
        gamma = np.power(gamma_final, 1.0 / ts)
        tr_loss = match_metric_or_loss('wmse', gamma=gamma, washout=10 if observer is None else 0)
        val_metric = match_metric_or_loss('mse', washout=10 if observer is None else 0)
        test_metric = {'mse': match_metric_or_loss('mse'), 'fit': match_metric_or_loss('fit')}

        training_params = TrainingParams(batch_size=bs, 
                                         lr=4e-4, 
                                         early_stopping_patience=patience, 
                                         training_loss=tr_loss, 
                                         validation_metric=val_metric, 
                                         test_metrics=test_metric)
        
        ns = int(2000 * 75 // ts)
        tbptt_params = TBPTTParams(Ts_train=ts, Ns_train=ns, Ts_val=-1, Ns_val=-1, context_window=cw)

        model = SingleLayerGRU(**model_params)

        print(f'[{id}/{NUM_PAR_INST}] Running experiment {i+1} of {len(exps)}...')

        run_training_helper(tbptt_params, 
                            training_params, 
                            observer, 
                            model,
                            datset_name='ph-10s',
                            show_progress=False, 
                            parent_dir=output_dir)


if __name__ == '__main__':
    processes = []
    for i, exps in enumerate(idx_split):
        p = mp.Process(target=run_batch_experiments, args=(exps, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    print('Training completed!')
    
