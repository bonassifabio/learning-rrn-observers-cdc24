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

import pytorch_lightning as pl
import torch

from s5.subspace import SubspaceEncodedRNN


def online_forecasting(full: SubspaceEncodedRNN, U: torch.Tensor, Y: torch.Tensor, context_window: int, prediction_horizon: int):
    """Get the average/expected forecasting error over the prediction horizon.

    Args:
        full (SubspaceEncodedRNN): The full model (observer + model).
        U (torch.Tensor): The input data (shape: (T, in_features)).
        Y (torch.Tensor): The output data (shape: (T, out_features)).
        context_window (int): The context window.
        prediction_horizon (int): The prediction horizon.

    Returns:
        torch.Tensor: The average/expected forecasting error (shape: (T_tilde, prediction_horizon, out_features)).
    """
    # Turn the network into evaluation mode
    full.eval()
    observer = full.obsv
    model = full.model

    full.input_scaler.bias = full.input_scaler.bias.to(dtype=torch.float32)
    full.input_scaler.scale = full.input_scaler.scale.to(dtype=torch.float32)
    full.output_scaler.bias = full.output_scaler.bias.to(dtype=torch.float32)
    full.output_scaler.scale = full.output_scaler.scale.to(dtype=torch.float32)

    Un = full.input_scaler.normalize(U)
    Yn = full.output_scaler.normalize(Y)

    error_over_sim = []
    x0_ = None
    for t in range(context_window, Un.shape[0] - prediction_horizon):
        u_past = Un[t-context_window:t, :].unsqueeze(0)
        u_future = Un[t:t+prediction_horizon, :].unsqueeze(0)
        y_past = Yn[t-context_window:t, :].unsqueeze(0)
        y_future = Yn[t:t+prediction_horizon, :].unsqueeze(0)
        context_data = torch.cat((u_past, y_past), dim=-1)

        # Get the observer output
        if observer is not None:
            x0_ = observer(context_data)

        if x0_ is not None:
            if x0_.ndim == 2:
                x0_ = x0_.unsqueeze(1)
            elif x0_.ndim == 3 and x0_.shape[1] > 1:
                x0_ = x0_[:, -1, :].unsqueeze(1)

        # Get the model output
        y_, x_ = model.simulate(u_future, x0_, return_state_seq=True)
        x0_ = x_[:, 0, :].unsqueeze(1)

        # Get the error
        error = torch.abs(y_ - y_future).squeeze(0)  # (shape: (prediction_horizon, out_features))
        error = full.output_scaler.scale * error
        error_over_sim.append(error)
    
    return torch.stack(error_over_sim, dim=0)  # (shape: (T_tilde, prediction_horizon, out_features))