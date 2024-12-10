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
from typing import Callable, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from s5.metrics import DecayingMSELoss, FITIndex
from s5.parametrizations import (linear_parametrization,
                                 log_parametrization_forward,
                                 log_parametrization_inverse)


def match_activation(activation: str) -> torch.nn.Module:
    """Retrieve the activation function given its string.

    Args:
        activation (str): The activation function. Can be 'relu', 'leaky_relu', 'gelu', 'tanh', 'sigmoid', 'glu', 'swish' or 'none'.

    Raises:
        NotImplementedError: If the activation function is not implemented.

    Returns:
        torch.nn.Module: The activation function.
    """
    match activation:
        case 'relu':
            return torch.nn.ReLU()
        case 'leaky_relu':
            return torch.nn.LeakyReLU()
        case 'gelu':
            return torch.nn.GELU()
        case 'tanh':
            return torch.nn.Tanh()
        case 'sigmoid':
            return torch.nn.Sigmoid()
        case 'glu':
            return torch.nn.GLU()
        case 'swish':
            return torch.nn.SiLU()
        case 'none':
            return torch.nn.Identity()
        case _:
            raise NotImplementedError(f'Activation "{activation}" not implemented')
    

def match_parametrization(parametrization: str) -> Tuple[Callable, Callable]:
    """Retrieve the parametrization and inverse-parametrization of the diagonal matrix \Lambda given its string.

    Args:
        parametrization (str): The parametrization of the diagonal matrix \Lambda. Can be 'log', 'loglin', or 'none'.

    Raises:
        NotImplementedError: If the parametrization is not implemented.

    Returns:
        Tuple[Callable, Callable]: The parametrization and inverse-parametrization of the diagonal matrix \Lambda.
    """
    match parametrization:
        case 'log' :
            return log_parametrization_forward, log_parametrization_inverse
        case 'lin' | 'none':
            return linear_parametrization, linear_parametrization
        case _:
            raise NotImplementedError(f'Parametrization "{parametrization}" not implemented')
        

def match_metric_or_loss(loss_name: str, **kwargs) -> torch.nn.Module:
    if loss_name.lower() == 'mse' or loss_name.lower() == 'wmse':
        washout = kwargs.get('washout', 0)
        return DecayingMSELoss(**kwargs)
    elif loss_name.lower() == 'fit':
        washout = kwargs.get('washout', 0)
        return FITIndex(washout)
    else:
        raise ValueError(f'Unknown loss function {loss_name}')


class TransposableBatchNorm1d(torch.nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim < 3:
            return super().forward(input)
        else: 
            x = input.transpose(1, 2)
            return super().forward(x).transpose(1, 2)
        
@torch.jit.script
def parallel_scan(u: torch.Tensor,
                  lam_d: torch.Tensor,
                  B: torch.Tensor,
                  x0: Optional[torch.Tensor] = None,
                  return_next_y: bool = False ) -> torch.Tensor:
    """
        Simulated output with parallel scan algorithm

        x_t+1 = lam_d*x_t + B*u_t
        y_t = (Real(x_t), Imag(x_t))

    Args:
        u: (..., time, state_size)
        lam_d: (state_size) Complex valued eigenvalues of the diagonal matrix
        B: (state_size) Complex valued input transformation
        x0: (..., state_size) Complex valued initial states
        return_next_y: If True, return y_t+1 instead of y_t

    Returns:
        y (..., time, 2*state_size)
    """

    # ensure B is complex
    B = B.type(torch.complex64)
    u = u.type(torch.complex64)

    x = u@B.t()

    if not return_next_y:
        x = F.pad(x[..., :-1, :], (0, 0, 1, 0))

    if return_next_y:
        if x0 is not None:
            x[..., 0, :] += lam_d*x0
    else:
        if x0 is not None:
            x[..., 0, :] += x0

    seq_length = u.shape[-2]
    log2_length = int(torch.ceil(torch.log2(torch.tensor(seq_length, dtype=torch.int32, device=u.device))).item())

    dA = lam_d
    dA_levels = [dA]


    for d in range(log2_length):
        width = int(2 ** d)
        step = 2 * width
        offset1 = width - 1
        offset2 = step - 1
        x_l = x[..., offset1:-width:step, :].clone()
        x_r = x[..., offset2::step, :].clone()

        x_new = x_l*dA + x_r
        x[..., offset2::step, :] = x_new

        dA = dA*dA
        dA_levels.append(dA)

    dA_levels.pop()
    for d in range(log2_length - 1, -1, -1):
        width = int(2 ** d)
        step = 2 * width
        offset1 = 2 * width - 1
        offset2 = step + width - 1
        dA = dA_levels.pop()

        x_l = x[..., offset1:-width:step, :].clone()
        x_r = x[..., offset2::step, :].clone()

        x_new = x_l * dA + x_r
        x[..., offset2::step, :] = x_new

    return x