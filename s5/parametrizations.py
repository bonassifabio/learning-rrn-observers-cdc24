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
import torch


def log_parametrization_forward(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.log(-x[0]), torch.log(x[1])], dim=0)

def log_parametrization_inverse(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([-torch.exp(x[0]), torch.exp(x[1])], dim=0)

def linear_parametrization(x: torch.Tensor) -> torch.Tensor:
    return x