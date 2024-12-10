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
from typing import Any, Callable, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from s5.initializations import S5Initializer
from s5.units import S5Layer


class S5Model(torch.nn.Module):

    train_loss: Callable
    val_metric: Callable
    test_metric: Callable
    optimizer: torch.optim.Optimizer
    optimizer_opts: dict
    washout: int = 1

    def __init__(self, 
                 N: list[int],
                 H: int | list[int],
                 in_features: int, 
                 out_features: int, 
                 default_Δ: torch.Tensor | float = 1.0,
                 use_parallel_scan: bool = False,
                 return_next_y: bool = True,
                 return_sequence: bool = True,
                 initialization: S5Initializer = None,
                 batch_norm: str = 'post',
                 dropout_p: float = 0.0,
                 activation: str = 'gelu',
                 skip_connection: bool = True,
                 skip_identity: bool = False,
                 discretization_method: str = 'zoh',
                 learnable_Λ: bool = True,
                 learnable_eig_scale: bool = False,
                 learnable_eig_rot: bool = False,
                 initial_eig_scale: float = 1.0,
                 initial_eig_phase: float = 0.0,
                 phase_max: float = 30.0) -> None:
        """Construct the S5 Model

        Args:
            N (list[int]): Number of states of each layer.
            H (int | list[int]): Number of hidden features of each layer. If `H` is an integer, then all layers have the same number of hidden features.
            in_features (int): Number of features of the input signal.
            out_features (int): Number of features of the output signal.
            default_Δ (torch.Tensor | float, optional): Default discretization step. Defaults to 1.0.
            use_parallel_scan (bool, optional): Whether to use the parallel scan. Defaults to False.
            return_next_y (bool, optional): Whether to return the next output of the model. Defaults to True.
            initialization (S5Initializer, optional): S5 Initializer. Defaults to None, which means that the default initializer is used.
            batch_norm (str, optional): Specifies which BatchNorm to apply. Options: ['pre', 'post', 'none']. Defaults to 'post'.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            activation (str, optional): Nonlinear activation function. Defaults to 'gelu'.
            skip_connection (bool, optional): Enable skip connection. Defaults to True.
            skip_identity (bool, optional): Whether to use the identity matrix for the skip connection (if in_features = out_features). Defaults to False.
            discretization_method (str, optional): Discretization method. Defaults to 'zoh'.            learnable_Λ (bool, optional): Whether to learn the state matrix. Defaults to True.
            learnable_eig_scale (bool, optional): Whether to learn the eigvenvalue timescale. Defaults to False.
            learnable_eig_rot (bool, optional): Whether to learn the eigvenvalue rotation. Defaults to False.
            initial_eig_scale (float, optional): The initial timescale. Defaults to 1.0.
            initial_eig_phase (float, optional): The initial eigvenvalue rotation. Defaults to 0.0 (deg).
            phase_max (float, optional): The maximum eigvenvalue rotation. Defaults to 30.0 (deg).
        """
        super().__init__()

        if not isinstance(N, list):
            N = [N]
        if not isinstance(H, list):
            H = [H] * len(N)
        
        self.in_features = in_features
        self.out_features = out_features
        self.discretization_method = discretization_method
        self.return_sequence = return_sequence
        self.layers = torch.nn.ModuleList()

        # Build up the S5 Model as a concatenation of S5 Layers
        for i, n in enumerate(N):
            in_features_ = in_features if i == 0 else H[i - 1]
            out_features_ = out_features if i == len(N) - 1 else H[i]
            _next_y = True if i < len(N) - 1 else return_next_y 

            self.layers.append(S5Layer(N=n, 
                                       in_features=in_features_, 
                                       out_features=out_features_, 
                                       default_Δ=default_Δ, 
                                       use_parallel_scan=use_parallel_scan,
                                       return_next_y=_next_y,
                                       initialization=initialization,
                                       batch_norm=batch_norm,
                                       dropout_p=dropout_p,
                                       activation=activation,
                                       skip_connection=skip_connection,
                                       skip_identity=skip_identity,
                                       learnable_Λ=learnable_Λ,
                                       learnable_eig_scale=learnable_eig_scale,
                                       learnable_eig_rot=learnable_eig_rot,
                                       initial_eig_scale=initial_eig_scale,
                                       initial_eig_phase=initial_eig_phase,
                                       phase_max=phase_max))

        self.params_str = {}
        self.params_str['model'] = self.__class__.__name__ 
        self.params_str['N']  = f'[{", ".join([str(n) for n in N])}]'
        self.params_str['H']  = f'[{", ".join([str(h) for h in H])}]'
        self.params_str['in_features'] = in_features
        self.params_str['out_features'] = out_features
        self.params_str['use_parallel_scan'] = use_parallel_scan
        self.params_str['batch_norm'] = batch_norm
        self.params_str['dropout_p'] = dropout_p
        self.params_str['activation'] = activation
        self.params_str['skip_connection'] = skip_connection
        self.params_str['skip_identity'] = skip_identity
        self.params_str['discretization_method'] = discretization_method
        self.params_str['learnable_Λ'] = learnable_Λ
        self.params_str['learnable_eig_scale'] = learnable_eig_scale
        self.params_str['learnable_eig_rot'] = learnable_eig_rot
        self.params_str['initial_eig_scale'] = initial_eig_scale
        self.params_str['initial_eig_phase'] = initial_eig_phase
        self.params_str['phase_max'] = phase_max


    def training_settings(self, 
                          train_loss: Callable = None, 
                          val_metric: Callable = None, 
                          test_metric: Callable = None,
                          optimizer: torch.optim.Optimizer = None,
                          optimizer_opts: dict = None,
                          washout: int = None) -> None:
        """Set the training settings of the model

        Args:
            train_loss (Callable, optional): Loss function. If None, the default loss function of the model is used.
            val_metric (Callable, optional): Validation metric. If None, the default validation metric of the model is used.
            test_metric (Callable, optional): Test metric. If None, the default test metric of the model is used.
            optimizer (torch.optim.Optimizer, optional): Optimizer. If None, the default optimizer of the model is used.
            optimizer_opts (dict, optional): Dictionary of optimizer options. If None, the default optimizer options of the model are used.
        """
        if train_loss is not None:
            self.train_loss = train_loss
        if val_metric is not None:
            self.val_metric = val_metric
        if test_metric is not None:
            self.test_metric = test_metric
        if optimizer is not None:
            self.optimizer = optimizer
        if optimizer_opts is not None:
            self.optimizer_opts = optimizer_opts
        if washout is not None:
            self.washout = washout

    def forward(self, 
                u: torch.Tensor, 
                x0: list[torch.Tensor] = None, 
                Δ: torch.Tensor = None, 
                method: str = None,
                return_hidden: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """ Simulate the whole S5 Model on the batch of input sequences

        Args:
            u (torch.Tensor): Input sequence of shape (B, T, M)
            x0 (list[torch.Tensor], optional): Initial state of the model. Defaults to None, which means that the initial state is zero.
            Δ (torch.Tensor, optional): Discretization step. Defaults to None, which means that the default discretization step of the model is used.
            method (str, optional): Discretization method. Defaults to None, which means that the default discretization method of the model is used.
            return_hidden (bool, optional): Whether to return the hidden outputs of the model. Defaults to False.

        Returns:
            torch.Tensor: Output sequence of shape (B, T, P) if `return_hidden` is False
            tuple[torch.Tensor, list[torch.Tensor]]: Output sequence of shape (B, T, P) and a list of hidden outputs of shape (B, T, H) if `return_hidden` is True
        """
        method_ = self.discretization_method if method is None else method

        if x0 is None:
            x0 = [None] * len(self.layers)

        hidden = [None] * len(self.layers)

        y = u
        for i, (x0, layer) in enumerate(zip(x0, self.layers)):
            res = layer(y, x0=x0, Δ=Δ, method=method_, return_hidden=return_hidden)
            y, hidden[i] = res if return_hidden else (res, None)

        if not self.return_sequence:
            y = y[:, -1, :].unsqueeze(1)

            if return_hidden:
                hidden = [h[:, -1, :].unsqueeze(1) for h in hidden]
            

        if return_hidden:
            return y, hidden
        else:
            return y
    
    def poles_zeros_gains(self):
        """Retrieve the poles, zeros, gains (range of singular values), eigenvalues scales and rotations of the model"""
        poles = []
        zeros = []
        gains = []
        kappa = []
        theta = []

        for layer in self.layers:
            p, z, k, t, r = layer.poles_zeros_gains()
            poles.append(p)
            zeros.append(z)
            gains.append(k)
            kappa.append(t)
            theta.append(r)

        return poles, zeros, gains, kappa, theta
    
    def skip_norm(self) -> float:
        """Get the norm of the skip connection

        Returns:
            float: The norm of the skip connection
        """
        skip_norm = [] 

        for layer in self.layers:
            skip_norm.append(layer.skip_norm())

        return skip_norm
    
    def short_params(self) -> dict:
        """Get the short parameters of the model

        Returns:
            dict: The short parameters of the model
        """
        short_params = dict()

        for i, layer in enumerate(self.layers):
            short_params[f'layer_{i}'] = layer.short_params()

        return short_params


class SingleLayerGRU(torch.nn.Module):

    def __init__(self, hidden_states: int, in_features: int, out_features: int) -> None:
        super().__init__()

        self.hidden_states = hidden_states
        self.in_features = in_features
        self.out_features = out_features

        _kernel_fz = torch.zeros(in_features + hidden_states, 2*hidden_states)
        _bias_fz = torch.zeros(2*hidden_states)
        _kernel_r = torch.zeros(in_features + hidden_states, hidden_states)
        _bias_r = torch.zeros(hidden_states)

        torch.nn.init.xavier_normal_(_kernel_fz, gain=0.5)
        torch.nn.init.xavier_normal_(_kernel_r, gain=0.5)
        
        self.kernel_fz = torch.nn.Parameter(_kernel_fz)
        self.bias_fz = torch.nn.Parameter(_bias_fz)
        self.kernel_r = torch.nn.Parameter(_kernel_r)
        self.bias_r = torch.nn.Parameter(_bias_r)
        self.out = torch.nn.Linear(hidden_states, out_features)

        self.params_str = {}
        self.params_str['model'] = self.__class__.__name__ 
        self.params_str['N'] = hidden_states
        self.params_str['in_features'] = in_features
        self.params_str['out_features'] = out_features
        
    def step(self,
             u: torch.Tensor,
             x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        # u: (B, in_features)
        # x0: (B, hidden_states)
        u_ = torch.cat([u, x0], dim=-1)   # (B, in_features + hidden_states)
        fz = u_ @ self.kernel_fz + self.bias_fz  # (B, 2*hidden_states)
        fz = F.sigmoid(fz)
        f, z = torch.split(fz, self.hidden_states, dim=-1)

        ur_ = torch.cat([u, f * x0], dim=-1) # (B, in_features + hidden_states)
        r = ur_ @ self.kernel_r + self.bias_r
        r = F.tanh(r)

        xk = (1 - z) * x0 + z * r
        y = self.out(x0)

        return y, xk
    

    def simulate(self,
                u: torch.Tensor,
                x0: list[torch.Tensor] = None,
                return_state_seq: bool = False) -> torch.Tensor:
        
        if x0 is None:
            x0 = torch.rand(u.shape[0],  1, self.hidden_states)

        y = torch.empty(u.shape[0], u.shape[1], self.out_features, device=u.device)

        if return_state_seq:
            x = torch.empty(u.shape[0], u.shape[1], self.hidden_states, device=u.device)

        xk_ = x0.squeeze(1)
        
        for t, u_ in enumerate(u.transpose(0, 1)):
            y_, xk_ = self.step(u_, xk_)
            y[:, t, :] = y_
            
            if return_state_seq:
                x[:, t, :] = xk_

        if return_state_seq:
            return y, x
        else:
            return y
    

    def forward(self, 
                u: torch.Tensor, 
                x0: list[torch.Tensor] = None) -> torch.Tensor:
        
        return self.simulate(u, x0)
    

class SingleLayerLSTM(torch.nn.Module):

    def __init__(self, hidden_states: int, in_features: int, out_features: int) -> None:
        super().__init__()

        self.hidden_states = hidden_states
        self.in_features = in_features
        self.out_features = out_features

        _kernel_fiz = torch.zeros(in_features + hidden_states, 3*hidden_states)
        _bias_fiz = torch.zeros(3*hidden_states)
        _kernel_r = torch.zeros(in_features + hidden_states, hidden_states)
        _bias_r = torch.zeros(hidden_states)

        torch.nn.init.xavier_normal_(_kernel_fiz, gain=0.5)
        torch.nn.init.zeros_(_bias_fiz)
        torch.nn.init.xavier_normal_(_kernel_r, gain=0.5)
        torch.nn.init.zeros_(_bias_r)
        
        self.kernel_fiz = torch.nn.Parameter(_kernel_fiz)
        self.bias_fiz = torch.nn.Parameter(_bias_fiz)
        self.kernel_r = torch.nn.Parameter(_kernel_r)
        self.bias_r = torch.nn.Parameter(_bias_r)
        self.out = torch.nn.Linear(hidden_states, out_features)

        self.params_str = {}
        self.params_str['model'] = self.__class__.__name__ 
        self.params_str['N'] = hidden_states
        self.params_str['in_features'] = in_features
        self.params_str['out_features'] = out_features
        
    def step(self,
             u: torch.Tensor,
             h0: torch.Tensor,
             c0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        # u: (B, in_features)
        # x0: (B, hidden_states)
        u_ = torch.cat([u, h0], dim=-1)   # (B, in_features + hidden_states)
        fiz = u_ @ self.kernel_fiz + self.bias_fiz  # (B, 3*hidden_states)
        fiz = F.sigmoid(fiz)
        f, i, z = torch.split(fiz, self.hidden_states, dim=-1)

        r = u_ @ self.kernel_r + self.bias_r  # (B, hidden_states)
        r = F.tanh(r)

        ck = f * c0 + i * r
        hk = z * F.tanh(ck)
        y = self.out(hk)

        return y, hk, ck
    

    def simulate(self,
                u: torch.Tensor,
                x0: torch.Tensor = None,
                return_state_seq: bool = False) -> torch.Tensor:
        
        if x0 is None:
            h0 = torch.rand(u.shape[0],  1, self.hidden_states)
            c0 = torch.rand(u.shape[0],  1, self.hidden_states)
        else:
            h0, c0 = x0.split(self.hidden_states, dim=-1)

        y = torch.empty(u.shape[0], u.shape[1], self.out_features, device=u.device)

        if return_state_seq:
            c = torch.empty(u.shape[0], u.shape[1], self.hidden_states, device=u.device)
            h = torch.empty(u.shape[0], u.shape[1], self.hidden_states, device=u.device)

        hk_ = h0.squeeze(1)
        ck_ = c0.squeeze(1)
        
        for t, u_ in enumerate(u.transpose(0, 1)):
            y_, hk_, ck_ = self.step(u_, hk_, ck_)
            y[:, t, :] = y_

            if return_state_seq:
                c[:, t, :] = ck_
                h[:, t, :] = hk_

        if return_state_seq:
            return y, torch.cat((c, h), dim=-1)
        else:
            return y
    

    def forward(self, 
                u: torch.Tensor, 
                x0: list[torch.Tensor] = None) -> torch.Tensor:
                
        return self.simulate(u, x0)
    

class FFNNObserver(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, context_window: int) -> None:
        super().__init__()

        self.in_features = in_features * context_window
        self.out_features = out_features

        self.ffnn = torch.nn.Sequential(torch.nn.Linear(self.in_features, self.in_features // 2),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(self.in_features // 2, self.out_features),
                                        torch.nn.Sigmoid())
        
        self.params_str = {}
        self.params_str['model'] = self.__class__.__name__ 
        self.params_str['fixed_context_window'] = context_window
        self.params_str['in_features'] = in_features
        self.params_str['out_features'] = out_features
    
    def forward(self, 
                u: torch.Tensor, 
                _: list[torch.Tensor] = None) -> torch.Tensor:
        
        return self.ffnn(u.flatten(1))
    

class CNNObserver(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, hidden_channels: int, scan_window: int) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_channels = hidden_channels
        self.scan_window = scan_window

        # Dimension of the inner convolution

        self.cnn = torch.nn.Sequential(torch.nn.Conv1d(in_channels=self.in_features, 
                                                       out_channels=self.hidden_channels, 
                                                       kernel_size=self.scan_window, 
                                                       padding=self.scan_window // 2),
                                        torch.nn.GELU(),
                                        torch.nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.out_features, kernel_size=scan_window*2, stride=scan_window//4))
        
        self.params_str = {}
        self.params_str['model'] = self.__class__.__name__ 
        self.params_str['hidden_channels'] = hidden_channels
        self.params_str['scan_window'] = scan_window
        self.params_str['in_features'] = in_features
        self.params_str['out_features'] = out_features
    
        
    def forward(self, 
            u: torch.Tensor, 
            _: list[torch.Tensor] = None) -> torch.Tensor:
            
        y = self.cnn(u.transpose(1, 2)).transpose(1, 2)
        y = y[:, -1, :].unsqueeze(1)
        y = F.tanh(y)
        return y
