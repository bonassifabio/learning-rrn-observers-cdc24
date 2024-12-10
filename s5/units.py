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
import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.nn.functional as F
from control import StateSpace
from scipy.linalg import svdvals

from s5.initializations import (DiagonalLTI, FullLTI,
                                HippoDiagonalizedInitializer, S5Initializer,
                                project_back)
from s5.util import (TransposableBatchNorm1d, match_activation,
                     match_parametrization, parallel_scan)


class S5Cell(nn.Module):
    """A Simplified Structured State Space (S5) cell"""

    def __init__(self, 
                 N: int, 
                 in_features: int, 
                 out_features: int, 
                 default_Δ: float, 
                 use_parallel_scan = False,
                 return_next_y: bool = True,
                 initialization: S5Initializer = None,
                 Λ_parametrization: str = 'log',
                 learnable_Λ: bool = True,
                 learnable_eig_scale: bool = False,
                 learnable_eig_rot: bool = False,
                 initial_eig_scale: float = 1.0,
                 initial_eig_phase: float = 0.0,
                 phase_max: float = 30.0) -> None:
        """
        Single S5 Cell

        Args:
            N (int): The number of internal states
            in_features (int): The number of input features
            out_features (int): The number of output features
            default_Δ (torch.Tensor | float): The discretization time-step
            use_parallel_scan (bool, optional): Whether to use parallel scan. Defaults to False.
            return_next_y (bool, optional): Whether to return the next output (y(k+1)). Defaults to True.
            initialization (S5Initializer, optional): The initialization method. Defaults to None.
            Λ_parametrization (str, optional): The parametrization of the diagonal matrices. Defaults to 'log'.
            learnable_Λ (bool, optional): Whether to learn the state matrix. Defaults to True.
            learnable_eig_scale (bool, optional): Whether to learn the eigvenvalue timescale. Defaults to False.
            learnable_eig_rot (bool, optional): Whether to learn the eigvenvalue rotation. Defaults to False.
            initial_eig_scale (float, optional): The initial timescale. Defaults to 1.0.
            initial_eig_phase (float, optional): The initial eigvenvalue rotation. Defaults to 0.0 (deg).
            phase_max (float, optional): The maximum eigvenvalue rotation. Defaults to 30.0 (deg).

        Raises:
            NotImplementedError: If the initialization method is not implemented

        Returns:
            torch.Tensor: The output sequence (shape: (B, T, P))
        """
        
        super().__init__()
        
        self.N = N
        self.in_features = in_features
        self.out_features = out_features
        self.return_next_y = return_next_y

        self.cpxtype = torch.complex64
        self.Λ_parametrization = Λ_parametrization
        self.learnable_lambda = learnable_Λ
        self.learnable_eig_scale = learnable_eig_scale
        self.learnable_eig_phase = learnable_eig_rot
        self.max_eig_phase = phase_max / 180.0 * torch.pi
        self.use_parallel_scan = use_parallel_scan

        # Define the S5 parameters: Lambda, V, B, C, D
        if initialization is None:
            initialization = HippoDiagonalizedInitializer()

        if not isinstance(initialization, S5Initializer):
            raise ValueError(f'The initialization method should be an instance of S5Initializer, but got {type(initialization)}')
        
        diag, V = initialization(N=N, in_features=in_features, out_features=out_features)

        self._parametrization, self._inverse_parametrization = match_parametrization(self.Λ_parametrization)

        # Lambda: Parameter of shape (2, N/2)
        if self.learnable_lambda:
            self._Lambda = nn.Parameter(self._parametrization(self._cpxmat_to_tensor(diag.Λ)), requires_grad=True)
        else:
            self.register_buffer('_Lambda', self._parametrization(self._cpxmat_to_tensor(diag.Λ)))
        
        # B: Parameter of shape (2, N/2, M)
        self._B = nn.Parameter(self._cpxmat_to_tensor(diag.Bd), requires_grad=True)
        # C: Parameter of shape (2, P, N)
        self.C = nn.Parameter(self._cpxmat_to_tensor(diag.Cd), requires_grad=True)

        self.register_buffer('V', torch.tensor(V, dtype=self.cpxtype, requires_grad=False))

        _Δ_default = torch.tensor(default_Δ, dtype=torch.float32, requires_grad=False).view(1)
        self.register_buffer('default_Δ', _Δ_default)

        # Timescale: Parameter of shape (1,)
        _eig_scale_default = torch.tensor([initial_eig_scale], dtype=torch.float32, requires_grad=False).log()
        _eig_phase_default = torch.tensor([initial_eig_phase], dtype=torch.float32, requires_grad=False) / 180.0 * torch.pi

        if self.learnable_eig_scale:
            self.kappa = nn.Parameter(_eig_scale_default, requires_grad=True)
        else:
            self.register_buffer('kappa', _eig_scale_default)

        if self.learnable_eig_phase:
            self.theta = nn.Parameter(_eig_phase_default, requires_grad=True)
        else:
            self.register_buffer('theta', _eig_phase_default)

    @property
    def Δ(self) -> torch.Tensor:
        """Get the default discretization time-step"""
        return self.default_Δ
    
    @property
    def Lambda(self) -> torch.Tensor:
        """Get the state matrix Lambda"""
        if self.learnable_eig_scale:
            _theta = self.max_eig_phase * F.tanh(self.theta)
            return self.kappa.exp() * torch.exp(_theta * 1j) * self._inverse_parametrization(self._Lambda)
        else:
            return self._inverse_parametrization(self._Lambda)
        
    @property
    def B(self) -> torch.Tensor:
        """Get the input matrix B"""
        if self.learnable_eig_scale:
            return  self.kappa.exp() * self._B 
        else:
            return self._B
    
    def poles_zeros_gains(self) -> tuple[np.ndarray, np.ndarray, tuple[float, float], float, float]:
        """Get the poles, zeros, gains (minumum-maximum singular values), scaling and rotation of the underlying continuous-time LTI system.

        Returns:
            tuple[np.ndarray, np.ndarray, tuple[float, float], float, float]: The tuple (poles, zeros, gains, kappa, theta)
        """
        Λ = self._tensor_to_cpxmat(self.Lambda).detach().numpy()
        B_tilde = self._tensor_to_cpxmat(self.B).detach().numpy()
        C_tilde = self._tensor_to_cpxmat(self.C).detach().numpy()
        D_tilde = np.zeros((self.out_features, self.in_features))
        diagonal = DiagonalLTI(Λ, B_tilde, C_tilde, D_tilde)
        V = self.V.detach().numpy()

        full = project_back(diagonal, V, force_real=True)

        ss = StateSpace(full.A, full.B, full.C, full.D)
        g0 = ss.dcgain().reshape((self.out_features, self.in_features))
        Σ = svdvals(g0)
        kappa = self.kappa.exp().detach().numpy().item()
        theta = (self.max_eig_phase * F.tanh(self.theta)).detach().numpy().item()

        return ss.poles(), ss.zeros(), (Σ.min().item(), Σ.max().item()), kappa, theta

    def _cpxmat_to_tensor(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert a (M, N) complex matrix into a tensor of shape (2, M, N)"""
        if isinstance(x, np.ndarray):
            return torch.stack((torch.tensor(x.real, dtype=torch.float32), torch.tensor(x.imag, dtype=torch.float32)), dim=0)
        elif isinstance(x, torch.Tensor):
            return torch.stack((x.real, x.imag), dim=0)
    
    def _tensor_to_cpxmat(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a tensor of shape (2, M, N) into a (M, N) complex matrix"""
        return (x[0] + 1j * x[1]).type(self.cpxtype)

    def discretize(self, Δ: torch.Tensor = None, method: str = 'zoh') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Discretize the S5Cell with the given discretization method

        Args:
            Δ (torch.Tensor): The discretization time-step. Defaults to None, i.e. use the default value.
            method (str, optional): The discretization method. Defaults to 'zoh'.

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The discretized system matrices (Λd, Bd, Cd)
        """

        # self.default_Δ is taken as fallback value in case Δ is not specificed. 
        # TODO: Ensure this is the best way to do this
        if Δ is None:
            Δ = self.Δ

        Λ_cpx = self._tensor_to_cpxmat(self.Lambda)  # shape: (N/2,)
        B_cpx = self._tensor_to_cpxmat(self.B)  # shape: (N/2, M)
        C_cpx = self._tensor_to_cpxmat(self.C)  # shape: (P, N/2)
        
        if method == 'zoh':
            Λd = torch.exp(Λ_cpx * Δ)   
            IL = torch.reshape((Λd - 1) / Λ_cpx, (-1, 1))
            Bd = IL * B_cpx
            Cd = C_cpx
        else: 
            raise NotImplementedError(f'Method "{method}" not implemented')

        return Λd, Bd, Cd
        
    def step(self,
             xk: torch.Tensor, 
             uk: torch.Tensor, 
             Λd: torch.Tensor, 
             Bd: torch.Tensor, 
             Cd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a single simulation step of the S5 Cell

        Args:
            Λd (torch.Tensor): The discretized state matrix Λd (with half the eigenvalues) (shape: (N/2,))
            Bd (torch.Tensor): The discretized input matrix Bd (shape: (N/2, M))
            Cd (torch.Tensor): The discretized output matrix Cd (shape: (P, N/2))
            xk (torch.Tensor): The current state (shape: (B, N/2))
            uk (torch.Tensor): The current input (shape: (B, M))

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The tuple (x_{k+1}, y_k)
        """
        # Print the shapes for debug
        # print(f'xk: {xk.shape}, uk: {uk.shape}, Λd: {Λd.shape}, Bd: {Bd.shape}, Cd: {Cd.shape}')
        # print(f'Λd.reshape(-1, 1): {Λd.reshape(1, -1).shape}, Bd.t().unsqueeze(0): {Bd.t().unsqueeze(0).shape}')
        
        # Compute the next state
        xk1 = xk * Λd.unsqueeze(0) + uk @ Bd.t().unsqueeze(0)  # shape: (B, N/2)

        # if self.Φ is not None:
        #     xk1 = xk1 * F.sigmoid(xk1 @ self.Φ)  # shape: (B, N/2

        if self.return_next_y:
            yk = xk1 @ Cd.t().unsqueeze(0)   # shape: (B, P)
        else:
            yk = xk @ Cd.t().unsqueeze(0)  # shape: (B, P)
        
        # Project the output back onto the real axis (as we are removing the complex conjugates)
        yk = 2 * torch.real(yk)

        return xk1, yk
    
    def simulation(self,
                   u: torch.Tensor, 
                   x0: torch.Tensor = None, 
                   Δ: torch.Tensor = None, 
                   method: str = 'zoh') -> torch.Tensor:
        """Simulate the S5 Cell for a given input sequence

        Args:
            u (torch.Tensor): The input sequence (shape: (B, T, M))
            x0 (torch.Tensor): The initial state (shape: (B, N/2)). Defaults to None, i.e. null state.
            Δ (torch.float32): The discretization time-step. Defaults to None, i.e. use the default value.
            method (str, optional): The discretization method. Defaults to 'zoh'.

        Returns:
            torch.Tensor: The output sequence (shape: (B, T, P))
        """
        # Discretize the S5 Cell
        Λd, Bd, Cd = self.discretize(Δ, method)
        device = u.device
        if x0 is None:
            xk = torch.zeros(u.shape[0], self.N // 2, dtype=self.cpxtype, requires_grad=False, device=device)
        else:
            xk = x0.type(self.cpxtype)

        y = torch.empty(u.shape[0], u.shape[1], self.out_features, device=u.device)

        for k, uk in enumerate(u.transpose(0, 1)):
            uk = uk.type(self.cpxtype)
            xk, yk = self.step(xk, uk, Λd, Bd, Cd)
            y[:, k, :] = yk

        # y = torch.cat(y, dim=0)
        # y = y.transpose(0, 1).type(torch.float32)  # shape: (B, T, P)
        return y

    def simulation_parallel_scan(self,
                                 u: torch.Tensor,
                                 x0: torch.Tensor = None,
                                 Δ: torch.Tensor = None,
                                 method: str = 'zoh') -> torch.Tensor:

        Λd, Bd, Cd = self.discretize(Δ, method)

        x = parallel_scan(u, Λd, Bd, x0, self.return_next_y)

        # x: ( B, T, N/2) Complex
        # Cd : (P, N/2) Complex
        y = x @ Cd.t().unsqueeze(0)
        y = 2*y.real
        return y

    def forward(self,
                u: torch.Tensor, 
                x0: torch.Tensor = None, 
                Δ: torch.Tensor = None, 
                method: str = 'zoh') -> torch.Tensor:
        """Simulate the S5 Cell for a given input sequence

        Args:
            u (torch.Tensor): The input sequence (shape: (B, T, M))
            x0 (torch.Tensor): The initial state (shape: (B, N/2)). Defaults to None, i.e. null state.
            Δ (torch.float32): The discretization time-step. Defaults to None, i.e. use the default value.
            method (str, optional): The discretization method. Defaults to 'zoh'.

        Returns:
            torch.Tensor: The output sequence (shape: (B, T, P))
        """
        if self.use_parallel_scan:
            return self.simulation_parallel_scan(u, x0, Δ, method)
        else:
            return self.simulation(u, x0, Δ, method)
    

class S5Layer(torch.nn.Module):
    def __init__(self, 
                 N: int, 
                 in_features: int, 
                 out_features: int, 
                 default_Δ: torch.Tensor | float,
                 use_parallel_scan: bool = False,
                 return_next_y: bool = True,
                 initialization: S5Initializer = None,
                 Λ_parametrization: str = 'log',
                 batch_norm: str = 'post',
                 dropout_p: float = 0.0,
                 activation: str = 'gelu',
                 skip_connection: bool = True,
                 skip_identity: bool = False,
                 learnable_Λ: bool = True,
                 learnable_eig_scale: bool = False,
                 learnable_eig_rot: bool = False,
                 initial_eig_scale: float = 1.0,
                 initial_eig_phase: float = 0.0,
                 phase_max: float = 30.0) -> None:
        """
        Single block of S5 layers, wrapping `S5Cell`

        Args:
            N (int): The number of internal states
            in_features (int): The number of input features
            out_features (int): The number of output features
            default_Δ (torch.Tensor | float): The discretization time-step
            use_parallel_scan (bool, optional): Whether to use parallel scan. Defaults to False.
            return_next_y (bool, optional): Whether to return the next output (y(k+1)). Defaults to True.
            initialization (S5Initializer, optional): The initialization method. Defaults to None.
            Λ_parametrization (str, optional): The parametrization of the diagonal matrices. Defaults to 'log'.
            batch_norm (str, optional): Specifies which BatchNorm to apply. Options: ['pre', 'post', 'none']. Defaults to 'post'.
            dropout_p (float, optional): The dropout probability. Defaults to 0.0.
            activation (str, optional): The activation function. Defaults to 'gelu'.
            skip_connection (bool, optional): Whether to apply a skip connection. Defaults to True.
            skip_identity (bool, optional): Whether to use the identity matrix for the skip connection (if in_features = out_features). Defaults to False.
            learnable_Λ (bool, optional): Whether to learn the state matrix. Defaults to True.
            learnable_eig_scale (bool, optional): Whether to learn the eigvenvalue timescale. Defaults to False.
            learnable_eig_rot (bool, optional): Whether to learn the eigvenvalue rotation. Defaults to False.
            initial_eig_scale (float, optional): The initial timescale. Defaults to 1.0.
            initial_eig_phase (float, optional): The initial eigvenvalue rotation. Defaults to 0.0 (deg).
            phase_max (float, optional): The maximum eigvenvalue rotation. Defaults to 30.0 (deg).

        Raises:
            NotImplementedError: If the activation function is not implemented

        Returns:
            torch.Tensor: The output sequence (shape: (B, T, P))
        """
        
        super().__init__()

        self.batch_norm_mode = batch_norm
        if batch_norm.startswith('pre'):
            affine = not batch_norm.endswith('noaffine')
            self.batch_norm = TransposableBatchNorm1d(in_features, affine=affine) 
        elif batch_norm.startswith('post'):
            affine = not batch_norm.endswith('noaffine')
            self.batch_norm = TransposableBatchNorm1d(out_features, affine=affine) 

        self.cell = S5Cell(N, in_features, out_features, default_Δ, 
                           return_next_y=return_next_y,
                           initialization=initialization, 
                           Λ_parametrization=Λ_parametrization,
                           use_parallel_scan=use_parallel_scan,
                           learnable_Λ=learnable_Λ,
                           learnable_eig_rot=learnable_eig_rot,
                           learnable_eig_scale=learnable_eig_scale,
                           initial_eig_scale=initial_eig_scale,
                           initial_eig_phase=initial_eig_phase,
                           phase_max=phase_max)
        
        self.activation = match_activation(activation)

        self.dropout = torch.nn.Dropout(dropout_p) if dropout_p > 0.0 else None

        if skip_connection:
            self.skip = torch.nn.Identity() if (in_features == out_features and skip_identity) else torch.nn.Linear(in_features, out_features)
        else:
            self.skip = None
        

    def poles_zeros_gains(self) -> tuple[np.ndarray, np.ndarray, tuple[float, float], float, float]:
        """Get the poles, zeros, gains, scaling and rotation of the underlying continuous-time LTI system.

        Returns:
            tuple[torch.Tensor, torch.Tensor, tuple[float, float], float, float]: The tuple (poles, zeros, gains, kappa, theta)
        """
        return self.cell.poles_zeros_gains()

    def skip_norm(self) -> float:
        """Get the norm of the skip connection

        Returns:
            float: The norm of the skip connection
        """
        if self.skip:
            if isinstance(self.skip, torch.nn.Identity):
                return 1.0
            else:
                W = torch.cat([self.skip.weight, self.skip.bias.unsqueeze(1)], dim=1)
                return torch.linalg.matrix_norm(W, ord=2).detach().numpy().item()
        else:
            return 0.0

    def forward(self, u: torch.Tensor, x0: torch.Tensor = None, Δ: torch.Tensor = None, method: str = 'zoh', return_hidden: bool = False) \
        -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Simulate the S5 Layer for a given input sequence

        Args:
            u (torch.Tensor): The input sequence (shape: (B, T, M))
            x0 (torch.Tensor): The initial state (shape: (B, N/2)). Defaults to None, i.e. null state.
            Δ (torch.float32): The discretization time-step. Defaults to None, i.e. use the default value.
            method (str, optional): The discretization method. Defaults to 'zoh'.
            return_hidden (bool, optional): Whether to return the hidden outputs of the model. Defaults to False.

        Returns:
            torch.Tensor: The output sequence (shape: (B, T, P)) if return_hidden is False, else the tuple (y, u)
            tuple[torch.Tensor, torch.Tensor]: The output sequence (shape: (B, T, P)) and the hidden output sequence (shape: (B, T, H)) if return_hidden is True
        """

        # Apply the pre-normalization
        u_ = self.batch_norm(u) if self.batch_norm_mode == 'pre' else u

        # Apply the S5 Cell
        y = self.cell(u_, x0, Δ, method)

        # Apply the activation
        h = self.activation(y)

        # Apply the dropout
        y = self.dropout(h) if self.dropout else h

        # Apply the skip connection
        # TODO: Skip before or after the Batch Normalization?
        # self.skip(u_) -> self.skip(u)
        y = y + self.skip(u_) if self.skip else y
        
        # Apply the post-normalization
        y = self.batch_norm(y) if self.batch_norm_mode == 'post' else y

        if return_hidden:
            return y, h
        else:
            return y





