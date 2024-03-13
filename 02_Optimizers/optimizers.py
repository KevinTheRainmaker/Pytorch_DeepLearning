from typing import Dict, Tuple, Any
import torch
from torch import nn
from torch.optim.optimizer import Optimizer

class GenericAdaptiveOptimizer(Optimizer):
    def __init__(self, params, defaults: Dict[str, any], lr: float, betas: Tuple[float, float], eps: float):
        '''
        Initialize
            params: 파라미터 collections
            defaults: default 하이퍼파라미터 dict
            lr: learning rate alpha
            betas: 튜플 (beta_1, beta_2)
            eps: epsilon 값
        '''
        # check hyper-parameters
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if eps < 0.0:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if betas[0] < 0.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if betas[1] < 0.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        
        # add the hyper-parameters to the defaults
        defaults.update(dict(lr=lr, betas=betas, eps=eps))
        
        # initialize PyTorch optimizer
        super().__init__(params, defaults)
        
    def init_state(self, state: Dict[str, any], group: Dict[str, any], param: nn.Parameter):
        '''parameter tensor로 state initialize'''
        pass
    
    def step_param(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor, param: torch.Tensor):
        '''Take optimizer step on a parameter tensor'''
        pass
    
    
            