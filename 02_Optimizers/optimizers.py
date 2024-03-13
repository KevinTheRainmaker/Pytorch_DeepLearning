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
    
    @torch.no_grad()
    def step(self, closure=None):
        '''Optimizer step: a template method for every Adam based optimizer needs'''
        
        # calculate loss
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        # iterate through the parameter groups
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    # skip if the parameter has no gradient
                    continue
                
                # get the gradient tensor
                grad = param.grad.data
                
                # get the state for the parameter
                state = self.state[param]
                
                if len(state)==0:
                    # initialize state if state is uninitialized
                    self.init_state(state, group, param)
                
                # take the optimization step on the parameter
                self.step_param(state, group, grad, param)
                
        return loss

