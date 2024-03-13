from typing import Dict, Tuple, Any
import torch
from torch import nn
from torch.optim.optimizer import Optimizer

# base class for Adam and extensions
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

# L2 Weight decay
class WeightDecay:
    def __init__(self, weight_decay: float = 0., weight_decouple: bool = True, absolute: bool = False):
        '''
        Initialize
            weight_decay: decay coefficient
            weight_decouple: gradient에 weight decay를 더할지(normal optim update)
            혹은 parameter로부터 직접 decay할지를 나타내는 flag
            absolute: weight decay coefficient가 절대값인지를 나타내는 flag
            이 옵션은 parameter로부터 직접 decay를 수행할 때만 유효
        '''
        
        # check hyper-parameter
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.absolute = absolute
        
    def defaults(self):
        '''return defaults parameter groups'''
        return dict(weight_decay=self.weight_decay)
    
    def __call__(self, param: torch.nn.Parameter, grad: torch.Tensor, group: Dict[str, any]):
        '''perform weight decay and return gradient'''

        if self.weight_decouple: # decay on the parameter directly
            if self.absolute:
                param.data.mul_(1.0 - group['weight_decay'])
            else:
                param.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                
            return grad
        else:
            if group['weight_decay'] != 0:
                
                # add the weight decay to the gradient and return modified gradient
                return grad.add(param.data, alpha=group['weight_decay'])
            else:
                return grad