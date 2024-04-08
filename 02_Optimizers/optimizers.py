from typing import Dict, Any, Tuple, Optional
import math

import torch
from torch import nn
from torch import Tensor
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

class Adam(GenericAdaptiveOptimizer):
    def __init__(self, params, 
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-16,
                 weight_decay: WeightDecay = WeightDecay(),
                 optimized_update: bool = True,
                 defaults: Optional[Dict[str, Any]] = None):
        '''
        Initialize the Optimizer
            - params: the list of parameters
            - lr: learning rate alpha
            - betas: tuple of (beta_1, beta_2)
            - eps: epsilon
            - weight_decay: instance of class WeightDecay
            - optimized_update: a flag whether to optimize the bias correction 
                                of the second moment by doing it after adding epsilon
            - defaults: a dict of default for group values
        '''
        defaults = {} if defaults is None else defaults
        defaults.update(weight_decay.defaults())
        super().__init__(params, defaults, lr, betas, eps)
            
        self.weight_decay = weight_decay
        self.optimized_update = optimized_update
    
    def init_state(self, state: Dict[str, any],
                   group: Dict[str, any],
                   param: nn.Parameter):
        '''
        Initialize a parameter state
            - state: the optimizer state of the parameter (tensor)
            - group: stores optimizer attributes of the parameter group
            - param: the parameter tensor theta at t-1
        '''
        state['step'] = 0 # the number of optimizer steps taken on t
        # exponential moving avg of gradients, m_t
        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format) 
        # exponeitial moving avg of squared gradient values, v_t
        state['exp_avg_sqrd'] = torch.zeros_like(param, memory_format=torch.preserve_format)
    
    def calc_mv(self, state: Dict[str, Any],
                group: Dict[str, Any], grad: torch.Tensor):
        '''
        Calculate m_t and v_t
            - state: the optimizer state of the parameter (tensor)
            - group: stores optimizer attributes of the parameter group
            - grad: current gradient tensor g_t for theta at t-1
        '''
        beta1, beta2 = group['betas']
        m, v = state['exp_avg'], state['exp_avg_sqrd']
        
        # calculation of m_t (inplace calculation)
        m.mul_(beta1).add_(grad, alpha=1-beta1)
        # == beta1 * m + (1 - beta1) * grad
        
        # calculation of v_t
        v.mul_(beta2).add_(grad**2, alpha=1-beta2)
        
        return m, v
    
    def get_lr(self, state: Dict[str, any], group: Dict[str, any]):
        '''
        returns the modified lr based on the state
        '''
        return group['lr']
    
    def update_adam(self, state: Dict[str, any],
                    group: Dict[str, any],
                    param: torch.nn.Parameter,
                    m: torch.Tensor, v: torch.Tensor):
        '''
        Update the Adam parameter
            - state: the optimizer state of the parameter (tensor)
            - group: stores optimizer attributes of the parameter group
            - param: the parameter tensor theta at t-1
            - m, v: the uncorrected first and second moments m_t and v_t
        '''
        beta1, beta2 = group['betas']
        
        # bias correction term 1-beta1^t
        bias_correction1 = 1 - beta1 ** state['step']
        
        # bias correction term 1-beta2^t
        bias_correction2 = 1 - beta2 ** state['step']
        
        lr = self.get_lr(state, group)
        
        if self.optimized_update:
            denominator = v.sqrt().add_(group['eps'])
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        else:
            # computation without optimization
            denominator = (v.sqrt()/math.sqrt(bias_correction2)).add_(group['eps'])
            step_size = lr / bias_correction1
            
        param.data.addcdiv_(m, denominator, value=-step_size)
        
    def step_param(self, state: Dict[str, Any], 
                   group: Dict[str, Any], grad: Tensor, param: Tensor):
        '''
        Take an update step for a given parameter tensor
            - state: the optimizer state of the parameter (tensor)
            - group: stores optimizer attributes of the parameter group
            - grad: current gradient tensor g_t for theta at t-1
            - param: the parameter tensor theta at t-1
        '''
        grad = self.weight_decay(param, grad, group)
        m, v = self.calc_mv(state, group, grad)
        
        # increment t
        state['step'] += 1
        
        # perform adam update
        self.update_adam(state, group, param, m, v)