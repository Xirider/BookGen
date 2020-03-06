# import torch
# import operator
# import functools
# from copy import copy
# from math import sqrt

# class AdaFactor(torch.optim.Optimizer):
#     def __init__(self, params, lr=None, beta1=0.9, beta2=0.999, eps1=1e-30, 
#                  eps2=1e-3, cliping_threshold=1,non_constant_decay = True,
#                  enable_factorization=True, ams_grad=True, weight_decay=0):
        
#         enable_momentum =  beta1 != 0
#         self.beta1_glob = copy(beta1)
#         self.beta2_glob = copy(beta2)
#         self.lr_glob = copy(lr)
        
#         beta1 = self.beta1_glob if hasattr(beta1,'__call__') else lambda x: self.beta1_glob
#         beta2 = self.beta2_glob if hasattr(beta2,'__call__') else lambda x: self.beta2_glob

#         if non_constant_decay:
#             ams_grad = False
#             if isinstance(self.beta1_glob,float):
#                 beta1 = lambda t: self.beta1_glob * (1 - self.beta1_glob ** (t-1)) / (1 - self.beta1_glob ** t)
#             if isinstance(self.beta2_glob,float):
#                 beta2 = lambda t: self.beta2_glob * (1 - self.beta2_glob ** (t-1)) / (1 - self.beta2_glob ** t)

#         relative_step_size  = True
        
#         if lr is None:
#             #default value from article
#             lr = lambda t: min(1e-2, 1 / sqrt(t))
            
#         if isinstance(self.lr_glob, float):
#             lr=lambda x: self.lr_glob
#             relative_step_size = False
  
                         
#         defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps1=eps1,
#                         eps2=eps2, cliping_threshold=cliping_threshold,
#                         weight_decay=weight_decay,ams_grad=ams_grad,
#                         enable_factorization=enable_factorization,
#                         enable_momentum=enable_momentum,relative_step_size=relative_step_size)
        
#         super(AdaFactor, self).__init__(params, defaults)

#     def __setstate__(self, state):
#         super(AdaFactor, self).__setstate__(state)       
     
#     def _experimental_reshape(self,shape):
#         temp_shape = shape[2:]
#         if len(temp_shape) == 1:
#             new_shape = (shape[0],shape[1]*shape[2])
#         else:
#             tmp_div = len(temp_shape) // 2 + len(temp_shape) % 2           
#             new_shape = (shape[0]*functools.reduce(operator.mul, temp_shape[tmp_div:],1),
#                          shape[1]*functools.reduce(operator.mul, temp_shape[:tmp_div],1))
#         return new_shape, copy(shape)
        
        
#     def _check_shape(self, shape):
#         '''
#         output1 - True - algorithm for matrix, False - vector;
#         output2 - need reshape
#         '''
#         if len(shape) > 2:
#             return True, True
#         elif len(shape) == 2:
#             return True, False
#         elif len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
#             return False, False
#         else:
#             return False, False
        
#     def _rms(self, x):
#         return sqrt(torch.mean(x.pow(2)))
    
    
    
#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             loss = closure()       
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 data_backup = p.data.clone().detach()

                    
#                 if grad.is_sparse:
#                     raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead') 
                    
#                 is_matrix, is_need_reshape = self._check_shape(grad.size())
#                 new_shape = p.data.size()
#                 if is_need_reshape and group['enable_factorization']:
#                     new_shape, old_shape =\
#                     self._experimental_reshape(p.data.size())
#                     grad = grad.view(new_shape)
               
#                 state = self.state[p]
#                 if len(state) == 0:
#                     state['step'] = 0
#                     if group['enable_momentum']:
#                         state['exp_avg'] = torch.zeros(new_shape, dtype=torch.float32, device=p.grad.device)
                                           
                       
#                     if is_matrix and group['enable_factorization']:
#                         state['exp_avg_sq_R'] = torch.zeros((1,new_shape[1]), dtype=torch.float32, device=p.grad.device)
#                         state['exp_avg_sq_C'] = torch.zeros((new_shape[0],1), dtype=torch.float32, device=p.grad.device)
#                     else:
#                         state['exp_avg_sq'] = torch.zeros(new_shape, dtype=torch.float32, device=p.grad.device)
#                     if group['ams_grad']:
#                         state['exp_avg_sq_hat'] = torch.zeros(new_shape, dtype=torch.float32, device=p.grad.device)
                    
                
#                 if group['enable_momentum']:
#                     exp_avg = state['exp_avg']
                    
#                 if is_matrix and group['enable_factorization']:
#                     exp_avg_sq_R = state['exp_avg_sq_R']
#                     exp_avg_sq_C = state['exp_avg_sq_C'] 
#                 else:
#                     exp_avg_sq = state['exp_avg_sq']
                
#                 if group['ams_grad']:
#                     exp_avg_sq_hat = state['exp_avg_sq_hat']
                
                
#                 state['step'] += 1
#                 lr_t = group['lr'](state['step'])
#                 if group['relative_step_size']:
#                     lr_t *= max(group['eps2'], self._rms(p.data))
                          
#                 if group['enable_momentum']:
#                     beta1_t = group['beta1'](state['step'])
#                     exp_avg.mul_(beta1_t).add_(1 - beta1_t, grad)
                    
#                 beta2_t = group['beta2'](state['step']) 

#                 if is_matrix and group['enable_factorization']:
#                     exp_avg_sq_R.mul_(beta2_t).add_(1 - beta2_t,                   
#                       torch.sum(torch.mul(grad,grad).add_(group['eps1']), dim=0, keepdim=True))
#                     exp_avg_sq_C.mul_(beta2_t).add_(1 - beta2_t,                   
#                       torch.sum(torch.mul(grad,grad).add_(group['eps1']), dim=1, keepdim=True))
#                     v = torch.mul(exp_avg_sq_C,exp_avg_sq_R).div_(torch.sum(exp_avg_sq_R))
#                 else:
#                     exp_avg_sq.mul_(beta2_t).addcmul_(1 - beta2_t, grad, grad).add_((1 - beta2_t)*group['eps1'])
#                     v = exp_avg_sq

                
#                 g = grad
#                 if group['enable_momentum']:
#                     g = torch.div(exp_avg,1 - beta1_t ** state['step'])
                               
#                 if group['ams_grad']:
#                     torch.max(exp_avg_sq_hat, v, out=exp_avg_sq_hat)
#                     v = exp_avg_sq_hat                    
#                     u = torch.div(g,(torch.div(v,1 - beta2_t ** state['step'])).sqrt().add_(group['eps1']))
#                 else:
#                     u = torch.div(g,v.sqrt()) 
       
#                 u.div_(max(1,self._rms(u) / group['cliping_threshold']))
#                 p.data.add_(-lr_t * (u.view(old_shape) if is_need_reshape and group['enable_factorization'] else u))
                
#                 if group['weight_decay'] != 0:
#                     p.data.add_(-group['weight_decay'] * lr_t, data_backup)
                    
#         return loss
import math
import torch
import torch.optim


class Adafactor(torch.optim.Optimizer):
    """Implements Adafactor algorithm.
    This implementation is based on:
    `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)
    Note that this optimizer internally adjusts the learning rate
    depending on the *scale_parameter*, *relative_step* and
    *warmup_init* options. To use a manual (external) learning rate
    schedule you should set `scale_parameter=False` and
    `relative_step=False`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constans for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3))
        clip_threshold (float): threshold of root mean square of
            final gradient update (default: 1.0)
        decay_rate (float): coefficient used to compute running averages of square
            gradient (default: -0.8)
        beta1 (float): coefficient used for computing running averages of gradient
            (default: None)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_parameter (bool): if True, learning rate is scaled by root mean square of
            parameter (default: True)
        relative_step (bool): if True, time-dependent learning rate is computed
            instead of external learning rate (default: True)
        warmup_init (bool): time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False)
    """

    def __init__(self, params, lr=None, eps=(1e-30, 1e-3), clip_threshold=1.0,
                 decay_rate=-0.8, beta1=None, weight_decay=0.0, scale_parameter=True,
                 relative_step=True, warmup_init=True):
        if lr is not None and relative_step:
            raise ValueError('Cannot combine manual lr and relative_step options')
        if warmup_init and not relative_step:
            raise ValueError('warmup_init requires relative_step=True')

        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold, decay_rate=decay_rate,
                        beta1=beta1, weight_decay=weight_decay, scale_parameter=scale_parameter,
                        relative_step=relative_step, warmup_init=warmup_init)
        super(Adafactor, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_lr(self, param_group, param_state):
        rel_step_sz = param_group['lr']
        if param_group['relative_step']:
            min_step = 1e-6 * param_state['step'] if param_group['warmup_init'] else 1e-2
            rel_step_sz = min(min_step, 1.0/math.sqrt(param_state['step']))
        param_scale = 1.0
        if param_group['scale_parameter']:
            param_scale = max(param_group['eps'][1], param_state['RMS'])
        return param_scale * rel_step_sz

    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col, output):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1).unsqueeze(-1)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=output)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adafactor does not support sparse gradients.')

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(grad)
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1]).type_as(grad)
                        state['exp_avg_sq_col'] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    state['RMS'] = 0
                else:
                    if use_first_moment:
                        state['exp_avg'] = state['exp_avg'].type_as(grad)
                    if factored:
                        state['exp_avg_sq_row'] = state['exp_avg_sq_row'].type_as(grad)
                        state['exp_avg_sq_col'] = state['exp_avg_sq_col'].type_as(grad)
                    else:
                        state['exp_avg_sq'] = state['exp_avg_sq'].type_as(grad)

                p_data_fp32 = p.data.float()

                state['step'] += 1
                state['RMS'] = self._rms(p_data_fp32)
                group['lr'] = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = (grad**2) + group['eps'][0]
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2t).add_(1.0 - beta2t, update.mean(dim=-1))
                    exp_avg_sq_col.mul_(beta2t).add_(1.0 - beta2t, update.mean(dim=-2))

                    # Approximation of exponential moving average of square of gradient
                    self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col, update)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg_sq.mul_(beta2t).add_(1.0 - beta2t, update)
                    torch.rsqrt(exp_avg_sq, out=update).mul_(grad)

                update.div_(max(1.0, self._rms(update) / group['clip_threshold']))
                update.mul_(group['lr'])

                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(1 - group['beta1'], update)
                    update = exp_avg

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                p_data_fp32.add_(-update)

                p.data.copy_(p_data_fp32)

        return loss