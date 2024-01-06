"""
References :- 

1. Momentum :- https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
2. RMSprop :- https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
3. Adam :- https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
4. https://arena3-chapter0-fundamentals.streamlit.app/

I have generally referred the pytorch psuedocode to get a better understanding of the optimizers.  
"""

import torch
import torch.optim as optim

class OptimizerBase:
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0,
        alpha=0,
        weight_decay=0,
        eps=1e-8,
    ):
        params = list(params)

        self.params = params
        self.lr = lr
        self.mu = momentum
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.steps = 0
        self.eps = eps

        self.bt_all = [torch.zeros_like(param) for param in self.params]
        self.vt_all = [torch.zeros_like(param) for param in self.params]

    def zero_grad(self):
        # zero out the gradient after one backward pass
        for param in self.params:
            param.grad = None
    
class SGD(OptimizerBase):
    """
    This is the simple SGD with momentum and weight decay. The momentum component allows us to reduce the oscillations
    and hence we are able to use a much larger learning rate.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @torch.inference_mode()
    def step(self):
        # take the optmization step given the gradients are computed.
        for i, (param, bt) in enumerate(zip(self.params, self.bt_all)):
            gt = param.grad
            
            # apply weight decay
            if self.weight_decay != 0:
                gt += self.weight_decay * param
            
            # apply momentum
            if self.mu != 0:
                if self.steps > 0:
                    new_bt = self.mu * bt + gt
                else:
                    new_bt = gt
            
                # this is the update now
                gt = new_bt
                self.bt_all[i] = new_bt
            
            self.params[i] -= self.lr * gt

        self.steps += 1

class RMSprop(OptimizerBase):
    """
    RMSprop allows us to have a adaptive learning rate (diff for each parameter)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.inference_mode()
    def step(self):
        
        # vt is the square gradient term
        for i, (param, vt, bt) in enumerate(zip(self.params, self.vt_all, self.bt_all)):
            gt = param.grad

            if self.weight_decay != 0:
                gt += self.weight_decay * param
            
            vt += (1 - self.alpha) * gt.pow(2)

            if self.mu:
                if self.steps:
                    new_bt = self.mu * bt + gt / (vt.sqrt() + self.eps)
                else:
                    # first step has no momentum build up
                    new_bt = gt / (vt.sqrt() + self.eps)
                
                self.params[i] -= self.lr * new_bt
                self.bt_all[i] = new_bt

            else:
                # no momentum case is simple
                self.params[i] -= self.lr * gt / (vt.sqrt() + self.eps)
            
            self.vt_all[i] = vt
            self.steps += 1


class Adam(OptimizerBase):
    def __init__(self, *args, beta1=0.9, beta2=0.999, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
    
    @torch.inference_mode()
    def step(self):
        
        # mt is the first moment and vt is the second moment
        for i, (param, mt, vt) in enumerate(zip(self.params, self.bt_all, self.vt_all)):

            gt = param.grad

            if self.weight_decay:
                gt += self.weight_decay * param
            
            new_mt = self.beta1 * mt + (1 - self.beta1) * gt
            new_vt = self.beta2 * vt + (1 - self.beta2) * gt.pow(2)

            mt_bar = new_mt / ( 1 - self.beta1 ** (self.steps + 1) )
            vt_bar = new_vt / ( 1 - self.beta2 ** (self.steps + 1) )

            self.params[i] -= self.lr * mt_bar / (vt_bar.sqrt() + self.eps)
            self.bt_all[i] = new_mt
            self.vt_all[i] = new_vt
        
        self.steps += 1