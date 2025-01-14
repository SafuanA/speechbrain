import torch
import torch.nn as nn

class Noam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        super(MomentumOptimizer, self).__init__(params, defaults={'lr': lr})
        self.momentum = momentum
        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(mom=torch.zeros_like(p.data))
      
    # Step Method
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    self.state[p] = dict(mom=torch.zeros_like(p.data))
                mom = self.state[p]['mom']
                mom = self.momentum * mom - group['lr'] * p.grad.data
                p.data += mom