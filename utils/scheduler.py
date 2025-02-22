from torch.optim.lr_scheduler import _LRScheduler, StepLR

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6, warmup_iters=20):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.warmup_iters = warmup_iters
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [ max( base_lr * ( 1 + self.last_epoch/self.warmup_iters ), self.min_lr) 
                    for base_lr in self.base_lrs]
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]