import math
import torch
from torch.optim.optimizer import Optimizer
import numpy as np


class AdamW(Optimizer):
    """Implements AdamW algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay using the method from
            the paper `Fixing Weight Decay Regularization in Adam` (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Fixing Weight Decay Regularization in Adam:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

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
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'], p.data)

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


# https://github.com/robintibor/adamw-eeg-eval/blob/master/adamweegeval/schedulers.py
# https://github.com/pytorch/pytorch/pull/1370/files
# Schedule weight decay should be enabled for AdamW
class CosineRestartsScheduler:
    def __init__(self, optimizer, first_decay_steps, t_mul=2.0, m_mul=1.0, alpha=0.0):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            group.setdefault('initial_weight_decay', group['weight_decay'])

    # Starting from epoch 0
    def step(self, global_step, decay_wd, normalize_wd):
        decay, wd_normalization = self.get_decay(global_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * decay
            if decay_wd:
                wd_decay = wd_normalization * decay if normalize_wd else decay
                param_group['weight_decay'] = param_group['initial_weight_decay'] * wd_decay

    def get_decay(self, global_step):
        completed_fraction = global_step / self.first_decay_steps
        if self.t_mul == 1.0:
            i_restart = np.floor(completed_fraction)
            completed_fraction = completed_fraction - i_restart
            num_of_updates = self.first_decay_steps
        else:
            i_restart = np.floor(np.log(1.0 - completed_fraction * (
                    1.0 - self.t_mul)) / np.log(self.t_mul))
            sum_r = (1.0 - self.t_mul ** i_restart) / (1.0 - self.t_mul)
            completed_fraction = (completed_fraction - sum_r) / self.t_mul ** i_restart
            num_of_updates = self.first_decay_steps * (self.t_mul ** i_restart)

        m_fac = self.m_mul ** i_restart
        cosine_decay = 0.5 * m_fac * (1.0 + np.cos(np.pi * completed_fraction))
        decay = (1 - self.alpha) * cosine_decay + self.alpha
        return decay, np.sqrt(1.0/num_of_updates)


class ScheduledOptimizer(object):
    def __init__(self, scheduler, optimizer, decay_wd, normalize_wd):
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.decay_wd = decay_wd
        self.normalize_wd = normalize_wd

        self.step_count = 0

    def step(self):
        self.scheduler.step(self.step_count, self.decay_wd, self.normalize_wd)

        self.optimizer.step()
        self.step_count += 1

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self.optimizer.zero_grad()


if __name__ == '__main__':
    # Plot decays from CosineRestartsScheduler
    v = torch.autograd.Variable(torch.randn(10, 10).type(torch.FloatTensor), requires_grad = True)
    optimizer = Optimizer([{'params': v,
                            'lr': 1.0,
                            'weight_decay': 10}], {})

    scheduler = CosineRestartsScheduler(optimizer, first_decay_steps=1000, alpha=0.0, m_mul=0.9)

    lr = [scheduler.get_decay(i)[0] for i in range(10000)]
    norm = [scheduler.get_decay(i)[1] for i in range(10000)]

    norm_decayed = [l * n for (l, n) in zip(lr, norm)]

    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(lr, color='red')
    axes[0].set_title('Learning Rate')
    axes[1].plot(norm, color='blue')
    axes[1].set_title('Weight decay normalization')
    axes[2].plot(norm_decayed, color='green')
    axes[2].set_title('Weight decay normalized')
    plt.show()
