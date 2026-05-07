from itertools import count
from math import pow
import math
from typing import Callable, Optional, Tuple
from contextlib import contextmanager

from click import group
import torch
import torch.optim
import torch.distributed as dist
from torch import Tensor


ClosureType = Callable[[], Tensor]


def _welford_mean(avg: Optional[Tensor], newval: Tensor, count: int) -> Tensor:
    return newval if avg is None else avg + (newval - avg) / count


class CoVON_wprior(torch.optim.Optimizer):
    # for CL: new codes to extend IVON for CL

    hessian_approx_methods = (
        'bonnet',  # Bonnet's theorem
        'gradsq',  # Gradient magnitude approx.
        'exact',   # Exact using backpack
    )

    def __init__(
        self,
        params,
        lr: float,
        ess: float,
        prior_mean=None,
        prior_prec=None,
        hess_init: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        weight_decay: float = 1e-4,
        mc_samples: int = 1,
        hess_approx: str = 'bonnet',
        clip_radius: float = float("inf"),
        sync: bool = False,
        debias: bool = True,
        differentiable: bool = False,
        alpha: float = 1,
        gamma_s: float = 1.0,
        gamma_m: float = 1.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 1 <= mc_samples:
            raise ValueError(f"Invalid number of MC samples: {mc_samples}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if not 0.0 < ess:
            raise ValueError(f"Invalid effective sample size: {ess}")
        if not 0.0 < clip_radius:
            raise ValueError(f"Invalid clipping radius: {clip_radius}")
        if not 0.0 <= beta1 <= 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 <= 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if hess_approx not in self.hessian_approx_methods:
            raise ValueError(f"Invalid hess_approx parameter: {hess_approx}")

        defaults = dict(
            lr=lr,
            mc_samples=mc_samples,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            hess_init=hess_init,
            ess=ess,
            clip_radius=clip_radius,
        )
        super().__init__(params, defaults)

        self.mc_samples = mc_samples
        self.hess_approx = hess_approx
        self.sync = sync
        self._numel, self._device, self._dtype = self._get_param_configs()
        self.current_step = 0
        self.debias = debias
        self.weight_decay = weight_decay
        self.differentiable = differentiable
        self.alpha = alpha
        self.gamma_s = gamma_s
        self.gamma_m = gamma_m
        # set initial temporary running averages
        self._reset_samples()
        # init all states
        self._init_buffers()
        # for CL
        # install prior
        self._install_prior(prior_mean, prior_prec)

    # for CL
    def _install_prior(self, prior_mean, prior_prec):
        """Install prior with ALREADY scaled precision"""
        for j, group in enumerate(self.param_groups):
            group["prior_mean"] = []
            group["weight_decay"] = []
            group["prior_prec"] = []

            for i, p in enumerate(group["params"]):
                if p is None:
                    continue

                if prior_mean is not None and prior_prec is not None:
                    group["prior_mean"].append(prior_mean[j][i])
                    group["prior_prec"].append(prior_prec[j][i])

                    if j == 1:  # No decay group
                        group["weight_decay"].append(0.0)
                    else:
                        group["weight_decay"].append(prior_prec[j][i] / group['ess'])
                else:
                    group["prior_mean"].append(None)
                    group["prior_prec"].append(None)
                    if j == 1:
                        group["weight_decay"].append(0.0)
                    else:
                        group["weight_decay"].append(self.weight_decay)


    def _get_param_configs(self):
        all_params = []
        for pg in self.param_groups:
            pg["numel"] = sum(p.numel() for p in pg["params"] if p is not None)
            all_params += [p for p in pg["params"] if p is not None]
        if len(all_params) == 0:
            return 0, torch.device("cpu"), torch.get_default_dtype()
        devices = {p.device for p in all_params}
        if len(devices) > 1:
            raise ValueError(f"Parameters are on different devices: {[str(d) for d in devices]}")
        device = next(iter(devices))
        dtypes = {p.dtype for p in all_params}
        if len(dtypes) > 1:
            raise ValueError(f"Parameters are on different dtypes: {[str(d) for d in dtypes]}")
        dtype = next(iter(dtypes))
        total = sum(pg["numel"] for pg in self.param_groups)
        return total, device, dtype

    def _reset_samples(self):
        self.state['count'] = 0
        for group in self.param_groups:
            group["avg_grad"] = [None] * len(group["params"])
            group["avg_lhess"] = [None] * len(group["params"])

    def _init_buffers(self):
        for group in self.param_groups:
            hess_init = group["hess_init"]
            group["momentum"] = [torch.zeros_like(p) for p in group["params"]]
            group["hess"] = [torch.zeros_like(p).add(hess_init) for p in group["params"]]

    @contextmanager
    def sampled_params(self, train: bool = False):
        param_avg, noise = self._sample_params()
        yield
        self._restore_param_average(train, param_avg, noise)

    def _restore_param_average(
        self, train: bool, param_avg: list[Tensor], noise: list[Tensor]
    ):
        if train:
            self.state["count"] = self.state.get("count", 0) + 1
            count = self.state["count"]

        # param_avg / noise are flat lists; avg_grad / avg_lhess live in each group
        param_idx = 0
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p is None:
                    continue

                p.data = param_avg[param_idx].clone()

                if train:
                    grad_sample = p.grad if (p.requires_grad and p.grad is not None) else torch.zeros_like(p)

                    group["avg_grad"][i] = _welford_mean(
                        group["avg_grad"][i], grad_sample, count
                    )

                    if self.hess_approx == 'bonnet':
                        lhess_sample = noise[param_idx] * grad_sample
                    elif self.hess_approx == 'gradsq':
                        lhess_sample = grad_sample ** 2
                    elif self.hess_approx == 'exact':
                        lhess_sample = getattr(p, 'diag_h', torch.zeros_like(p))
                    else:
                        raise NotImplementedError(f"Unknown hessian approx: {self.hess_approx}")

                    group["avg_lhess"][i] = _welford_mean(
                        group["avg_lhess"][i], lhess_sample, count
                    )

                param_idx += 1


    @torch.no_grad()
    def step(self, closure: ClosureType = None) -> Optional[Tensor]:
        if closure is None:
            loss = None
        else:
            losses = []
            for _ in range(self.mc_samples):
                with torch.enable_grad():
                    loss = closure()
                losses.append(loss)
            loss = sum(losses) / self.mc_samples
        if self.sync and dist.is_initialized():
            self._sync_samples()
        self._update()
        self._reset_samples()
        return loss

    def _sync_samples(self):
        world_size = dist.get_world_size()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p is None:
                    continue

                dist.all_reduce(group["avg_grad"][i])
                group["avg_grad"][i].div_(world_size)
                dist.all_reduce(group["avg_lhess"][i])
                group["avg_lhess"][i].div_(world_size)


    def _sample_params(self):
        param_avgs = []
        noise_samples = []

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p is None:
                    continue

                h_i = group["hess"][i]

                noise = torch.randn_like(p) / (group["ess"] * (group["hess"][i] + group["weight_decay"][i])).sqrt()

                param_avgs.append(p.detach().clone())
                noise_samples.append(noise)

                p.data.add_(noise)

        return param_avgs, noise_samples

    def _update(self):
        self.current_step += 1
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            clip_radius = group["clip_radius"]
            ess = group["ess"]
            debias = 1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0

            for i, p in enumerate(group["params"]):
                if p is None:
                    continue

                grad_avg = group["avg_grad"][i]
                group["momentum"][i] = b1 * group["momentum"][i] + (1 - b1) * grad_avg

                hess = group["hess"][i]
                wd = group["weight_decay"][i]
                lhess_avg = group["avg_lhess"][i]

                group["hess"][i] = self._new_hess('bonnet', hess, lhess_avg, ess, b2, wd)

                prior_mean = group["prior_mean"][i]
                p.data = self._new_param_averages(
                    p.data,
                    prior_mean,
                    group["hess"][i],
                    group["momentum"][i],
                    lr,
                    wd,
                    clip_radius,
                    debias,
                    group["hess_init"],
                    self.alpha,
                    group["ess"],
                )

    @staticmethod
    def _get_nll_hess(method: str, hess, avg_lhess, ess=None, wd=None) -> Tensor:
        if method == 'bonnet':
            return avg_lhess * hess
        elif method == 'gradsq':
            return avg_lhess
        elif method == 'exact':
            return avg_lhess
        else:
            raise NotImplementedError(f'unknown hessian approx.: {method}')

    @staticmethod
    def _new_hess(method, hess, avg_lhess, ess, beta2, wd):
        f = CoVON_wprior._get_nll_hess(method, hess + wd, avg_lhess)
        if method == 'bonnet':
            f = f * ess
        return beta2 * hess + (1 - beta2) * f + (0.5 * (1 - beta2) ** 2) * (hess - f).square() / (hess + wd)

    @staticmethod
    def _new_param_averages(param, prior_mean, hess, momentum, lr, wd, clip_radius, debias, hess_init, alpha, ess):
        if prior_mean is None:
            return param - lr * torch.clip(
                (momentum/debias +   wd * param) / (hess + wd),
                min=-clip_radius, max=clip_radius
            )
        else:
            return param - lr * torch.clip(
                ((momentum/debias ) +  (param - prior_mean))/ (hess + wd),
                min=-clip_radius, max=clip_radius
            )

    def get_prior(self):
        """Get prior BEFORE resetting buffers"""
        mean = []
        precision = []
        for num_, group in enumerate(self.param_groups):
            mean.append([])
            precision.append([])
            ess = group["ess"]
            prior_means = group.get("prior_mean", [None] * len(group["params"]))
            prior_prec = group.get("prior_prec", [None] * len(group["params"]))

            for p, h, wd, prior_m, prec_p in zip(group["params"], group["hess"], group["weight_decay"], prior_means, prior_prec):
                if p is None:
                    continue
                m_current = p.data.clone()
                s_t = prec_p if prec_p is not None else 0

                s_next = s_t + self.gamma_s * ess * h
                if prior_m is not None and self.gamma_m != 1.0:
                    numerator = (1 - self.gamma_m) * s_t * prior_m + self.gamma_m * (s_t + ess * h) * m_current
                    m_next = numerator / s_next
                else:
                    m_next = m_current

                mean[num_].append(m_next)
                precision[num_].append(s_next.clone())
        return mean, precision
    
    def get_naive_prior(self):
        """Get naive prior (current param values, unscaled precision)"""
        mean = []
        precision = []
        for num_, group in enumerate(self.param_groups):
            mean.append([])
            precision.append([])
            ess = group["ess"]
            for p, h, wd in zip(group["params"], group["hess"], group["weight_decay"]):
                if p is None:
                    continue

                s_next =  ess * (h + wd)
                mean[num_].append(p.data.clone())
                precision[num_].append(s_next.clone())
        return mean, precision

    def set_for_new_task(self, ess, hess_init, model=None, reset_step=True):
        """Reset optimizer for new task with prior from current state"""
        # Step 1: Get prior BEFORE resetting anything
        prior_mean, prior_precision = self.get_prior()

        # Step 2: Update ESS in param groups
        if reset_step:
            for num_gr in range(len(self.param_groups)):
                self.param_groups[num_gr]["ess"] = ess
                self.param_groups[num_gr]["hess_init"] = hess_init

        # Step 3: Reset optimizer state (skip step counter for within-task updates)
        if reset_step:
            self.current_step = 0
        self._reset_samples()

        # Step 4: Initialize buffers (resets Hessian to hess_init)
        self._init_buffers()

        # Step 5: Install prior with corrected precision
        self._install_prior(prior_mean, prior_precision)
