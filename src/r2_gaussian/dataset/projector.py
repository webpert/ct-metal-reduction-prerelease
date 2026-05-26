import sys
import torch
from torch import nn
import numpy as np

sys.path.append("./")
from r2_gaussian.utils.gaussian_utils import get_expon_lr_func
from r2_gaussian.utils.general_utils import t2a
from r2_gaussian.arguments import ModelParams, OptimizationParams
from xray_gaussian_rasterization_voxelization import getBhcEtaCount
import os.path as osp


def inverse_sigmoid(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = torch.clamp(y, eps, 1 - eps)
    return torch.log(y / (1 - y))


def inverse_tanh(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = torch.clamp(y, -1 + eps, 1 - eps)
    return 0.5 * torch.log((1 + y) / (1 - y))


def linear_constant_mixture(
    ratio: torch.Tensor,
    thres: torch.Tensor,
    axis: torch.Tensor,
    gamma: float = 0.8,
    k: float = 10.0,
):
    s0 = gamma * thres + 1.0
    a = (ratio - 1.0) / (s0 - (1.0 - gamma))
    linear_component = a * (axis - (1.0 - gamma)) + 1.0
    constant_component = a * (s0 - (1.0 - gamma)) + 1.0
    weight = torch.sigmoid(k * (axis - s0))
    y = (1.0 - weight) * linear_component + weight * constant_component
    y = y / y.sum()
    return y.unsqueeze(1)


class Projector(nn.Module):
    def __init__(
        self,
        args: ModelParams,
    ):
        super(Projector, self).__init__()

        self.optimizer = None
        self._bhc_eta_ratio = torch.empty(0)
        self._bhc_eta_thres = torch.empty(0)
        self.bhc_eta_count = self.get_bhc_eta_count
        self._bhc_gamma = torch.empty(0)

        self.bhc_eta_activation = linear_constant_mixture
        self.bhc_eta_ratio_activation = torch.nn.Sigmoid()
        self.bhc_eta_ratio_inverse_activation = inverse_sigmoid
        self.bhc_eta_thres_activation = torch.nn.Tanh()
        self.bhc_eta_thres_inverse_activation = inverse_tanh
        self.bhc_gamma_activation = torch.nn.Identity()
        self.iteration = 0

    @property
    def get_bhc_eta(self):
        bhc_gamma = self.get_bhc_gamma.item()
        axis = (
            torch.linspace(1 - bhc_gamma, 1 + bhc_gamma, self.bhc_eta_count)
            .cuda()
            .float()
        )
        return self.bhc_eta_activation(
            self.bhc_eta_ratio_activation(self._bhc_eta_ratio),
            self.bhc_eta_thres_activation(self._bhc_eta_thres),
            axis,
            bhc_gamma,
            k=10.0,
        )

    @property
    def get_bhc_gamma(self):
        return self.bhc_gamma_activation(self._bhc_gamma)

    @property
    def get_bhc_eta_count(self):
        return getBhcEtaCount()

    @property
    def get_optimal_weight_for_b(self):
        with torch.no_grad():
            thres = self.bhc_eta_thres_activation(self._bhc_eta_thres)
            gamma = self.get_bhc_gamma
            s_min = 1.0 - gamma
            s0 = gamma * thres + 1.0
            s_thres = (s0 - s_min) * 0.5 + s_min
            optimal_weight_tensor = (1 / s_thres) ** 3
            return optimal_weight_tensor.cpu().numpy()

    def training_setup(self, training_args: OptimizationParams):
        bhc_gamma_init = training_args.bhc_gamma_init_value
        self._bhc_gamma = nn.Parameter(
            torch.tensor([bhc_gamma_init]).float().cuda().requires_grad_(True)
        )

        ratio_init = torch.tensor([0.2]).float().cuda()
        self._bhc_eta_ratio = nn.Parameter(
            self.bhc_eta_ratio_inverse_activation(ratio_init).requires_grad_(True)
        )
        thres_init = torch.tensor([0.0]).float().cuda()
        self._bhc_eta_thres = nn.Parameter(
            self.bhc_eta_thres_inverse_activation(thres_init).requires_grad_(True)
        )

        l = [
            {
                "params": [self._bhc_gamma],
                "lr": training_args.bhc_gamma_lr_init,
                "name": "bhc_gamma",
            },
            {
                "params": [self._bhc_eta_ratio],
                "lr": training_args.bhc_eta_ratio_lr_init,
                "name": "bhc_eta_ratio",
            },
            {
                "params": [self._bhc_eta_thres],
                "lr": training_args.bhc_eta_thres_lr_init,
                "name": "bhc_eta_thres",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.bhc_gamma_scheduler_args = get_expon_lr_func(
            lr_init=training_args.bhc_gamma_lr_init,
            lr_final=training_args.bhc_gamma_lr_final,
            max_steps=training_args.bhc_gamma_lr_max_steps,
        )
        self.bhc_eta_ratio_scheduler_args = get_expon_lr_func(
            lr_init=training_args.bhc_eta_ratio_lr_init,
            lr_final=training_args.bhc_eta_ratio_lr_final,
            max_steps=training_args.bhc_eta_ratio_lr_max_steps,
        )
        self.bhc_eta_thres_scheduler_args = get_expon_lr_func(
            lr_init=training_args.bhc_eta_thres_lr_init,
            lr_final=training_args.bhc_eta_thres_lr_final,
            max_steps=training_args.bhc_eta_thres_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        self.iteration = iteration
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "bhc_eta_ratio":
                param_group["lr"] = self.bhc_eta_ratio_scheduler_args(iteration)
            if param_group["name"] == "bhc_eta_thres":
                param_group["lr"] = self.bhc_eta_thres_scheduler_args(iteration)
            if param_group["name"] == "bhc_gamma":
                param_group["lr"] = self.bhc_gamma_scheduler_args(iteration)

    def pause_learning_rate(self, iteration):
        self.iteration = iteration
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = 0.0

    def save(self, path):
        bhc_gamma = self.get_bhc_gamma.item()
        bhc_eta_count = self.get_bhc_eta_count
        bhc_axis = np.linspace(1 - bhc_gamma, 1 + bhc_gamma, bhc_eta_count)
        bhc_eta = t2a(self.get_bhc_eta.squeeze())
        eta_pred = np.stack([bhc_axis, bhc_eta], axis=0)
        # np.save(path, eta_pred)

        eta_ratio = self.bhc_eta_ratio_activation(self._bhc_eta_ratio)
        eta_thres = self.bhc_eta_thres_activation(self._bhc_eta_thres)
        # np.savez(
        #     osp.join(osp.dirname(path), "eta_ratio_thres.npz"),
        #     ratio=t2a(eta_ratio),
        #     thres=t2a(eta_thres),
        # )
