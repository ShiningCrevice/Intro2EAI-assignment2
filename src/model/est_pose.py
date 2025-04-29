from typing import Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F
import math

from ..config import Config


class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config
        # raise NotImplementedError("You need to implement some modules here")
        self.shared_mlp = nn.Sequential(        # input: (B, 3, N)
            nn.Conv1d(3, 64, 1, bias=False),    # -> (B, 64, N)
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, 1, bias=False),  # -> (B, 128, N)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, 1, bias=False), # -> (B, 256, N)
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)     # -> (B, 256)
        # for translation vector
        self.head_t = nn.Sequential(
            nn.Linear(256, 128, bias=False),    # -> (B, 128)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),                  # -> (B, 3)
        )
        # for rotation matrix
        self.head_r = nn.Sequential(
            nn.Linear(256, 128, bias=False),    # -> (B, 128)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 9),                  # -> (B, 9)
        )

    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape \(B, 3\)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape \(B, 3, 3\)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        # raise NotImplementedError("You need to implement the forward function")
        x = pc.permute(0, 2, 1)         # -> (B, 3, N)
        f = self.shared_mlp(x)          # -> (B, 256, N)
        f = self.pool(f).squeeze(-1)    # -> (B, 256)

        t_pred = self.head_t(f)         # -> (B, 3)
        r_pred = self.head_r(f)         # -> (B, 9)

        U, S, Vt = torch.linalg.svd(r_pred.view(-1, 3, 3), full_matrices=False)
        UVt = U.matmul(Vt)
        det_UVt = torch.linalg.det(UVt)
        D = torch.diag_embed(torch.stack([
            torch.ones_like(det_UVt),
            torch.ones_like(det_UVt),
            det_UVt
        ], dim=-1))
        R_pred = U.matmul(D).matmul(Vt)

        loss_t = F.mse_loss(t_pred, trans)
        loss_r = F.mse_loss(R_pred, rot)
        loss =  self.config.lambda_t * loss_t + self.config.lambda_r * loss_r

        metric = dict(
            loss=loss,
            loss_t=loss_t,
            loss_r=loss_r,
        )

        return loss, metric

    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape \(B, 3\)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape \(B, 3, 3\)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.
        """
        # raise NotImplementedError("You need to implement the est function")
        self.eval()
        
        x = pc.permute(0, 2, 1)         # -> (B, 3, N)
        f = self.shared_mlp(x)          # -> (B, 256, N)
        f = self.pool(f).squeeze(-1)    # -> (B, 256)

        t_pred = self.head_t(f)         # -> (B, 3)
        r_pred = self.head_r(f)         # -> (B, 9)

        U, S, Vt = torch.linalg.svd(r_pred.view(-1, 3, 3), full_matrices=False)
        UVt = U.matmul(Vt)
        det_UVt = torch.linalg.det(UVt)
        D = torch.diag_embed(torch.stack([
            torch.ones_like(det_UVt),
            torch.ones_like(det_UVt),
            det_UVt
        ], dim=-1))
        R_pred = U.matmul(D).matmul(Vt)

        return t_pred, R_pred
