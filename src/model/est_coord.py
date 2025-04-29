from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..config import Config
from ..vis import Vis


class EstCoordNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config
        # raise NotImplementedError("You need to implement some modules here")
        self.conv1 = nn.Sequential(         # input: (B, 3, N)
            nn.Conv1d(3, 64, 1),            # -> (B, 64, N)
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),          # -> (B, 128, N)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),        # -> (B, 1024, N)
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(64+1024, 512, 1),     # -> (B, 512, N)
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 256, 1),         # -> (B, 256, N)
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 128, 1),         # -> (B, 128, N)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Conv1d(128, 3, 1)   # -> (B, 3, N)

    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape \(B, N, 3\)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        # raise NotImplementedError("You need to implement the forward function")
        B, N, _ = pc.shape

        x = pc.permute(0, 2, 1)     # -> (B, 3, N)
        x1 = self.conv1(x)          # -> (B, 64, N)
        x2 = self.conv2(x1)         # -> (B, 128, N)
        x3 = self.conv3(x2)         # -> (B, 1024, N)

        g = torch.max(x3, dim=2, keepdim=True).values   # -> (B, 1024, 1)
        h = torch.cat([x1, g.expand(-1, -1, N)], dim=1) # -> (B, 1088, N)
        h = self.conv4(h)         # -> (B, 512, N)
        h = self.conv5(h)         # -> (B, 256, N)
        h = self.conv6(h)         # -> (B, 128, N)
        h = self.conv7(h)         # -> (B, 3, N)
        coord_pred = h.permute(0, 2, 1)     # -> (B, N, 3)

        loss = F.mse_loss(coord_pred, coord)

        metric = dict(
            loss=loss,
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

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        # raise NotImplementedError("You need to implement the est function")
        B, N, _ = pc.shape

        x = pc.permute(0, 2, 1)     # -> (B, 3, N)
        x1 = self.conv1(x)          # -> (B, 64, N)
        x2 = self.conv2(x1)         # -> (B, 128, N)
        x3 = self.conv3(x2)         # -> (B, 1024, N)

        g = torch.max(x3, dim=2, keepdim=True).values   # -> (B, 1024, 1)
        h = torch.cat([x1, g.expand(-1, -1, N)], dim=1) # -> (B, 1088, N)
        h = self.conv4(h)           # -> (B, 512, N)
        h = self.conv5(h)           # -> (B, 256, N)
        h = self.conv6(h)           # -> (B, 128, N)
        h = self.conv7(h)           # -> (B, 3, N)
        pc_obj = h.permute(0, 2, 1)     # -> (B, N, 3)

        def solve_procrustes_single(P, Q):
            """
            P: (N, 3)
            Q: (N, 3)
            """
            P_centred = P - P.mean(dim=0, keepdim=True)
            Q_centred = Q - Q.mean(dim=0, keepdim=True)

            U, S, Vt = torch.linalg.svd(P_centred.t().matmul(Q_centred), full_matrices=False)
            UVt = U.matmul(Vt)
            det_UVt = torch.linalg.det(UVt)
            D = torch.diag_embed(torch.stack([
                torch.ones_like(det_UVt),
                torch.ones_like(det_UVt),
                det_UVt
            ], dim=-1))
            R_pred = U.matmul(D).matmul(Vt)
            t_pred = (P.t() - R_pred.matmul(Q.t())).mean(dim=1)

            return t_pred, R_pred
        
        # RANSAC
        max_iters = 1000
        res_thresh = 0.01
        end_thresh = 0.8

        t_pred, R_pred = [], []
        for b in range(B):
            best_inliers = torch.zeros(N, device=pc.device)
            best_inlier_count = 0
            best_model = (
                torch.zeros(3, device=pc.device),
                torch.zeros(3, 3, device=pc.device)
            )

            for _ in range(max_iters):
                idx = torch.randint(0, N, (3,), device=pc.device)
                t_i, R_i = solve_procrustes_single(pc[b, idx], pc_obj[b, idx])

                transformed = R_i.matmul(pc_obj[b].t()).t() + t_i
                residuals = torch.norm(transformed - pc[b], dim=1)

                inliers = residuals < res_thresh
                inlier_count = inliers.sum(dim=0)

                if inlier_count > best_inlier_count:
                    best_inliers = inliers
                    best_inlier_count = inlier_count
                    best_model = (t_i, R_i)
                
                if best_inlier_count >= end_thresh * N:
                    break
            
            if best_inlier_count >= 3:
                t_final, R_final = solve_procrustes_single(pc[b, best_inliers], pc_obj[b, best_inliers])
                t_pred.append(t_final)
                R_pred.append(R_final)
            else:
                t_pred.append(best_model[0])
                R_pred.append(best_model[1])
        
        t_pred = torch.stack(t_pred, dim=0)
        R_pred = torch.stack(R_pred, dim=0)
        return t_pred, R_pred
