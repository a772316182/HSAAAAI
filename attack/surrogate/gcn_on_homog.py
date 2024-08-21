import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, device, in_size, out_size, hidden_size=16):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.device = device
        # self.W1 = nn.Linear(in_size, out_size)
        self.W1 = nn.Linear(in_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, out_size, bias=False)

    def forward(self, A, X):
        if A.sum() == 0:
            return torch.zeros([len(A), self.out_size], device=self.device)
        else:
            I = torch.eye(len(A), device=self.device)
            A_hat = A + I
            A_hat[torch.where(A_hat > 1)] = 1.0
            D_hat = torch.diag(A_hat.sum(1)) ** -0.5
            D_hat[torch.isinf(D_hat)] = 0.0
            A_norm = D_hat @ A_hat @ D_hat
            Z1 = A_norm @ self.W1(X)
            Z2 = F.relu(Z1)
            Z3 = A_norm @ self.W2(Z2)
            return Z3
            # return Z1


class GCN_without_norm(GCN):
    def __init__(self, device, in_size, out_size, hidden_size=16):
        super().__init__(device, in_size, out_size, hidden_size)

    def forward(self, A, X):
        Z1 = A @ self.W1(X)
        Z2 = F.relu(Z1)
        Z3 = A @ self.W2(Z2)
        return Z3
