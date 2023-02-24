import torch
from torch import nn
from torch.nn import init


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, na, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.na = na

        # different slots use the same mu
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # Set a GRU cell
        self.gru = nn.GRUCell(dim, dim)

        # set hidden dimension
        hidden_dim = max(dim, hidden_dim)

        # set mlp layers
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        # input & slots & pre_ff normalization?
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, n, na, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, self.na, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, self.na, -1)

        # slots: bz x n_s x dim
        slots = mu + sigma * torch.randn(mu.shape, device=device)

        # todo: the influence of LayerNorm?
        inputs = self.norm_input(inputs)
        # inputs: bz x N x dim
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            # q: bz x n_s x na x dim
            # k: bz x N x na x dim
            q = self.to_q(slots)

            # q: bz x num_slots x dim; k: bz x N x dim; dots: bz x num_slots x N
            # q: bz x n_s x dim; k: bz x N x dim; -> dots: bz x n_s x N
            # dots: bz x n_s x N x na
            dots = torch.einsum('biad,bjad->bija', q, k) * self.scale
            # attn: bz x n_s x N  x na--- why applying softmax on the points dimension?
            attn_ori = dots.softmax(dim=1) + self.eps
            # attn: bz x n_s x N x na
            attn = attn_ori / attn_ori.sum(dim=-2, keepdim=True)

            # v: bz x N x na x dim
            # attn: bz x n_s x N x na
            # updates: bz x n_s x na x dim
            updates = torch.einsum('bjad,bija->biad', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            # get slots' representations --- slots and others; for clustering...; how many clusters wished to cluster to
            slots = slots.reshape(b, -1, self.na, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        # bz x n_slots x dim
        return slots, attn_ori
