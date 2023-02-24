import torch
from torch import nn
from torch.nn import init


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        # set mu
        # self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))

        # self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots, dim))
        init.xavier_uniform_(self.slots_logsigma)

        ''' Query, Key, Value '''
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.to_q = nn.ModuleList()
        for i_s in range(self.num_slots):
            self.to_q.append(nn.Linear(dim, dim))
        self.to_k = nn.ModuleList()
        for i_s in range(self.num_slots):
            self.to_k.append(nn.Linear(dim, dim))
        self.to_v = nn.ModuleList()
        for i_s in range(self.num_slots):
            self.to_v.append(nn.Linear(dim, dim))

        # Set a GRU cell
        self.gru = nn.GRUCell(dim, dim)

        self.gru = nn.ModuleList()
        for i_s in range(self.num_slots):
            self.gru.append(nn.GRUCell(dim, dim))

        # set hidden dimension
        hidden_dim = max(dim, hidden_dim)

        # set mlp layers
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.mlp = nn.ModuleList()
        for i_s in range(self.num_slots):
            self.mlp.append(
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, dim)
                )
            )

        # input & slots & pre_ff normalization?
        # no communication among slots
        # self.norm_input = nn.LayerNorm(dim)
        # self.norm_slots = nn.LayerNorm(dim)
        # self.norm_pre_ff = nn.LayerNorm(dim)

        self.norm_input = nn.ModuleList()
        self.norm_slots = nn.ModuleList()
        self.norm_pre_ff = nn.ModuleList()
        for i_s in range(self.num_slots):
            self.norm_input.append(nn.LayerNorm(dim))
            self.norm_slots.append(nn.LayerNorm(dim))
            self.norm_pre_ff.append(nn.LayerNorm(dim))


    def apply_to_k_v(self, feats, mods):
        # feats: bz x N x dim
        feat_dim_len = len(feats.size())

        res = []

        for i_s, mod in enumerate(mods):
            if feat_dim_len == 3:
                cur_transformed_feat = mod(feats)
            else:
                cur_transformed_feat = mod(feats[:, i_s, :, :])
            res.append(cur_transformed_feat.unsqueeze(1))
        res = torch.cat(res, dim=1)
        return res

    def apply_to_q(self, slot_feats, mods):
        # slot_feats: bz x n_s x dim
        res = []
        for i_s, mod in enumerate(mods):
            cur_transformed_feat = mod(slot_feats[:, i_s, :])
            res.append(cur_transformed_feat.unsqueeze(1))
        res = torch.cat(res, dim=1)
        return res

    def apply_gru(self, prev_slots, updates):
        # prev_slots: bz x n_s x dim
        # updates: bz x n_s x dim
        cur_slots = []
        for i_s, cur_slot_gru in enumerate(self.gru):
            cur_slot = cur_slot_gru(updates[:, i_s, :], prev_slots[:, i_s, :])
            cur_slots.append(cur_slot.unsqueeze(1))
        cur_slots = torch.cat(cur_slots, dim=1)
        return cur_slots

    def apply_layer_norm_feats(self, feats, norm_layers):
        res = []
        for i_s, mod in enumerate(norm_layers):
            if len(feats.size()) == 4:
                normed_feats = mod(feats[:, i_s, :, :])
            else:
                normed_feats = mod(feats)
            res.append(normed_feats.unsqueeze(1))
        res = torch.cat(res, dim=1)
        return res

    def apply_layer_norm_slot_feats(self, slot_feats, norm_layers):
        res =[]
        for i_s, mod in enumerate(norm_layers):
            normed_slot_feats = mod(slot_feats[:, i_s, :])
            res.append(normed_slot_feats.unsqueeze(1))
        res = torch.cat(res, dim=1)
        return res

    def forward(self, inputs, num_slots=None):  # can be generalize to other number of slots

        assert num_slots is None

        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots

        # mu = self.slots_mu.expand(b, n_s, -1)
        # sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        # slots: bz x n_s x dim
        slots = mu + sigma * torch.randn(mu.shape, device=device)

        # inputs = self.norm_input(inputs)
        inputs = self.apply_layer_norm_feats(inputs, self.norm_input)
        # inputs: bz x N x dim
        # k, v = self.to_k(inputs), self.to_v(inputs)

        k = self.apply_to_k_v(inputs, self.to_k)
        v = self.apply_to_k_v(inputs, self.to_v)

        for _ in range(self.iters):
            slots_prev = slots

            # slots = self.norm_slots(slots)
            slots = self.apply_layer_norm_slot_feats(slots, self.norm_slots)
            # q = self.to_q(slots)

            q = self.apply_to_q(slots, self.to_q)

            # q: bz x num_slots x dim; k: bz x N x dim; dots: bz x num_slots x N
            # q: bz x n_s x dim; k: bz x N x dim; -> dots: bz x n_s x N
            dots = torch.einsum('bid,bijd->bij', q, k) * self.scale
            # attn: bz x n_s x N --- why applying softmax on the points dimension?
            attn_ori = dots.softmax(dim=1) + self.eps
            # hard_attn_ori = (3 * dots).softmax(dim=1) + self.eps
            # attn: bz x N x num_slots
            attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)


            updates = torch.einsum('bijd,bij->bid', v, attn)

            ''' Update for current iteration'''
            # slots = self.gru(
            #     updates.reshape(-1, d),
            #     slots_prev.reshape(-1, d)
            # )

            slots = self.apply_gru(
                slots_prev, updates
            )

            # get slots' representations --- slots and others; for clustering...; how many clusters wished to cluster to
            # slots = slots.reshape(b, -1, d)
            # slots = slots + self.mlp(self.norm_pre_ff(slots))
            # slots = slots + self.apply_to_q(self.norm_pre_ff(slots), self.mlp)
            slots = slots + self.apply_to_q(self.apply_layer_norm_slot_feats(slots, self.norm_pre_ff), self.mlp)
        # bz x n_slots x dim
        return slots, attn_ori
        # return slots, hard_attn_ori
