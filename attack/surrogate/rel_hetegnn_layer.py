import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RelGraphEmbed(nn.Module):
    def __init__(
        self,
        ntype_num,
        embed_size,
    ):
        super(RelGraphEmbed, self).__init__()
        self.embed_size = embed_size
        self.dropout = nn.Dropout(0.0)

        # Create weight embeddings for each node for each relation.
        self.embeds = nn.ParameterDict()
        for ntype, num_nodes in ntype_num.items():
            embed = nn.Parameter(th.Tensor(num_nodes, self.embed_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain("relu"))
            self.embeds[ntype] = embed

    def forward(self):
        return self.embeds


class HeteroRelationalGraphConv(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        relation_names,
        activation=None,
    ):
        super(HeteroRelationalGraphConv, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.relation_names = relation_names
        self.activation = activation

        self.W = nn.ModuleDict(
            {str(rel): nn.Linear(in_size, out_size) for rel in relation_names}
        )

        self.dropout = nn.Dropout(0.0)

    def forward(self, A, inputs):
        hs = {}
        for rel in A:
            src_type, edge_type, dst_type = rel
            if dst_type not in hs:
                hs[dst_type] = th.zeros(inputs[dst_type].shape[0], self.out_size)
            hs[dst_type] = hs[dst_type] + (
                A[rel].T @ self.W[str(edge_type)](inputs[src_type])
            )
            if self.activation:
                hs[dst_type] = self.activation(hs[dst_type])
            hs[dst_type] = self.dropout(hs[dst_type])

        return hs


class EntityClassify(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        relation_names,
        embed_layer,
    ):
        super(EntityClassify, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.relation_names = relation_names
        self.relation_names.sort()
        self.embed_layer = embed_layer

        self.layers = nn.ModuleList()
        # Input to hidden.
        self.layers.append(
            HeteroRelationalGraphConv(
                self.in_size,
                self.in_size,
                self.relation_names,
                activation=F.relu,
            )
        )
        # Hidden to output.
        self.layers.append(
            HeteroRelationalGraphConv(
                self.in_size,
                self.out_size,
                self.relation_names,
            )
        )

    def forward(self, A):
        h = self.embed_layer()
        for layer in self.layers:
            h = layer(A, h)
        return h
