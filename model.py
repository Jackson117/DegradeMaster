import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_scatter import scatter_mean, scatter_softmax, scatter_add

from torch_geometric.nn import GCNConv, global_max_pool, SAGEConv, MessagePassing, global_mean_pool, GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import numpy as np

class GraphConv(nn.Module):
    def __init__(self, num_embeddings, graph_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim = hidden_size)
        self.gcn1 = GCNConv(hidden_size, int(hidden_size/2))
        self.gcn2 = GCNConv(int(hidden_size/2), hidden_size)
        self.graph_fc = nn.Linear(graph_dim, hidden_size)
        self.att_fc = nn.Linear(hidden_size*2, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, data, graph_fea):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr.to(torch.float)
        x = self.embed(x)
        x = self.gcn1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.gcn2(x, edge_index, edge_attr)

        m = self.graph_fc(graph_fea)[batch]
        att_logits = self.att_fc(torch.tanh(torch.concat([x, m], dim=-1)))
        att_weights = scatter_softmax(att_logits, batch, dim=0)

        x_wei =0.2 * x * att_weights + 0.8 * m
        graph_embd = scatter_add(x_wei, batch, dim=0)

        # x = global_mean_pool(x, batch)
        graph_embd = self.mlp(graph_embd)
        return graph_embd

class GATTConv(nn.Module):
    def __init__(self, num_embeddings, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim=hidden_size)
        self.gat1 = GCNConv(hidden_size, hidden_size*2)
        self.gat2 = GCNConv(hidden_size*2, hidden_size)

    def forward(self, data, graph_fea):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr.to(torch.float)
        x = self.embed(x)
        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return x

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return Parameter(initial)

def glorot_init_2(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return initial

class SageConv(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim = 64)
        self.sage1 = SAGEConv(64, 128)
        self.sage2 = SAGEConv(128, 64)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr.to(torch.float)
        x = self.embed(x)
        x = self.sage1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.sage2(x, edge_index, edge_attr)
        x = global_max_pool(x, batch)
        return x

class SmilesNet(nn.Module):
    def __init__(self, batch_size, hidden_size):
        super().__init__()
        self.batch_size = batch_size
        self.embed = nn.Embedding(41, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x, s):
        x = self.embed(x)
        x = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(x, None)
        out, _ = pad_packed_sequence(out, batch_first=True)
        y = self.fc(out[:,-1,:])
        return y

class EGNNConv(nn.Module):
    def __init__(self, num_embeddings, in_node_nf, in_edge_nf, graph_nf, hidden_nf, act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=0):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.graph_nf = graph_nf

        # Encoder
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=hidden_nf)
        self.node_attr = node_attr
        self.attention = attention

        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.graph_fc = nn.Linear(graph_nf, hidden_nf)
        self.att_fc = nn.Linear(hidden_nf * 2, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.ReLU(),
        )


    def forward(self, data, graph_fea):
        h0, coords, edge_index, batch = data.x, data.coord, data.edge_index, data.batch
        edge_attr = torch.unsqueeze(data.edge_attr, 1).to(torch.float)
        h = self.embed(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edge_index, coords, edge_attr, node_attr=h0)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edge_index, coords, edge_attr)

        h = F.relu(h)

        if self.attention:
            m = self.graph_fc(graph_fea)[batch]
            att_logits = self.att_fc(torch.tanh(torch.concat([h, m], dim=-1)))
            att_weights = scatter_softmax(att_logits, batch, dim=0)

            x_wei = 0.2 * h * att_weights + 0.8 * m
            graph_embd = scatter_add(x_wei, batch, dim=0)

            graph_embd = self.mlp(graph_embd)
        else:
            graph_embd = global_mean_pool(h, batch)


        return graph_embd

class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1


        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class ProtacModel(nn.Module):
    def __init__(self,
                 protac_model,
                 ligase_pocket_model,
                 target_pocket_model,
                 hidden_size):
        super().__init__()
        self.protac_model = protac_model
        self.ligase_pocket_model = ligase_pocket_model
        self.target_pocket_model = target_pocket_model
        self.fc1 = nn.Linear(hidden_size * 3, hidden_size)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self,
                protac_graphs,
                ligase_pocket,
                target_pocket,
                feature
                ):
        fea_protac, fea_tar, fea_e3 = feature[:, :167], feature[:, 167:197], feature[:, 197:227]
        v_0 = self.protac_model(protac_graphs, fea_protac)
        v_1 = self.ligase_pocket_model(ligase_pocket, fea_e3)
        v_3 = self.target_pocket_model(target_pocket, fea_tar)

        v_f = torch.cat((v_0, v_1, v_3), 1)
        latent = self.relu(self.fc1(v_f))
        v_f = self.fc2(latent)
        return v_f, latent