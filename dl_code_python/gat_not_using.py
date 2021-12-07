''' Implementation adapted from:
 https://github.com/pradeep-k/GraphPy_workflow'''

import torch as th
from torch import nn
import sparse


# pylint: enable=W0235
class GATLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GATConv, self).__init__()
        self._in_features = in_feats
        self._out_features = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        self.linear = nn.Linear(
            self._in_features, out_feats, bias=False)
        self.fc = nn.Linear(
            self._in_features, out_feats, bias=False)
        self.attn_src = nn.Parameter(th.FloatTensor(size=(1, out_feats)))
        self.attn_dest = nn.Parameter(th.FloatTensor(size=(1, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        # re-initilize the parameter for linear layer
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        # re-initilize the parameter for attention layer
        nn.init.xavier_normal_(self.attn_src, gain=gain)
        nn.init.xavier_normal_(self.attn_dest, gain=gain)
        # re-initilize the parameter for linear layer
        # if isinstance(self.res_fc, nn.Linear):
        #     nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""
        Description
        -----------
        Set allow_zero_in_degree flag.
        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):

	# Linear Projection
	# map higher dimensionality features to lower dimensionality
	# features using a weight matrix of size F' (new size) by F
        self.fc_src, self.fc_dst = self.fc, self.fc
	# view can effectively reshape tensor data
	# passing -1 allows it to infer shape of the out features
        feat_src = feat_dest = self.fc(feat).view(-1, self._out_features)
        print('size edge_src = ' , edge_src.size())
        print('size attn_src =', self.attn_src.size())
	# these calls project the features using the attn parameters
	# which are the shape of the out_features
        edge_src = (feat_src * self.attn_src).sum(dim=-1).unsqueeze(-1)
        print('size edge_src = ' , edge_src.size())
        edge_dest = (feat_dest * self.attn_dest).sum(dim=-1).unsqueeze(-1)
        print('size edge_dest= ' , edge_dest.size())



        # map_input = self.linear(feat).view(-1, self._num_heads, self._out_feats)
        # print (map_input.size())
        # print(self.attn_l.size())
        # el = th.matmul(map_input, self.attn_l)
        # er = th.matmul(map_input,  self.attn_r)

        num_vcount = graph.get_vcount()
        num_ecount = graph.get_ecount()
        # since the score is scalar
        dim = feat_src.size(1)


	# compute self-attention e_{ij} = a(W h_i, W h_j)
	# single layer FF on concatenated features of 2 neighbor nodes
	# that have been projected into output space using attn paramd
	# apply leaky Relu non-linearity
        # apply_edge adds the attention_score by vertex
	# this will call the kernel
        edge_score = sparse.apply_edge(graph, edge_src, edge_dest)
        efficient_score = self.leaky_relu(edge_score)

	# apply softmax to normalize over all choices of destination
	# makes coefficients easily comparable for different nodes
        edge_score_by_softmax = sparse.edge_softmax(graph,
						efficient_score,
						num_vcount,
						num_ecount,
						dim)
	print('size edge_score_by_softmax = ', edge_score_by_softmax.size())

        #     // std::cout << "nani6" << std::endl;
        # torch::Tensor

	# aggregate calculations over the neighborhood
	# this calls a forward or backward method via apply
	# both of which rely on spmmw_op
        print('size feat_src = ' , feat_src.size())
        #result = sparse.run_gspmv_op(graph, feat_src, edge_score_by_softmax, num_vcount, dim)
	result = th.bmm(edge_score_by_softmax, feat_src)
        print ('size result =', result.size())

        # activation
        if self.activation:
            result = self.activation(result)
        return result

class GAT(nn.Module):
    def __init__(self,
                 graph,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 activation = None,
                 feat_drop = 0.,
                 attn_drop = 0.,
                 negative_slope =0.2,
                 residual = False):
        super(GAT, self).__init__()
        self.graph = graph
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden,
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden, num_hidden,
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
          num_hidden, num_classes, feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers + 1):
            h = self.gat_layers[l](self.graph, h).flatten(1)
        # output projection
        # logits = self.gat_layers[-1](self.graph, h).mean(1)
        return h
