''' This module is adapted from
https://github.com/gordicaleksa/pytorch-GAT.git/models/definitions/GAT.py'''

import torch
import torch.nn as nn
import sparse



class GAT(nn.Module):
    """
    only using 1 of the 3 GAT implementations provided on the github
    """

    def __init__(self, graph, num_of_layers, num_heads_per_layer,
            num_features_per_layer, add_skip_connection=True, 
            bias=True, dropout=0.6, log_attention_weights=False):
        super().__init__()

        # set the graph attribute
        self.graph = graph

        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        # trick - so that I can nicely create GAT layers below
        num_heads_per_layer = [1] + num_heads_per_layer 

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = GATLayer(
                    graph=self.graph,
                    num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                    num_out_features=num_features_per_layer[i+1],
                    num_heads=num_heads_per_layer[i+1],
                    concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                    activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                    dropout_prob=dropout,
                    add_skip_connection=add_skip_connection,
                    bias=bias,
                    log_attention_weights=log_attention_weights
                    )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(*gat_layers,)

    # data is just a (in_nodes_features, topology) tuple, 
    # I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but...
    def forward(self, data):
        ''' this forward method gets called by the framework'''
        return self.gat_net(data)


class GATLayer(torch.nn.Module):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.
    """

    head_dim = 1

    def __init__(self, graph, num_in_features, num_out_features, num_heads,
            concat=True, activation=nn.ELU(), dropout_prob=0.6,
            add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        # Saving these as we'll need them in forward propagation in child layers 
        self.num_of_heads = num_heads
        self.num_out_features = num_out_features
        # whether we should concatenate or average the attention heads
        self.concat = concat 
        self.add_skip_connection = add_skip_connection

        #
        # Trainable weights:
        # linear projection matrix (denoted as "W" in paper)
        # attention target/source (denoted as "a" in the paper)
        # bias (not mentioned in the paper but present in the official GAT repo)
        #

        self.proj_param = nn.Parameter(torch.Tensor(num_heads,
            num_in_features,
            num_out_features))

        # After we concatenate target node (node i) 
        # and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". 
        # Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y]
        # (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" 
        # and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1,
            num_heads,
            num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1,
            num_heads,
            num_out_features))

        self.scoring_fn_target = nn.Parameter(self.scoring_fn_target.reshape(num_heads, num_out_features, 1))
        self.scoring_fn_source = nn.Parameter(self.scoring_fn_source.reshape(num_heads, num_out_features, 1))

        # Bias is definitely not crucial to GAT
        # feel free to experiment (I pinged the main author, Petar, on this)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features,
                    num_heads * num_out_features,
                    bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        # using 0.2 as in the paper, no need to expose every setting
        self.leakyReLU = nn.LeakyReLU(0.2)
        
        # -1 stands for apply the log-softmax along the last dimension
        self.softmax = nn.Softmax(dim=-1) 
        
        self.activation = activation
        
        # Probably not the nicest design but I use the same module 
        # in 3 locations, before/after features projection
        # and for attention coefficients.
        # Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        # whether to log the attention weights
        self.log_attention_weights = log_attention_weights
        # cached for visualization purposes
        self.attention_weights = None

        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization 
        is because it's a default TF initialization:
        https://stackoverflow.com/questions/37350131/what-is-the-default-va...
        
        The original repo was developed in TensorFlow (TF)
        and they used the default initialization.
        """
        
        nn.init.xavier_uniform_(self.proj_param)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        # potentially log for later visualization in playground.p
        if self.log_attention_weights:
            self.attention_weights = attention_coefficients

        # if the tensor is not contiguously stored in memory,
        # we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            #if FIN == FOUT
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), 
                # out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times
                # and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors
                # into dimension that can be added to output feature vectors.
                # skip_proj adds lots of additional capacity may cause overfit
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1,
                        self.num_of_heads,
                        self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1,
                    self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #

        in_nodes_features, connectivity_mask = data  # unpack data
        num_of_nodes = in_nodes_features.shape[0]
        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'

        # shape = (N, FIN)
        # where N - number of nodes in the graph
        # FIN number of input features per node
        # We apply the dropout to all the input node features (per the paper)
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (1, N, FIN) * (NH, FIN, FOUT) -> (NH, N, FOUT)
        # where NH - number of heads, FOUT num of output features
        # We project the input node features into
        # NH independent output features (one for each attention head)
        nodes_features_proj = torch.matmul(in_nodes_features.unsqueeze(0),
                self.proj_param)

         # in the official GAT imp they did dropout here as well
        nodes_features_proj = self.dropout(nodes_features_proj) 

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function 
        # (* represents element-wise (a.k.a. Hadamard) product)
        # batch matrix multiply,
        ## shape = (NH, N, FOUT) * (NH, FOUT, 1) -> (NH, N, 1)
        scores_source = torch.bmm(nodes_features_proj, self.scoring_fn_source)
        scores_target = torch.bmm(nodes_features_proj, self.scoring_fn_target)

        # shape = (NH, N, 1) + (NH, 1, N) -> (NH, N, N) due to auto broadcast
        # Tip: it's conceptually easier to understand what
        # happens here if you delete the NH dimension
        all_scores = self.leakyReLU(scores_source + 
                scores_target.transpose(1, 2))
        # connectivity mask will put -inf on all locations 
        # where there are no edges, after applying the softmax
        # this will result in attention scores being computed only
        # for existing edges
        all_attention_coefficients = self.softmax(all_scores +
                connectivity_mask)

        #
        # Step 3: Neighborhood aggregation
        #

        # shape = (NH, N, N) * (NH, N, FOUT) -> (NH, N, FOUT)
        out_nodes_features = torch.bmm(all_attention_coefficients,
                nodes_features_proj)

        # shape = (N, NH, FOUT)
        out_nodes_features = out_nodes_features.transpose(0, 1)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(all_attention_coefficients,
                in_nodes_features, out_nodes_features)
        return (out_nodes_features, connectivity_mask)
