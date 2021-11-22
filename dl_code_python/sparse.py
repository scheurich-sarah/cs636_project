import torch as th
import gp_apis


# GSpmm is derived from Pytorch autograd
# func signature is def gp_gspmm(g, X, dim0, dim1, inverse, norm)

class GSpmm(th.autograd.Function):
    
    # must provide
    @staticmethod
    def forward(ctx, graph, X, norm, num_vcount, dim):
        res = gp_apis.gp_gspmm(graph, X, num_vcount, dim, 0, norm)  # do not specify the reduce operation
        ctx.backward_cache = graph, norm, num_vcount, dim
        return res

    # must provide
    @staticmethod
    def backward(ctx, dZ):
        graph, norm, num_vcount, dim = ctx.backward_cache
        res = gp_apis.gp_gspmm(graph, dZ, num_vcount, dim, 1, norm)  # do not specify the reduce operation
        return None, res, None, None, None

# the gspmv has only 1 input
# and then apply different operations such as sum, max on it
# apply calls something in Pytorch that helps it decide whether
# use the user's forward or backward method
def run_gspmm(graph, X, norm, num_vcount, dim):
    return GSpmm.apply(graph, X, norm, num_vcount, dim)

class GSpmm_mt(th.autograd.Function):
    
    # must provide
    @staticmethod
    def forward(ctx, graph, X, norm, num_vcount, dim):
        res = gp_apis.gp_gspmm_mt(graph, X, num_vcount, dim, 0, norm)  # do not specify the reduce operation
        ctx.backward_cache = graph, norm, num_vcount, dim
        return res

    # must provide
    @staticmethod
    def backward(ctx, dZ):
        graph, norm, num_vcount, dim = ctx.backward_cache
        res = gp_apis.gp_gspmm_mt(graph, dZ, num_vcount, dim, 1, norm)  # do not specify the reduce operation
        return None, res, None, None, None

# the gspmv has only 1 input
# and then apply different operations such as sum, max on it
# apply calls something in Pytorch that helps it decide whether
# use the user's forward or backward method
def run_gspmm_mt(graph, X, norm, num_vcount, dim):
    return GSpmm_mt.apply(graph, X, norm, num_vcount, dim)
