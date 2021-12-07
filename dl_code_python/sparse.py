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


class EdgeSoftmax(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, efficient_score, num_vcount, dim):

        #feat = th.utils.dlpack.to_dlpack(efficient_score)
        feat = efficient_score
        score_max = th.zeros(num_vcount, dim)
        result = th.utils.dlpack.to_dlpack(score_max)
        # todo find max edge value
        # for score_max
	# this function finds the neighbor with the maximum edge weight
	# for each vertex 
        graph.spmmw(feat, result, enumOP.eMAX.value, 0)
        # sub from score_max
        score_max = th.utils.dlpack.to_dlpack(score_max)
        score = th.zeros(num_vcount, dim)
        result = th.utils.dlpack.to_dlpack(score)
        # todo find score - score_max
        # for score
	# this function just performs a subtraction
        graph.sddmm(score_max, feat, result, enumOP.eSUB.value, 0)
        # apply expo for score
        score = th.exp(score)
        score = th.utils.dlpack.to_dlpack(score)
        score_sum = th.zeros(num_vcount, dim)
        result = th.utils.dlpack.to_dlpack(score_sum)
        # todo score_sum
        graph.spmmw(score, result, enumOP.eSUM.value, 0)
        score_sum = th.utils.dlpack.to_dlpack(score_sum)
        out = th.zeros(num_vcount, dim)
        result = th.utils.dlpack.to_dlpack(out)
        # todo score % score_sum.out is | E |
        graph.sddmm(score_sum, score, result, enumOP.eDIV.value, 0)
        ctx.backward_cache = graph, num_vcount, dim, out
        return out

    @staticmethod
    def backward(ctx, dZ):
        graph, num_vcount, dim, out = ctx.backward_cache
        sds = out * dZ

        fea = th.utils.dlpack.to_dlpack(sds)
        accum = th.zeros(num_vcount, dim)
        result = th.utils.dlpack.to_dlpack(accum)
        # for accum
        graph.spmmw(fea, result, enumOP.eSUM.value, 0)
        accum = th.utils.dlpack.to_dlpack(accum)
        out = th.utils.dlpack.to_dlpack(out)
        temp = th.zeros(num_vcount, dim)
        result = th.utils.dlpack.to_dlpack(temp)
        temp = graph.sddmm(accum, out, result, enumOP.eMUL.value, 0)
        grad_score = sds - temp

        return None, grad_score, None, None

'''
class GSpmv_op(th.autograd.Function):
    @staticmethod
    def forward(ctx, X, graph, edge_score_by_softmax, num_vcount, dim):
        ''must provide for framework
	input dim = 2; output dim = 1
	src and destination vertices are concatenated
	into onse set of features''
	feat_X_tensor = th.utils.dlpack.to_dlpack(X)
	feat_edge_score_by_softmax_tensor = th.utils.dlpack.to_dlpack(edge_score_by_softmax)
	# 
	result = th.zeros(num_vcount, dim)
	result_tensor = th.utils.dlpack.to_dlpack(result)
	#
	graph.spmmw_op(feat_X_tensor, feat_edge_score_by_softmax_tensor, result_tensor,
			enum.OP.eSUM.value, 0)
	ctx.backward_cache = graph, edge_score_by_softmax, num_vcount, dim
	return result

    def backward(ctx, dZ):
        '' must provide for framework
	input dim = 1 (dZ); output dim = 2
	need to map a single dimension back to residual
	errors for sources and destinations''
	graph, edge_score_by_softmax, num_vcount, dim = ctx.backward_cache
        feat_X_tensor = th.utils.dlpack.to_dlpack(dZ)
        feat_edge_score_by_softmax_tensor = th.utils.dlpack.to_dlpack(edge_score_by_softmax)
        result = th.zeros(num_vcount, dim)
        result_tensor = th.utils.dlpack.to_dlpack(result)
        graph.spmmw_op(feat_X_tensor, feat_edge_score_by_softmax_tensor, result_tensor, enumOP.eSUM.value, 1)
        return result, None, None, None, None
'''


# the gspmv has only 1 input
# and then apply different operations such as sum, max on it
# apply calls something in Pytorch that helps it decide whether
# use the user's forward or backward method
def run_gspmm(graph, X, norm, num_vcount, dim):
    return GSpmm.apply(graph, X, norm, num_vcount, dim)

'''
def run_gspmv_op(graph, X, edge_score_by_softmax, num_vcount, dim):
    '' used with graph transformer network self-attention
    calls forward or backward method''
    return GSpmv_op.apply(X, graph, edge_score_by_softmax, num_vcount, dim)	
'''

def apply_edge(graph, edge_src, edge_dest):
    ''' this function performs the attention between two neighbor nodes'''
    dim = edge_src.size(1)
    feat_edge_src_tensor = th.utils.dlpack.to_dlpack(edge_src) 
    feat_edge_dest_tensor = th.utils.dlpack.to_dlpack(edge_dest)
    edge_count = graph.get_edge_count()
    result= th.zeros(edge_count, dim)
    result_tensor = th.utils.dlpack.to_dlpack(result)
    # this function is located in gp_api.py
    graph.apply_edges_op2d(feat_edge_src_tensor, feat_edge_dest_tensor, result_tensor)
    return result

def edge_softmax(graph, efficient_score, num_vcount, dim):
    ''' performs the softmax over neighbors'''
    result = EdgeSoftmax.apply(graph, efficient_score, num_vcount, dim)
    return result
