import torch as th
import gp_apis


# GSpmm is derived from Pytorch autograd
# func signature is def gp_gspmm(g, X, dim0, dim1, inverse, norm)

class Attention(th.Autograd.Function):
    @staticmethod
    def forward(graph, edge_src_feat, edge_dest_feat):
        ''' this function calculated edge attention between two neighbors'''
        dim = edge_src.size(1)
        print('sparse.py Attn fwd edge_src ' , edge_src_feat)
        print('sparse.py Attn fwd edge_src.shape ' , edge_src_feat.shape)
        print('sparse.py Attn fwd edge_dest ' , edge_dest_feat)
        print('sparse.py Attn fwd edge_dest.shape ' , edge_dest_feat.shape)
        feat_edge_src_tensor = th.utils.dlpack.to_dlpack(edge_src_feat) 
        feat_edge_dest_tensor = th.utils.dlpack.to_dlpack(edge_dest_feat)
        edge_count = graph.get_edge_count()
        result= th.zeros(edge_count, dim)
        result_tensor = th.utils.dlpack.to_dlpack(result)
        # this function is located in gp_api.py
        gp_apis.forward_edge_attention(graph, feat_edge_src_tensor, feat_edge_dest_tensor, result_tensor)
        return result

    @staticmethod
    def backward(graph, edge_feat):
        ''' this function calculated edge attention between two neighbors'''
        dim = edge_feat.size(1)
        print('sparse.py Attn bwd edge_feat ' , edge_feat)
        print('sparse.py Attn bwd edge_feat.shape ' , edge_feat.shape)
        feat_edge_tensor = th.utils.dlpack.to_dlpack(edge_feat) 
        node_count = graph.num_vcount()
        result1= th.zeros(node_count, dim)
        result2= th.zeros(node_count, dim)
        result_tensor1 = th.utils.dlpack.to_dlpack(result1)
        result_tensor2 = th.utils.dlpack.to_dlpack(result2)
        # this function is located in gp_api.py
        gp_apis.backprop_attention(graph, feat_edge_tensor, result_tensor1, result_tensor2)
        return result1, result2



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

'''
class EdgeSoftmax(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, efficient_score, num_vcount, num_ecount, dim):
	feat = th.utils.dlpack.to_dlpack(efficient_score)
        score_max = th.zeros(num_vcount, dim)
        result = th.utils.dlpack.to_dlpack(score_max)
        # find max edge value among neighbors
        # for score_max
        gp_apis.pick_largest_edge_weight(feat, result, 0)
        # subtract max edge weight value among neighbor from
	# all edge weights
        score_max = th.utils.dlpack.to_dlpack(score_max)
	# this dimension doesn't seem right
        score = th.zeros(num_vcount, dim)
        # score = th.zeros(num_ecount, dim)
        result = th.utils.dlpack.to_dlpack(score)
        gp_apis.subtract_max_score(score_max, feat, result, 0)
        # apply expo for score
        score = th.exp(score)
        # sum scores for all neighbors
	# result is size of num_vcount
        score = th.utils.dlpack.to_dlpack(score)
        score_sum = th.zeros(num_vcount, dim)
        result = th.utils.dlpack.to_dlpack(score_sum)
        gp_apis.sum_scores_for_neighbors(score, result, 0)
        score_sum = th.utils.dlpack.to_dlpack(score_sum)
	# this dimension doesn't seem right
        out = th.zeros(num_vcount, dim)
        # should it be this out = th.zeros(num_ecount, dim)
        result = th.utils.dlpack.to_dlpack(out)
        # todo score % score_sum.out is | E |
	# divide every edge score by the summed score for all neighbors
        gp_apis.div_edge_score_by_neighborhood(score_sum, score, result, 0)
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
        gp_apis.sum_scores_for_neighbors(fea, result, 0)
        accum = th.utils.dlpack.to_dlpack(accum)
        out = th.utils.dlpack.to_dlpack(out)
        temp = th.zeros(num_vcount, dim)
        result = th.utils.dlpack.to_dlpack(temp)
        temp = gp_apis.mult_edge_score_by_neighborhood(accum, out, result, 0)
        grad_score = sds - temp

        return None, grad_score, None, None


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

'''
def edge_softmax(graph, efficient_score, num_vcount, dim):
    '' performs the softmax over neighbors''
    result = EdgeSoftmax.apply(graph, efficient_score, num_vcount, dim)
    return result
'''
