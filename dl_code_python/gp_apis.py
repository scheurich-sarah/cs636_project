import torch as th
import torch.utils.dlpack
import kernel as gpk
import datetime

# this is for hiding abstractions
def gp_gspmm(g, X, dim0, dim1, inverse, norm):
    X_dl = th.utils.dlpack.to_dlpack(X)

    # declare the output tensor here
    # can't convert the tensor to a numpy array
    # tensors are very tightly integrated with the computational DAG
    res = th.zeros(dim0, dim1)
    # allocated then convert back to Pytorch tensor
    res_dl = th.utils.dlpack.to_dlpack(res)

    # this is actually calling kernel gpk
    # since using own kernel, need to implement a 0 copy pass using pointers
    # copying the data will break the computational graph
    
    # x and x_dl point to the same mem location
    # x_dl is not exactly a C++ data struc, data type is python capsule
    gpk.gspmm(g, X_dl, res_dl, inverse, norm)  # do not specify the reduce operation

    return res
    
def forward_attention(graph, feat_edge_src_tensor, feat_edge_dest_tensor, result_tensor):

    gpk.forward_edge_attn(graph, feat_edge_src_tensor, feat_edge_dest_tensor, result_tensor)

def backprop_attention(graph, feat_edge_tensor, result_tensor1, result_tensor2):
    gpk.backprop_attn(graph, feat_edge_tensor, result_tensor1, result_tensor2)
'''
def pick_largest_edge_weight(feat, result, inverse):
    gpk.pick_largest_edge_weight((feat, result, inverse)

def  subtract_max_score(score_max, feat, result, inverse):
    gpk.subtr subtract_max_score(score_max, feat, result, inverse)
        
def sum_scores_for_neighbors(score, result, inverse):
    gpk.sum_scores_for_neighbors(score, result, inverse)
        
def div_edge_score_by_neighborhood(score_sum, score, result, inverse):
    gpk.div_edge_score_by_neighborhood(score_sum, score, result, inverse)

def mult_edge_score_by_neighborhood(accum, out, result, inverse):
    gpk.mult_edge_score_by_neighborhood(accum, out, result, inverse)
'''
