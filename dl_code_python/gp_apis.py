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

def gp_gspmm_mt(g, X, dim0, dim1, inverse, norm):
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
    gpk.gspmm_mt(g, X_dl, res_dl, inverse, norm)  # do not specify the reduce operation

    return res
