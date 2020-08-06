"""
Interface of Attention Module (including Position Attention Module and
Channel Attention Module) using pytorch.
Author: Zhu Liu
Time: 2019/12/27

Refer:
1. Fu, Jun, et al. "Dual attention network for scene segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
2. https://github.com/Andy-zhujunwen/danet-pytorch
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def PAM(x, isSoftmax=True, isEuclidean=True):
    """Position Attention Module
    Similarity/Attention map within an feature map each channel and each instance in the minibatch.

    Parameters:
    -------------
    x : input tensor of shape (batch, in_channels, iH, iW).
        It can be the feature map after several convolutions

    isSoftmax : bool (default: True)
        Decide the final probability using whether softmax or not.
        If True, use the softmax.
        If False, use the element divided by the sum.

    isEuclidean : bool (default: True)
        Decide the similarity metrics whether in the isEuclidean space.
        If True, use the inner product to decide the similarity.
        If False, use the Mahalanobis space. (This module will scale later if needed)

    Returns:
    --------
    y : output tensor of shape (batch, iH*iW, iH*iW).
        Each position is the normalized similarity from each row to each column.
    """
    batch_size, _, height, width = x.size()
    x = x.contiguous()
    query = x.view(batch_size, -1, height*width).permute(0,2,1)
    gallery = x.view(batch_size, -1, height*width)
    if isEuclidean:
        att = torch.bmm(query, gallery)

    if isSoftmax:
        y = F.softmax(att, dim=-1)
    else:
        y = att / torch.sum(att,dim=-1,keepdim=True)


    return y

def CAM(x, isSoftmax=True, isEuclidean=True):
    """Channel Attention Module
    Similarity/Attention map across channel using each instance in the minibatch

    Parameters:
    ------------
    x : input tensor of shape (batch, in_channels, iH, iW).
        It can be the feature map after sereval convolutions.

    isSoftmax : bool (default: True)
        Decide the final probability using whether softmax or not.
        If True, use the softmax.
        If False, use the element divided by the sum.

    isEuclidean : bool (default: True)
        Decide the similarity metrics whether in the isEuclidean space.
        If True, use the inner product to decide the similarity.
        If False, use the Mahalanobis space. (This module will scale later if needed)

    Returns:
    --------
    y : output tensor of shape (batch, in_channels, in_channels).
        Each channel of a position is normalized across different channels.
    """
    batch_size, channel, height, weight = x.size()
    x = x.contiguous()
    query = x.view(batch_size, -1, height*weight)
    gallery = query.permute(0,2,1)
    y = torch.bmm(query, gallery)
    att = torch.max(y, dim=-1, keepdim=True)[0].expand_as(y) - y
    if isSoftmax==True:
        y = F.softmax(y, dim=-1)
    else:
        y = att / torch.sum(att,dim=-1,keepdim=True)
    return y

#test
if __name__ == "__main__":
    t= torch.randn(2,3,5,5)
    print(t)

    p = PAM(t, isSoftmax=False)
    print(p)

    c = CAM(t, isSoftmax=False)
    print(c)
