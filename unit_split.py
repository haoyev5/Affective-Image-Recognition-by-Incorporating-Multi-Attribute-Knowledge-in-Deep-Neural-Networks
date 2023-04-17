# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:54:37 2020

@author: haoye
"""
import mxnet as mx
import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import nn, utils as gutils

class Split_C_block(nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        super(Split_C_block, self).__init__(**kwargs)        
        self.c1 = nn.Conv2D(channels=channels//4, kernel_size=1, strides=1)
        self.c2 = nn.Conv2D(channels=channels//4, kernel_size=3, strides=1, padding=1)
        #self.c3 = nn.Conv2D(channels=channels//4, kernel_size=5, strides=1, padding=2)
        #self.c4 = nn.Conv2D(channels=channels//4, kernel_size=7, strides=1, padding=3)
        self.c3 = nn.Conv2D(channels=channels//4, kernel_size=3, dilation=2, padding=2)
        self.c4 = nn.Conv2D(channels=channels//4, kernel_size=3, dilation=3, padding=3)
        
    def forward(self, X):
        split = nd.split(data=X, axis=1, num_outputs=4)
        c1    = self.c1(split[0])
        c2    = self.c2(split[1])
        c3    = self.c3(split[2])
        c4    = self.c4(split[3])
        return nd.concat(c1, c2, c3, c4, dim=1)

class Split_C_block2(nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        super(Split_C_block2, self).__init__(**kwargs)        
        self.c1_1 = nn.Conv2D(channels=channels//4, kernel_size=1, strides=1)
        self.c2_1 = nn.Conv2D(channels=channels//4, kernel_size=3, strides=1, padding=1)
        #self.c3_1 = nn.Conv2D(channels=channels//4, kernel_size=5, strides=1, padding=2)
        #self.c4_1 = nn.Conv2D(channels=channels//4, kernel_size=7, strides=1, padding=3)
        self.c3_1 = nn.Conv2D(channels=channels//4, kernel_size=3, dilation=2, padding=2)
        self.c4_1 = nn.Conv2D(channels=channels//4, kernel_size=3, dilation=3, padding=3)
        
    def forward(self, x):
        split = nd.split(data=x, axis=1, num_outputs=4)
        c1    = self.c1_1(split[0])
        c2    = self.c2_1(split[1])
        print('c2 shape', c2.shape)
        c3    = self.c3_1(split[2])
        print('c3 shape', c3.shape)
        c4    = self.c4_1(split[3])
        print('c4 shape', c4.shape)
        return nd.concat(c1, c2, c3, c4, dim=1)


# multi channels for chaunxing 
class Attention(nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        super(Attention, self).__init__(**kwargs)        
        self.mean = nn.GlobalAvgPool2D()
        self.flat = nn.Flatten()               
        self.dense1 = nn.Dense(units=channels//16, activation='relu')
        self.dense2 = nn.Dense(units=channels, activation='relu')                
        self.spatial_conv1  = Split_C_block(channels)
        self.spatial_bn1    = nn.BatchNorm(epsilon=2e-5, momentum=0.9)
        self.spatial_conv2  = nn.Conv2D(channels=channels//16, kernel_size=1, strides=1, padding=0)
        self.spatial_bn2    = nn.BatchNorm(epsilon=2e-5, momentum=0.9)
        self.spatial_conv3  = nn.Conv2D(channels=1, kernel_size=1, strides=1, padding=0)
        self.CA = 1
        self.SA = 1        
        
    def forward(self, X):
        # channel attention
        c = self.dense1(self.flat(self.mean(X)))
        c = self.dense2(c)
        channel_att = nd.sigmoid(c)
        self.CA = nd.reshape(channel_att, shape=(-1, channel_att.shape[1], 1, 1))
        channel_att = nd.broadcast_mul(X, nd.reshape(channel_att, shape=(-1, channel_att.shape[1], 1, 1)))
        #spatial attention
        spatial_att = nd.relu(self.spatial_bn1(self.spatial_conv1(channel_att)))
        spatial_att = nd.sigmoid(self.spatial_conv3(self.spatial_bn2(self.spatial_conv2(spatial_att))))
        self.SA = spatial_att
        out = nd.broadcast_mul(channel_att, spatial_att)
        return X+out
    

'''
X = nd.random.uniform(shape=(1,32,32,32))
blk = Split_C_block2(32)
blk.initialize()
y = blk(X)
print(y.shape)
'''

class Attention_Residual_Unit(nn.HybridBlock):
    def __init__(self, channels, strides, same_shape=True, **kwargs):
        super(Attention_Residual_Unit, self).__init__(**kwargs)
        epsilon= 2e-5        
        momentum=0.9
        self.same_shape = same_shape
        self.bn1    = nn.BatchNorm(epsilon=epsilon, momentum=momentum)
        self.conv1  = nn.Conv2D(channels=channels, kernel_size=3, strides=1, padding=1)
        self.bn2    = nn.BatchNorm(epsilon=epsilon, momentum=momentum)
        self.conv2  = nn.Conv2D(channels=channels, kernel_size=3, strides=strides, padding=1)
        self.bn3    = nn.BatchNorm(epsilon=epsilon, momentum=momentum)
        self.att    = Attention(channels)
        
        if not same_shape:
            self.conv = nn.Conv2D(channels=channels,kernel_size=1,strides=strides, padding=0)
            self.bn   = nn.BatchNorm(epsilon=epsilon,momentum=momentum)            
        
    def forward(self, X):
        y = nd.relu(self.bn2(self.conv1(self.bn1(X))))
        y = self.att(self.bn3(self.conv2(y)))
        if not self.same_shape:
            X = self.bn(self.conv(X))
        return X+y

def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n

def _get_batch(batch, ctx):
    """Return features and labels on ctx."""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
        
def plot(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(7, 5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.plot(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.plot(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
        
