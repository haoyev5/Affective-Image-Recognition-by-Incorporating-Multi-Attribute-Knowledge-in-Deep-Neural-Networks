

# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:17:57 2020

@author: haoye
"""
import sys
import os
#sys.path.append(os.getcwd())
os.system('dir')
os.system('ls')
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils,nn
#import gluoncv
import time
from einops import rearrange
import d2lzh as d2l
sys.path.append(os.getcwd())
import unit_split
import matplotlib.pyplot as plt
from gluoncv import model_zoo
import utils
from emosimi_loss import convert_label_to_similarity, EmosimiLoss
#from mxnet.gluon import model_zoo

# some superparameters that we can fine-tuning
batch_size          = 32
num_epochs, lr, wd  = 80, 0.001, 1e-4
lr_period, lr_decay = 10, 0.1
epsilon, momentum   = 2e-5, 0.9

# modual-net parameters
num_classes = 8

#data file name
data_dir   = 'D:/data/FI'
train_dir, test_dir = 'train', 'valid'

# try to train model on GPU / GPUs
ctx = [mx.gpu(0),mx.gpu(1),mx.gpu(2),mx.gpu(3)]


# Preprocessing data
transform_train = gdata.vision.transforms.Compose([
    # 随机对图像裁剪出面积为原图像面积0.08~1倍、且高和宽之比在3/4~4/3的图像，再放缩为高和
    # 宽均为224像素的新图像
    gdata.vision.transforms.Resize(256),
    gdata.vision.transforms.RandomResizedCrop(224, scale=(0.20, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.RandomColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
    #gdata.vision.transforms.RandomColorJitter(brightness=0.4, contrast=0.4),
    gdata.vision.transforms.RandomLighting(0.1),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(224),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])])


# load data
train_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, train_dir), flag=1)
if test_dir is not None:
    test_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, test_dir), flag=1)
else:
    test_ds = None
train_iter = gdata.DataLoader(train_ds.transform_first(transform_train), batch_size=batch_size, shuffle=True, last_batch='rollover')
print('train iter complete!')
if test_ds is not None:
    test_iter = gdata.DataLoader(test_ds.transform_first(transform_test), batch_size=batch_size, shuffle=False, last_batch='rollover')
    print('test iter complete!')
else:
    test_iter = None
    print('No test iter! Go ahead---->')
    
############################################################################
#loss = gloss.SoftmaxCrossEntropyLoss()

def AugFocalLoss(y_hat, y, gamma=1, ctx=ctx):
    y_hat = nd.softmax(y_hat)
    alpha = nd.array([0.053, 0.211, 0.086, 0.049, 0.158, 0.092, 0.258, 0.094],ctx=ctx) + nd.zeros(shape=(y.size, 1),ctx=ctx)
    loss1 = gloss.SoftmaxCrossEntropyLoss()
    #Fl =     -alpha       *         (1-p)**gamma        *  ln(p) 
    fl = nd.pick(alpha, y) * ((1-nd.pick(y_hat, y))**gamma) * loss1(y_hat, y)
    return fl


def loss(y_hat, y, alpha=2):
    vote_l = AugFocalLoss(y_hat, y)
    loss1 = gloss.SoftmaxCrossEntropyLoss()
    entropy_l = loss1(y_hat, y)
    return entropy_l + alpha*vote_l 

############################################################################
class Gram(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Gram, self).__init__(**kwargs)        
        
    def forward(self, X):
        B,C,H,W = X.shape 
        X = X.reshape((B, C, H*W))
        Y = X.transpose((0,2,1))
        return nd.linalg_gemm2(X, Y) / (C*H*W)
    

class Tokenizer(nn.HybridBlock):
    def __init__(self, batch_size=32, channel=64, num_tokens=64, embed=32, ctx=ctx, **kwargs):
        super(Tokenizer, self).__init__()
        self.L = num_tokens
        self.c = channel
        self.cT = embed
        
        with self.name_scope():
            self.token_wA = self.params.get('weight', shape=(batch_size, self.L, self.c))
            self.token_wV = self.params.get('weight1', shape=(batch_size, self.c, self.cT))
        
    def forward(self, features, mask = None):
        # softmax(x*wa) * (xv).T
        x = rearrange(features, 'b c h w -> b (h w) c')
        #print('x shape:', x.shape)
        wa = rearrange(self.token_wA.data(ctx=ctx), 'b h w -> b w h')
        A = nd.linalg.gemm2(x, wa)
        
        A = rearrange(A, 'b h w -> b w h')
        A = A.softmax(axis=-1)

        V = nd.linalg.gemm2(x, self.token_wV.data(ctx=ctx))
        T = nd.linalg.gemm2(A, V)
        
        return T


def get_resnet_features_extractor(ctx):
    # get pretrained model from mxnet model_zoo
    resnet = model_zoo.get_model('resnet50_v2', pretrained=True, ctx=ctx)
    features_extractor = resnet.features
    return features_extractor

def get_net(num_classes, batch_size, ctx):
    class ResNet(nn.HybridBlock):
        def __init__(self, ctx, **kwargs):
            super( ResNet, self).__init__(**kwargs)
            
            embed_size = 64 
            reduction  = 1
            
            dropout = 0.4
            
            self.feature_extractor = get_resnet_features_extractor(ctx)

            with self.name_scope():
                self.branch_1 = nn.HybridSequential(prefix='branch_1')
                with self.branch_1.name_scope():
                    self.branch_1.add(nn.LayerNorm(),
                                      Tokenizer(batch_size, 64, num_tokens=16),
                                      nn.Flatten(),
                                      nn.BatchNorm(epsilon=2e-5, momentum=0.9),
                                      nn.Dense(embed_size//reduction, activation='relu'),
                                      nn.BatchNorm(epsilon=2e-5, momentum=0.9))
                self.branch_1.initialize(init.Xavier(), ctx=ctx)
                    
                self.branch_2 = nn.HybridSequential(prefix='branch_2')
                with self.branch_2.name_scope():
                    self.branch_2.add(nn.LayerNorm(),
                                      Tokenizer(batch_size, 256, num_tokens=16),
                                      nn.Flatten(),
                                      nn.BatchNorm(epsilon=2e-5, momentum=0.9),
                                      nn.Dense(embed_size//reduction, activation='relu'),
                                      nn.BatchNorm(epsilon=2e-5, momentum=0.9))
                self.branch_2.initialize(init.Xavier(), ctx=ctx)
                    
                self.branch_3 = nn.HybridSequential(prefix='branch_3')
                with self.branch_3.name_scope():
                    self.branch_3.add(nn.LayerNorm(),
                                      Tokenizer(batch_size, 512, num_tokens=16),
                                      nn.Flatten(),
                                      nn.BatchNorm(epsilon=2e-5, momentum=0.9),
                                      nn.Dense(embed_size//reduction, activation='relu'),
                                      nn.BatchNorm(epsilon=2e-5, momentum=0.9))
                self.branch_3.initialize(init.Xavier(), ctx=ctx)
                
                self.branch_4 = nn.HybridSequential(prefix='branch_4')
                with self.branch_4.name_scope():
                    self.branch_4.add(nn.LayerNorm(),
                                      Tokenizer(batch_size, 1024, num_tokens=16),
                                      nn.Flatten(),
                                      nn.BatchNorm(epsilon=2e-5, momentum=0.9),
                                      nn.Dense(embed_size//reduction, activation='relu'),
                                      nn.BatchNorm(epsilon=2e-5, momentum=0.9))
                self.branch_4.initialize(init.Xavier(), ctx=ctx)
                
                self.branch_5 = nn.HybridSequential(prefix='branch_5')
                with self.branch_5.name_scope():
                    self.branch_5.add(nn.LayerNorm(),
                                      Tokenizer(batch_size, 2048, num_tokens=16),
                                      nn.Flatten(),
                                      nn.BatchNorm(epsilon=2e-5, momentum=0.9),
                                      nn.Dense(embed_size//reduction, activation='relu'),
                                      nn.BatchNorm(epsilon=2e-5, momentum=0.9))
                self.branch_5.initialize(init.Xavier(), ctx=ctx)
    
                self.styleVector = nn.HybridSequential(prefix='styleVector')
                with self.styleVector.name_scope():
                    self.styleVector.add(#nn.Dropout(.8),
                                         nn.BatchNorm(epsilon=2e-5, momentum=0.9),
                                         nn.Dense(512//reduction, activation='relu'))
                self.styleVector.initialize(init.Xavier(), ctx=ctx)
                
    
                self.output = nn.HybridSequential(prefix='output')
                with self.output.name_scope():
                    self.output.add(nn.Dropout(dropout),
                                    #nn.BatchNorm(epsilon=2e-5, momentum=0.9),
                                    nn.Dense(num_classes))
                self.output.initialize(init.Xavier(), ctx=ctx)
                                     
            self.flat = nn.Flatten()
               

        def hybrid_forward(self, F, X, *args, **kwargs):
            
            # branch_1
            y = self.feature_extractor[:4](X)
            branch_1 = self.branch_1(y)
            #print('gram1 shape:', gram1.shape)                    
            
            # branch_2
            y = self.feature_extractor[4:6](y)
            #print('branch_2 y shape:', y.shape)
            branch_2 = self.branch_2(y) 
            
            # branch_3
            y = self.feature_extractor[6:7](y)
            #print('branch_3 y shape:', y.shape)
            branch_3 = self.branch_3(y)
            
            # branch_4
            y = self.feature_extractor[7:8](y)
            #print('branch_3 y shape:', y.shape)
            branch_4 = self.branch_4(y)
            
            # branch_5
            y = self.feature_extractor[8:9](y)
            #print('branch_3 y shape:', y.shape)
            branch_5 = self.branch_5(y)
            
            #style feature representation
            style = nd.concat(branch_1, branch_2, branch_3, branch_4, branch_5, dim=1)
            style = self.styleVector(style)
            
            # high-level feature representation: backbone feature
            y = self.feature_extractor[9:](y)
            y = self.flat(y)
            
            # integrate low- and high-level feature and output
            out = nd.concat(style, y, dim=1)
            #out = nd.concat(y, style, dim=1) or nd.broadcast_mul(y,style) or nd.broadcast_add(style, y)
            out = self.output(out)                    
            
            return out

    return ResNet(ctx)

net = get_net(num_classes, batch_size, ctx)
#net.initialize(init.Xavier(), ctx=ctx)

X = nd.random.uniform(shape=(8,3,224,224)).as_in_context(ctx)
print('Input shape:', X.shape)
X = net(X)
print('Output shape:', X.shape)

print("==============================================================")
print("==============================================================")

############################################################################
  
def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])
############################################################################
def train(net, train_iter, test_iter, num_epochs, lr, wd, ctx, lr_period, lr_decay):
    print('training on', ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    train_acc, test_acc = [], []
    train_loss = []
    alpha = 0.0
    criterion = EmosimiLoss(m=0.25, gamma=80)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        duandian = lr_period *2 + 1
        if epoch>0 and epoch % lr_period == 0 and epoch < duandian:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        #超过40轮每10轮学习率自乘lr_decay
        if epoch>duandian and epoch % 10 == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                loss1 = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
                loss2 = criterion(*convert_label_to_similarity(y_hats.asnumpy(), ys.asnumpy()))
                print(loss1, loss2)
                ls = alpha * loss1 + (1-alpha) * loss2
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                 for y_hat, y in zip(y_hats, ys)])
            m += sum([y.size for y in ys])
        time_s = "time %.2f sec" % (time.time() - start)
        train_acc.append(train_acc_sum/n)
        train_loss.append(train_l_sum/n)
        if test_iter is not None:
            test_accu = d2l.evaluate_accuracy(test_iter, net, ctx)
            epoch_s = ("epoch %d, loss %f, train acc %f, test_acc %f, "
                       % (epoch + 1, train_l_sum / n, train_acc_sum / n,
                          test_accu))
            test_acc.append(test_accu)
        else:
            epoch_s = ("epoch %d, loss %f, train acc %f, " %
                       (epoch + 1, train_l_sum / n, train_acc_sum / n))
        print(epoch_s + time_s + ', lr ' + str(trainer.learning_rate))
    unit_split.plot(range(1, num_epochs + 1), train_acc, 'epochs', 'accuracy',
              range(1, num_epochs + 1), test_acc, ['train', 'test'])
    plt.figure()
    unit_split.plot(range(1, num_epochs + 1), train_loss, 'epochs', 'loss', legend=['loss'])
    
'''
# train branch_1    
trainer = gluon.Trainer(net.branch_1.collect_params(), 'adam', {'learning_rate': lr})
train_branch1(net, train_iter, test_iter, 10, lr, ctx, trainer)
# train branch_2  
trainer = gluon.Trainer(net.branch_1.collect_params(), 'adam', {'learning_rate': lr})
train_branch1(net, train_iter, test_iter, 5, lr, ctx, trainer)
# train branch_3  
trainer = gluon.Trainer(net.branch_1.collect_params(), 'adam', {'learning_rate': lr})
train_branch1(net, train_iter, test_iter, 5, lr, ctx, trainer)
# train styleVector  
trainer = gluon.Trainer(net.styleVector.collect_params(), 'adam', {'learning_rate': lr})
train_branch1(net, train_iter, test_iter, 5, lr, ctx, trainer)
# train output  
trainer = gluon.Trainer(net.output.collect_params(), 'adam', {'learning_rate': lr})
train_branch1(net, train_iter, test_iter, 5, lr, ctx, trainer)
'''
# train net
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
utils.train(net, train_iter, test_iter, num_epochs, lr, wd, ctx, lr_period, lr_decay)


'''
# save the model
saved_filename = 'model.params'
net.save_parameters(saved_filename)
'''