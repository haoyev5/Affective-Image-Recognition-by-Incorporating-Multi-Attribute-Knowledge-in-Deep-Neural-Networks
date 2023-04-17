# -*- coding: utf-8 -*-

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
import d2lzh as d2l
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import unit_split


def confusion_matrix(test_iter, net, ctx, loss):
    """
    生成混淆矩阵
    :param test_iter:
    :param net:
    :param ctx:
    :param loss:
    :return:
        0   1
    0   X   X
    1   X   X
    """
    test_acc_sum = 0.0
    test_n = 0
    test_l_sum = 0.0
    m = nd.zeros(shape=(2, 2), ctx=ctx)
    for X, y in test_iter:
        X = X.as_in_context(ctx)
        y = y.as_in_context(ctx)
        y_hat = net(X)
        y_ = y_hat.argmax(axis=1)
        test_acc_sum += (y_ == y).sum().asscalar()
        test_n += y.size
        test_l_sum += loss(y_hat, y).sum().asscalar()
        for i in range(y.size):
            if y[i] == y_[i]:
                if y[i] == 1:
                    m[1][1] += 1
                else:
                    m[0][0] += 1
            else:
                if y[i] == 0:
                    m[0][1] += 1
                else:
                    m[1][0] += 1
    return m, test_acc_sum, test_n, test_l_sum

############################################################################
loss = gloss.SoftmaxCrossEntropyLoss()

def VoteLoss1(y_hat, y, v=5):
    p = nd.pick(y_hat, y)
    p = nd.square(p-1)
    p = v*p
    y_hat[y] = p
    print(y_hat)
    l = nd.square(y_hat)
    return nd.sum(l, axis=0)


def VoteLoss(y_hat, y, v=5):
    p = nd.pick(y_hat, y)
    loss = nd.square(p-0.6)
    #loss = v*p
    return nd.sum(loss, axis=0)
############################################################################

def Loss(y_hat, y, alpha=0.1):
    vote_l = VoteLoss(y_hat, y)
    loss1 = gloss.SoftmaxCrossEntropyLoss()
    entropy_l = loss1(y_hat, y)
    return entropy_l + alpha*vote_l 

def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])

def train(net, train_iter, test_iter, num_epochs, lr, wd, ctx, lr_period, lr_decay):
    print('training on', ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    train_acc, test_acc = [], []
    train_loss = []
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
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
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