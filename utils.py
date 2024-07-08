import mlx.core as mx
import mlx.nn as nn


def loss_and_acc(model, X, y):
    output = model(X)
    loss = mx.mean(nn.losses.nll_loss(output, y))
    acc = mx.sum(mx.argmax(output, axis=1) == y)
    return loss, acc
