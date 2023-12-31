from NN.autograd import Tensor
import numpy as np


class Adam:
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps

        self.m = [np.zeros_like(param.data) for param in self.params]
        self.v = [np.zeros_like(param.data) for param in self.params]

        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * param.grad ** 2

            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.params:
            param.grad = None if param.grad is None else np.zeros_like(param.grad) # ??? 0 or None



class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            
            param.data -= self.lr * param.grad


    def zero_grad(self):
        for param in self.params:
            param.grad = None if param.grad is None else np.zeros_like(param.grad) # ??? 0 or None



class Momentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum

        self.m = [np.zeros_like(param.data) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            self.m[i] = self.momentum * self.m[i] + (1 - self.momentum) * param.grad
            param.data -= self.m[i] * self.lr

    def zero_grad(self):
        for param in self.params:
            param.grad = None if param.grad is None else np.zeros_like(param.grad)
