from .base_layer import *

class BatchNormalization(Layer):
    def __init__(
            self,
            momentum=0.9,
            epsilon=1e-10,
            name=None
    ):
        saved_locals = locals()
        super().__init__(saved_locals)

    def do_init(self, kwargs):
        input_shape = self.get_inp_shape()
        self.shape = (None, *input_shape)
        self.batches = 1
        self.inp_shape = (self.batches, *input_shape)
        self.biases = np.zeros(input_shape, dtype=self.dtype)  # biases is beta
        self.weights = np.ones(input_shape, dtype=self.dtype)  # weights is gamma
        self.gamma = self.weights
        self.beta = self.biases
        self.kernels = self.weights
        self.w_m = np.zeros_like(self.weights, dtype=self.dtype)
        self.w_v = np.zeros_like(self.weights, dtype=self.dtype)
        self.b_m = np.zeros_like(self.biases, dtype=self.dtype)
        self.b_v = np.zeros_like(self.biases, dtype=self.dtype)
        self.epsilon = kwargs.get('epsilon')
        self.momentum = kwargs.get('momentum')
        self.moving_mean = None
        self.moving_var = None
        self.param = 4 * input_shape[-1]
        self.activation = echo

    def forward(self, inp, training=True):
        self.inp_shape = inp.shape
        if training:
            mean = inp.mean(axis=0)
            self.xmu = inp - mean
            var = (self.xmu ** 2).mean(axis=0)
            self.ivar = 1 / (var + self.epsilon)
            self.istd = np.sqrt(self.ivar)
            self.xnorm = self.xmu * self.istd
            with self.backp_stream:
                self.backp_stream.wait_event(self.grad_event)
                if self.moving_mean is None:
                    self.moving_mean = mean
                    self.moving_var = var
                else:
                    self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
                    self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * var
        else:
            if self.moving_mean is None:
                mean = inp.mean(axis=0)
                self.xmu = inp - mean
                var = (self.xmu ** 2).mean(axis=0)
                self.ivar = 1 / (var + self.epsilon)
                self.istd = np.sqrt(self.ivar)
                self.moving_mean = mean
                self.moving_var = var
                self.xnorm = self.xmu * self.istd
            else:
                self.xmu = inp - self.moving_mean
                self.ivar = 1 / (self.moving_var + self.epsilon)
                self.istd = np.sqrt(self.ivar)
                self.xnorm = self.xmu * self.istd
        return self.xnorm * self.weights + self.biases

    def backprop(self, grads, do_d_inp=True):
        batches = self.inp_shape[0]
        if batches != self.batches:
            self.batches = batches

        self.d_c_b = grads.sum(axis=0)

        with self.backp_stream:
            self.backp_stream.wait_event(self.grad_event)
            self.d_c_w = (self.xnorm * grads).sum(axis=0)

        d_inp = self.istd * self.weights * (
                self.batches * grads - self.d_c_b - self.xmu * self.ivar * ((grads * self.xmu).sum(axis=0)))
        return d_inp
