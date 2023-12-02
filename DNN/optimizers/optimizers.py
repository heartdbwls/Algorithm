import numpy as np
import cupy as cp

def iterative(sequence, learning_rate=0.01, beta=0):
	for obj in sequence:
		if obj.param > 0:
			obj.weights -= (learning_rate * obj.d_c_w)
			obj.biases -= (learning_rate * obj.d_c_b)


def momentum(sequence, learning_rate=0.1, beta1=0.9, weight_decay=0.0005):  # will have to specify it
	for obj in sequence:
		if obj.param > 0:
			obj.w_m = beta1 * obj.w_m - learning_rate * obj.d_c_w - weight_decay * learning_rate * obj.weights
			obj.weights += obj.w_m
			obj.b_m = beta1 * obj.b_m - learning_rate * obj.d_c_b - weight_decay * learning_rate * obj.biases
			obj.biases += obj.b_m


adamkern = cp.ElementwiseKernel(
		'T grad, float32 one_minus_beta1, float32 one_minus_beta2, float32 epsilon, float32 learning_rate',
		'T param, T m, T v',
		'''	m += one_minus_beta1 * (grad - m);
			v += one_minus_beta2 * (grad * grad - v);
			T mcap = m / one_minus_beta1;
			T vcap = v / one_minus_beta2;
			param -= learning_rate * (mcap / (sqrt(vcap) + epsilon));''',
		'adamkern')


def adam(sequence, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
	for obj in sequence:
		if obj.param > 0:
			# Update weights
			with obj.backp_stream:
				adamkern(obj.d_c_w, 1 - beta1, 1 - beta2, epsilon, learning_rate,
						obj.weights, obj.w_m, obj.w_v)
				# Update biases
				if obj.bias_is_not_0:
					adamkern(obj.d_c_b, 1 - beta1, 1 - beta2, epsilon, learning_rate,
							obj.biases, obj.b_m, obj.b_v)
