import numpy as np

def momentum(sequence, learning_rate=0.1, beta1=0.9, weight_decay=0.0005):  # will have to specify it
	for obj in sequence:
		if obj.param > 0:
			obj.w_m = beta1 * obj.w_m - learning_rate * obj.d_c_w - weight_decay * learning_rate * obj.weights
			obj.weights += obj.w_m
			obj.b_m = beta1 * obj.b_m - learning_rate * obj.d_c_b - weight_decay * learning_rate * obj.biases
			obj.biases += obj.b_m
			
def adamkern(GRAD, ONE_MINUS_BETA1, ONE_MINUS_BETA2, EPSILON, LEARNING_RATE, PARAM, M, V):
    M += ONE_MINUS_BETA1 * (GRAD - M)
    V += ONE_MINUS_BETA2 * (GRAD * GRAD - V)
    MCAP = M / ONE_MINUS_BETA1
    VCAP = V / ONE_MINUS_BETA2
    PARAM -= LEARNING_RATE * (MCAP / (np.sqrt(VCAP) + EPSILON))

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