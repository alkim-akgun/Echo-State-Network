import numpy as np
import esn

# alternating signals classification task
def generate_signal(length, probability, modulus):
	input_signal = []
	target_signal = []

	x = np.random.randint(modulus) # initial signal value
	signal_type = +1.0 # initial signal type
	for i in range(length):
		# switch signal type with probability p
		if np.random.random() < probability:
			signal_type *= -1 # invert

		x = (x + signal_type) % modulus
		target_signal.append([signal_type, -signal_type])
		input_signal.append([x])

	return input_signal, target_signal

e = esn.ESN(1, 2)
#e.random_reservoir(size=100, connectivity=0.05, spectral_radius=0.3)
e.unit_circle_reservoir(size=100, spectral_radius=0.05, dense=True)

modulus = 5
probability = 0.2 # default

plot = False

# TRAINING
length = 1000
transient = length//10
input_signal, target_signal = generate_signal(length, probability, modulus)
e.train(length, transient, input_signal, target_signal, \
	regression='ridge', ridge_coefficient=0.0001, noise='gaussian', noise_spread=0.1)
error = e.test(length, transient, input_signal, target_signal, error='NRMSE', plot=plot)
print("training error:", error)

# EVALUATION - same probability
length = 2000
transient = 1 # memory necessary
input_signal, target_signal = generate_signal(length, probability, modulus)
error = e.test(length, transient, input_signal, target_signal, \
	error='NRMSE', plot=plot)
print("evaluation error:", error)

# EVALUATION 2 - slower switching
length = 2000
transient = 1 # memory necessary
probability = 0.1 # different switching probability
input_signal, target_signal = generate_signal(length, probability, modulus)
error = e.test(length, transient, input_signal, target_signal, \
	error='NRMSE', plot=plot)
print("evaluation 2 error:", error)

# EVALUATION 3 - slightly faster switching
length = 2000
transient = 1 # memory necessary
probability = 0.3 # different switching probability
input_signal, target_signal = generate_signal(length, probability, modulus)
error = e.test(length, transient, input_signal, target_signal, \
	error='NRMSE', plot=plot)
print("evaluation 3 error:", error)

# EVALUATION 4 - faster switching
length = 2000
transient = 1 # memory necessary
probability = 0.7 # different switching probability
input_signal, target_signal = generate_signal(length, probability, modulus)
error = e.test(length, transient, input_signal, target_signal, \
	error='NRMSE', plot=plot)
print("evaluation 3 error:", error)
