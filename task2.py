import numpy as np
import esn

# sliding window parity task
def generate_signal(length, stride, delay):
# INPUT:  w w w w w d d d d d d d d c (stride, delay, current)
# OUTPUT: ? ? ? ? ? ? ? ? ? ? ? ? ? r (r = XOR(all w))
# length >= delay + stride
	input_signal = []
	target_signal = []

	for i in range(length):
		if np.random.random() < 0.5:
			input_signal.append([0.0])
		else:
			input_signal.append([1.0])


	for i in range(length):
		if i < delay+stride-1: # delay+stride-1 test transient steps needed
			target_signal.append([-1.0]) # this part should be skipped in transient
		
		else:
			r = 0.0 # identity element of XOR
			for j in range(i - (delay+stride-1), i - (delay-1)):
				if input_signal[j] == [1.0]: # XORing with True
					r = 1.0 - r          # flips the results
				# else: # dont change

			target_signal.append([r])

	return input_signal, target_signal

e = esn.ESN(1, 1)
e.random_reservoir(size=1000, connectivity=0.01, spectral_radius=0.3)
#e.unit_circle_reservoir(size=1000, spectral_radius=0.01, dense=True)

stride = 4
delay = 3
# roughly when stride+delay >= 8,
# the NRMS-error goes >0.1 with 1000 neurons

print("Stride:", stride, "Delay:", delay)

plotTraining = False
plotTesting = True

# TRAINING
length = 1000
transient = length//10
input_signal, target_signal = generate_signal(length, stride, delay)
e.train(length, transient, input_signal, target_signal, \
	regression='linear', ridge_coefficient=0.0001, noise=None, noise_spread=0.1)
error = e.test(length, transient, input_signal, target_signal, \
	error='NRMSE', plot=plotTraining)
print("training error:", error)

# TESTING - same probability
length = 250
transient = delay+stride-1 # memory necessary
input_signal, target_signal = generate_signal(length, stride, delay)
error = e.test(length, transient, input_signal, target_signal, \
	error='NRMSE', plot=plotTesting)
print("testing error:", error)
