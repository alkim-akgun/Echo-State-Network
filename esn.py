import numpy as np
import matplotlib.pyplot as plt

_NP_RANDOM_SEED = 0
np.random.seed(_NP_RANDOM_SEED)

class ESN(object):

	def __init__(self, input_size, readout_size, bias=1.0, leak=1.0, activation=np.tanh):

		self.input_size = input_size + 1 # +1 bias
		self.readout_size = readout_size

		self.reservoir_weights = None

		self.bias = bias
		self.leak = leak
		self.activation = activation

	# RESERVOIR
	# Utility

	def make_sparse(self, connectivity):
		for i in range(self.reservoir_size()):
			for j in range(self.reservoir_size()):
				if np.random.rand() >= connectivity:
					self.reservoir_weights[i][j] = 0.0
			
	def reservoir_size(self):
		return self.reservoir_weights.shape[0]

	def max_abs_eigval(self):
		abs_eigvals = [np.abs(eig_val) for eig_val \
									in np.linalg.eigvals(self.reservoir_weights)]
		return np.amax(abs_eigvals)

	def set_spectral_radius(self, spectral_radius):
		if spectral_radius != None:
			self.reservoir_weights *= (spectral_radius / self.max_abs_eigval())

	# Main
	
	def _post_reservoir_generation(self): # call after every reservoir generation
		# set initial state
		self.state = np.zeros((self.reservoir_size(), 1))
		# connect layers
		ressize = self.reservoir_size()
		self.input_weights = np.random.rand(ressize, self.input_size)
		if self.readout_size > 0: # if there will be read-out
			self.readout_weights = np.random.rand(self.readout_size, \
											  	  self.input_size+ressize)

	def random_reservoir(self, size, connectivity, spectral_radius=None):
		self.reservoir_weights = np.random.rand(size, size) * 10
		self.make_sparse(connectivity)
		self.set_spectral_radius(spectral_radius) # handles None
		self._post_reservoir_generation()

	def unit_circle_reservoir(self, size, dense=False, spectral_radius=None):
		# generates a reservoir whose eigen-values are conjugate pairs
		# size must be even
		size += (size % 2) # if odd, add 1
		n = size
		# angle difference between eigenvaues
		diff = 2.0*(np.pi)/n
		# offset = diff / 2.0 # can be used to avoid 1 as an eigen-value
		offset = 0 
		# first one is offset, last one is np.pi-offset
		# angles for the eigenvalues on the upper half of the unit circle
		theta_up = [t*diff + offset for t in range(n//2)] # a + ib
		eigval_up_real = [np.cos(t) for t in theta_up]
		eigval_up_img = [np.sin(t) for t in theta_up]

		block_matrix = np.array([[0.0 for j in range(n)] for i in range(n)])
		for i in range(n//2):
			k = 2*i
			block_matrix[k][k] = eigval_up_real[i]
			block_matrix[k][k+1] = (-1)*eigval_up_img[i]
			block_matrix[k+1][k] = eigval_up_img[i]
			block_matrix[k+1][k+1] = eigval_up_real[i]
		weight_matrix = None
		# make dense: get rid of all of the zeros in the matrix
		if dense:
		# multiplay with a real matrix from left and its inverse from right
			random_matrix = np.random.rand(n, n)
			while np.linalg.det(random_matrix) == 0.0:
				random_matrix = np.random.rand(n, n)
			random_matrix_inv = np.linalg.inv(random_matrix)
			weight_matrix = np.matmul(np.matmul(random_matrix, block_matrix), \
										random_matrix_inv)
		else:
			weight_matrix = block_matrix

		self.reservoir_weights = weight_matrix
		# if the spectral radius is non-None, it's not a unit circle anymore
		self.set_spectral_radius(spectral_radius) # handles None
		self._post_reservoir_generation()

	
	# OPERATION
	# Utility

	def _state_update(self):
		if not np.any(self.reservoir_weights):
			# if reservoir weight matrix is all zeros
			state_weighted = 0
		else:
			state_weighted = self.reservoir_weights @ self.state
		input_weighted = self.input_weights @ self.input
		self.state = (1-self.leak)*self.state + self.leak*self.activation(state_weighted + input_weighted)

	def _read_out(self):
		if self.readout_size > 0:
			self.readout = self.readout_weights @ self.extended_state()

	# Main

	def inject_input(self, input_data):
		input_data = input_data.copy()
		input_data.insert(0, self.bias) # add bias
		self.input = np.array([input_data]).T
	
	def iterate_without_readout(self, input_data): # during training, there is no need to read out
		self.inject_input(input_data)
		self._state_update()

	def iterate(self, input_data):
		self.iterate_without_readout(input_data)
		self._read_out()

	def run(self, input_signal):
		for input_data in input_signal:
			self.iterate(input_data)

	def extended_state(self):
		return np.concatenate((self.input, self.state), axis=0)

	def get_readout_weights(self):
		return self.readout.copy()

	def train(self, length, transient, input_signal, target_signal, \
			  regression='linear', ridge_coefficient=0.1, noise=None, \
			  noise_spread = 0.01):

		extended_state_collection = []

		# RUN
		for iteration in range(length):
			input_data = input_signal[iteration]
			self.iterate_without_readout(input_data)
			target_data = target_signal[iteration]

			if iteration >= transient:
				extended_state_collection.append(self.extended_state().T[0])

		# READ-OUT weights calculation
		# list to array
		target_signal = np.array(target_signal)

		# NOISE
		noise = None if noise==None else noise.lower()
		if not noise in [None, 'none']:
			# introduce noise
			shape = (length, self.readout_size)
			if not noise == 'uniform': # gaussian noise is default
				# spread here is the standard deviation for the normal distribution
				noise = np.random.normal(0.0, noise_spread, shape)
			else:
				# spread here is the max. magnitude of each noise component
				noise = np.random.random(shape) * noise_spread
			target_signal += noise

		esc = np.array(extended_state_collection)
		if not regression == 'ridge': # linear regression is default
			# linear regression with pseudo-inverse
			self.readout_weights = (np.linalg.pinv(esc) \
									@ target_signal[transient:length] \
								   ).T
		else:   # ridge regression
			self.readout_weights = \
				target_signal[transient:length].T @ esc \
				@ np.linalg.inv(esc.T @ esc + ridge_coefficient \
				* np.eye(self.input_size+self.reservoir_size()))


	def test(self, length, transient, input_signal, target_signal, \
			 error='NRMSE', plot=True):

		readout_collection = []
		# SE := squared error
		SE = [0.0 for i in range(self.readout_size)]
		# RUN
		for iteration in range(length):
			input_data = input_signal[iteration]
			self.iterate(input_data)
			target_data = target_signal[iteration]

			if iteration >= transient:
				if plot:
					readout_collection.append(self.readout.T[0])
				for i in range(self.readout_size):
					#print(iteration, i, self.readout[i][0], target_data[i])
					SE[i] += ((self.readout[i][0] - target_data[i])**2)

		# PLOT
		if plot:
			readout_collection = np.array(readout_collection)
			# plot for each readout node
			time = list(range(length-transient))
			for i in range(self.readout_size):
				readout_sequence = [readout_data[i] for readout_data \
													in readout_collection]
				target_sequence = [target_data[i] for target_data \
												in target_signal[transient:length]]
				plt.title("Readout Node " + str(i))
				plt.plot(time, readout_sequence, time, target_sequence)
				plt.show()

		# ERROR
		# sum/readout_size averages over the readout nodes
		error = error.upper()
		# MSE := mean-square error
		MSE = [SE_d / length for SE_d in SE]
		if not error.endswith('RMSE'):
			return 	sum(MSE)/self.readout_size # MSE was the last valid option
		else:
			# RMSE := root mean-square error
			RMSE = [MSE_d**0.5 for MSE_d in MSE]

		if not error == 'NRMSE':
			return sum(RMSE)/self.readout_size # RMSE was the last valid option
		else:
			# NRMSE := normalized root mean-square error
			target_signal_T = np.array(target_signal).T # _T is Dimension x Length
			# divide error by the variance of the target signal for each readout node
			NRMSE = [RMSE[d]/np.var(target_signal_T[d]) for d \
														in range(self.readout_size)]
			return (sum(NRMSE)/self.readout_size) # NRMSE