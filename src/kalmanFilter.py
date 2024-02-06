#################### Import Section of the code #############################

try:
	import numpy as np	
except Exception as e:
	print(e,"\nPlease Install the package")

#################### Import Section ends here ################################


class KalmanFilter(object):
	"""docstring for KalmanFilter"""

	def __init__(self, dt=1,stateVariance=0,measurementVariance=0, 
														method="Velocity", initial_state = [[0],[1],[0],[1]]):
		super(KalmanFilter, self).__init__()
		self.method = method
		self.stateVariance = stateVariance
		self.measurementVariance = measurementVariance
		self.dt = dt
		self.initModel(initial_state=initial_state)
	
	"""init function to initialise the model"""
	def initModel(self, initial_state=[[0],[1],[0],[1]]): 
		if self.method == "Accerelation":
			self.U = 1
		else: 
			self.U = 0
		self.A = np.matrix( [[1 ,self.dt, 0, 0], [0, 1, 0, 0], 
										[0, 0, 1, self.dt],  [0, 0, 0, 1]] )

		self.B = np.matrix( [[self.dt**2/2], [self.dt], [self.dt**2/2], 
																[self.dt]] )
		
		self.H = np.matrix( [[1,0,0,0], [0,0,1,0]] ) 
		self.P = np.matrix(self.stateVariance*np.identity(self.A.shape[0]))
		self.R = np.matrix(self.measurementVariance*np.identity(
															self.H.shape[0]))
		
		self.Q = np.matrix( [[self.dt**4/4 ,self.dt**3/2, 0, 0], 
							[self.dt**3/2, self.dt**2, 0, 0], 
							[0, 0, self.dt**4/4 ,self.dt**3/2],
							[0, 0, self.dt**3/2,self.dt**2]])
		
		self.erroCov = self.P
		self.state = np.matrix(initial_state)


	"""Predict function which predicst next state based on previous state"""
	def predict(self):
		self.predictedState = self.A*self.state + self.B*self.U
		print(f"Predicted state: \n{self.predictedState}")
		self.predictedErrorCov = self.A*self.erroCov*self.A.T + self.Q
		print(f"predictedErrorCov: \n{self.predictedErrorCov}")
		temp = np.asarray(self.predictedState)

		return temp[0], temp[2]

	"""Correct function which correct the states based on measurements"""
	def correct(self, currentMeasurement):
		self.kalmanGain = self.predictedErrorCov*self.H.T*np.linalg.pinv(
								self.H*self.predictedErrorCov*self.H.T+self.R)
		print(f"kalmanGain: \n{self.kalmanGain}")
		self.state = self.predictedState + self.kalmanGain*(currentMeasurement
											   - (self.H*self.predictedState))
		print(f"state: \n{self.state}")

		self.erroCov = (np.identity(self.P.shape[0]) - 
								self.kalmanGain*self.H)*self.predictedErrorCov
		print(f"erroCov: \n{self.erroCov}")
