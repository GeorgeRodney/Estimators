import numpy as np
from Estimator import Estimator

# >-----------------------------------------------------
#   Function    :   BlackmanMethod4
#   Method      :   LKF with OOSM handling method 4
#   Source      :   Modern Tracking Systems, Samuel Blackman
#   Author      :   Snater
# >-----------------------------------------------------

class BlackmanMethod4(Estimator):

    def __init__ (self, x, P, R):
        super().__init__(x, P, R)
  
        self.ye = 0
        
        self.He = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        
        self.Qtau = np.eye(self.P.shape[0])

        self.Qe = np.eye(self.P.shape[0])

        self.Pe = np.eye(self.P.shape[0])

        self.Re = np.eye(self.R.shape[0])
        
        self.oosm = False
        

    def predict(self, dt, Q, oosm):
        self.oosm = oosm

        # tau = t_(k) - t_(k-1)
        tau = float(dt)

        # Process noise covariance for time tau
        self.Qtau = Q

        F = np.array([[1, 0, tau, 0],
                      [0, 1, 0, tau],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        if (True == self.oosm):

            # Measurement matrix of delayed observation (10.9)
            self.He = self.H @ F 

            # Expected measurement location (10.9)
            self.ye = self.He @ self.x

            # Process noise covariance for time tau
            self.Qe = ( np.eye(4) - self.K @ self.H ) @ self.Qtau

            # Pe
            self.Pe = self.P - self.Qe

            return

        self.x_ = F @ self.x    
        self.P_ = F @ self.P @ F.T + Q

    def update(self, z, oosm):
        self.oosm = oosm

        if (True == self.oosm):

            # Innovation
            self.y = z - self.ye

            # Re 
            self.Re = self.He @ (self.Qtau - self.Qe.T) @ self.He.T + self.R

            # Kalman gain
            self.S = self.He @ self.Pe @ self.He.T + self.Re

            self.K = self.Pe @ self.He.T @ np.linalg.inv(self.S)

            # Update the State [Mean and Covariance]
            self.x = self.x + self.K @ self.y
            self.P = ( np.eye(4) - self.K @ self.He ) @ self.P + self.K @ self.He @ self.Qe.T

            # Rest oosm flag
            self.oosm = False

        elif (False == self.oosm):

            # Innovation
            self.y = z - self.H @ self.x_

            # Gain Calcs
            self.S = self.H @ self.P_ @ self.H.T + self.R
            self.K = self.P_ @ self.H.T @ np.linalg.inv(self.S)

            # Update the State [Mean and Covariance]
            self.x = self.x_ + self.K @ self.y

            I = np.eye(self.P.shape[0])
            self.P = (I - self.K @ self.H) @ self.P_ @ (I - self.K @ self.H).T + self.K @ self.R @ self.K.T

        else:
            raise ValueError('Something broke in update.')
        
    def get_predP(self):

        if (True == self.oosm):
            return self.Pe
        
        else:
            return self.P_