import numpy as np
from Estimator import Estimator

# >-----------------------------------------------------
#   Function    :   Simon
#   Method      :   LKF with Dan Simon method of OOSM handling
#   Author      :   Snater
# >-----------------------------------------------------

# NOT EVEN STARTED

class Simon(Estimator):

    def __init__ (self, x, P, R):
        super().__init__(x, P, R)

        self.y = 0
        self.oosm = False
        self.Pw = np.eye(self.P.shape[0])
        self.Pxw = np.eye(self.P.shape[0])
        self.retroP = np.eye(self.P.shape[0])
        
    def predict(self, dt, Q, oosm):
        self.oosm = oosm

        t = float(dt)
        F = np.array([[1, 0, t, 0],
                      [0, 1, 0, t],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        # Retrodict from time k to k0
        if (True == oosm):
            # Retrodict and remove the previous residual info
            self.S = self.H @ self.P_ @ self.H.T + self.R
            self.x_ = F @ ( self.x - Q @ self.H.T @ np.linalg.inv(self.S) @ self.y) # Equation (10.118)

            # Compute the covariance of the retrodicted state
            self.Pw = Q - Q @ self.H.T @ np.linalg.inv(self.S) @ self.H @ Q # Equations (10.122)
            self.Pxw = Q - self.P_ @ self.H.T @ np.linalg.inv(self.S) @ self.H @ Q # Equations (10.122)
            return 
        
        # Predict
        self.x_ = F @ self.x
        self.P_ = F @ self.P @ F.T + Q
            
    def update(self, z, oosm):
        self.oosm = oosm

        if (False == self.didAssociate):

            if (True == self.oosm):
                return
            
            self.x = self.x_
            self.P = self.P_
            return

        # Innovation
        self.y = z - self.H @ self.x_

        # Gain Calcs
        self.K = self.P_ @ self.H.T @ np.linalg.inv(self.S)

        # Update the State [Mean and Covariance]
        self.x = self.x_ + self.K @ self.y

        I = np.eye(self.P.shape[0])
        self.P = (I - self.K @ self.H) @ self.P_ @ (I - self.K @ self.H).T + self.K @ self.R @ self.K.T