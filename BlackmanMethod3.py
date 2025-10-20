import numpy as np
from Estimator import Estimator

# >-----------------------------------------------------
#   Function    :   BlackmanMethod3
#   Method      :   LKF with OOSM handling method 3
#   Source      :   Modern Tracking Systems, Samuel Blackman
#   Author      :   Snater
# >-----------------------------------------------------

class BlackmanMethod3(Estimator):

    def __init__ (self, x, P, R):
        super().__init__(x, P, R)
        self.y = 0
        
        self.He = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        
        self.oosm = False
        

    def predict(self, dt, Q, oosm):
        self.oosm = oosm

        t = float(dt)
        F = np.array([[1, 0, t, 0],
                      [0, 1, 0, t],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        
        self.x_ = F @ self.x
        self.P_ = F @ self.P @ F.T + Q

        if (True == self.oosm):
            self.He = self.H @ F
            self.S = self.He @ self.P_ @ self.He.T + self.R

        elif (False == self.oosm):
            self.S = self.H @ self.P_ @ self.H.T + self.R

    def update(self, z, oosm):
        self.oosm = oosm

        if (False == self.didAssociate):

            if (True == self.oosm):
                return
            
            self.x = self.x_
            self.P = self.P_
            return

        if (True == self.oosm):
            # Innovation
            self.y = z - self.He @ self.x

            # Gain calcs
            self.K = self.P_ @ self.He.T @ np.linalg.inv(self.S)

            # Update the State [Mean and Covariance]
            self.x = self.x + self.K @ self.y

            I = np.eye(self.P.shape[0])
            self.P = (I - self.K @ self.He) @ self.P_ @ (I - self.K @ self.He).T + self.K @ self.R @ self.K.T

            # Rest oosm flag
            self.oosm = False

        elif (False == self.oosm):

            # Innovation
            self.y = z - self.H @ self.x_

            # Gain Calcs
            self.K = self.P_ @ self.H.T @ np.linalg.inv(self.S)

            # Update the State [Mean and Covariance]
            self.x = self.x_ + self.K @ self.y

            I = np.eye(self.P.shape[0])
            self.P = (I - self.K @ self.H) @ self.P_ @ (I - self.K @ self.H).T + self.K @ self.R @ self.K.T

        else:
            raise ValueError('Something broke in update.')