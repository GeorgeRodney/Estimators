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

        self.oosm = False
        self.Pw = np.eye(self.P.shape[0])
        self.Pxw = np.eye(self.P.shape[0])
        self.retroP = np.eye(self.P.shape[0])
        self.retroS = np.eye(self.S.shape[0])
        self.retroPxy = np.eye(self.P.shape[0])
        self.retroK = np.eye(self.K.shape[0])
        
    def predict(self, dt, Q, oosm):
        self.oosm = oosm

        t = float(dt)
        F = np.array([[1, 0, t, 0],
                      [0, 1, 0, t],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        # print(oosm)

        # Retrodict from time k to k0
        if (True == self.oosm):
            # Retrodict and remove the previous residual info
            self.S = self.H @ self.P_ @ self.H.T + self.R
            self.x_ = F @ ( self.x - Q @ self.H.T @ np.linalg.inv(self.S) @ self.y) # Equation (10.118 / 10.128)

            # Compute the covariance of the retrodicted state
            self.Pw = Q - Q @ self.H.T @ np.linalg.inv(self.S) @ self.H @ Q # Equations (10.122)
            self.Pxw = Q - self.P_ @ self.H.T @ np.linalg.inv(self.S) @ self.H @ Q # Equations (10.122)
            self.retroP = F @ ( self.P - self.Pxw - self.Pxw.T + self.Pw ) @ F.T # Equation (10.129)

            # Compute the innovation covariance of the retrodicted measurement
            self.retroS = self.H @ self.retroP @ self.H.T + self.R # Equation (10.130)

            # Compute the covariance between state at time k and measurement at time k0
            self.retroPxy = ( self.P - self.Pxw ) @ F.T @ self.H.T # Equation (10.131)

        else:
            # Predict
            self.x_ = F @ self.x
            self.P_ = F @ self.P @ F.T + Q
            self.S = self.H @ self.P_ @ self.H.T + self.R
            
    def update(self, z, oosm):
        self.oosm = oosm

        if (False == self.didAssociate):
            print("DidntAssociate")
            if (True == self.oosm):
                return
            
            self.x = self.x_
            self.P = self.P_
            return

        if (True == self.oosm):
            
            # Compute updated state x^{^}(k | k0)
            self.x = self.x + self.retroPxy @ np.linalg.inv(self.retroS) @ ( z - self.H @ self.x_ ) # Equations (10.132)
            self.P = self.P - self.retroPxy @ np.linalg.inv(self.retroS) @ self.retroPxy.T # Equations (10.132)

            # For performance reasons I want the retrodicted innovation kalman gain K
            # The value is implicit in the above simon equations
            self.retroK = self.retroPxy @ np.linalg.inv(self.retroS)

        else:
            # Innovation
            self.y = z - self.H @ self.x_
            # print(np.linalg.norm(self.y))

            # Gain Calcs
            self.K = self.P_ @ self.H.T @ np.linalg.inv(self.S)

            # Update the State [Mean and Covariance]
            self.x = self.x_ + self.K @ self.y

            I = np.eye(self.P.shape[0])
            self.P = (I - self.K @ self.H) @ self.P_ @ (I - self.K @ self.H).T + self.K @ self.R @ self.K.T

    # Override baseline getters
    def get_predP(self):

        if (True == self.oosm):
            return self.retroP
        
        else:
            return self.P_
        
    def get_K(self):
        
        if (True == self.oosm):
            return self.retroK
        
        else:
            return self.K
        
    def get_S(self):

        if (True == self.oosm):
            return self.retroS
        
        else:
            return self.S