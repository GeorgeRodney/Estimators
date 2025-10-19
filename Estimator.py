import numpy as np
from abc import ABC, abstractmethod

# >-----------------------------------------------------
#   Function    :   Estimator
#   Method      :   Linear Kalman Filter (LKF)  
#   Author      :   Snater
# >-----------------------------------------------------

class Estimator:

    def __init__ (self, x, P):
        self.x_ = x # predicted
        self.P_ = P # predicted 
        self.x = x # estimate
        self.P = P # estimate
        self.K = np.array([[1, 0],
                           [0, 1],
                           [1, 0],
                           [0, 1]])
        self.S = np.array([[1, 0],
                           [0, 1]])
        
        self.y = 0

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

    def predict(self, dt, Q, oosm):
        t = float(dt)

        F = np.array([[1, 0, t, 0],
                      [0, 1, 0, t],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        self.x_ = F @ self.x
        self.P_ = F @ self.P @ F.T + Q

    def update(self, z, R, oosm):

        # Innovation
        self.y = z -  self.H @ self.x_

        # Statistics
        self.S = self.H @ self.P_ @ self.H.T + R
        self.K = self.P_ @ self.H.T @ np.linalg.inv(self.S)

        # Update the State [Mean and Covariance]
        self.x = self.x_ + self.K @ self.y

        I = np.eye(self.P.shape[0])
        self.P = (I - self.K @ self.H) @ self.P_ @ (I - self.K @ self.H).T + self.K @ R @ self.K.T

    def get_estState(self):
        return self.x
    
    def get_predState(self):
        return self.x_

    def get_estP(self):
        return self.P
    
    def get_predP(self):
        return self.P_
    
    def get_K(self):
        return self.K
    
    def get_S(self):
        return self.S
    
    def get_y(self):
        return self.y