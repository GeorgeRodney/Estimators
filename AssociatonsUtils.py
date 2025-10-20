from enum import Enum

class AssociationState(Enum):
    CLOSED      = 0
    OPEN        = 1
    CONVERGED   = 2

class AssociationObject():

    def __init__(self, id, x, p):
        self.id_    = id
        # location
        self.x_     = x
        # covariance
        self.p_     = p
        self.state_ = AssociationState.OPEN


    # SETTERS/UPDATERS

    # function: updateState
    # inputs: state
    def updateState(self, state):
        self.state_ = state

    # function: updateLocation
    # inputs: newLocation
    def updateLocation(self, newX):
        self.x_ = newX

    # function: updateCovariance
    # inputs: newCovariance
    def updateCovariance(self, newP):
        self.p_ = newP
