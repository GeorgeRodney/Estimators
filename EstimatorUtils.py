import numpy as np
from enum import Enum, auto

ASSOCIATION_THRESHOLD = 10

# Function: isAssociation
#   - Uses mahalanobis distance to determine if an association was made or not
# Inputs:
#   - x: a vector location of point 1
#   - y: a vector location of point 2
#   - p: semi-definite covariance matrix
def isAssociation(x, y, p):
    d = np.sqrt((x - y).T @ np.linalg.inv(p) @ (x - y))
    return d[0] < ASSOCIATION_THRESHOLD, d[0]

def move_element(a, i, j):
    a = list(a)
    elem = a.pop(i)
    a.insert(j, elem)
    return a

def convertToOOSM(obs):
    obs_OOSM = list(obs)
    n = len(obs_OOSM)
    interval = 5

    for idx in range(0, n, interval):
        if idx > 0:
            obs_OOSM = move_element(obs_OOSM, idx, idx - 1)
    return obs_OOSM

class FilterMethod(Enum):
    BASELINE = auto()
    BLACKMAN2 = auto()
    BLACKMAN3 = auto()
    BLACKMAN4 = auto()
    SIMON = auto()
    CUSTOM = auto()

class SequenceMethod(Enum):
    NOOOSM = auto()
    OOSM = auto()
