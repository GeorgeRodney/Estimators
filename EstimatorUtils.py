
from enum import Enum, auto

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

# def threeSigmaCheck(z, x_ S):
#     # Stefan if you could implement a mahalanobis distance check and then return true if 
#     # within 3 sigma.Then verify that the state coasts when this is false.
#     withinGate = True
#     return withinGate
