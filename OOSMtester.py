import numpy as np
import matplotlib.pyplot as plt
import Estimator as est
import EstimatorUtils as estUtils
import BlackmanMethod3 as bm3
import BlackmanMethod4 as bm4
import XYStepper as vid
import Simon as si

# Number of samples!
N = 45

# Process noise function
def Q_dt(t, sigma_a):
    dt = np.abs(float(t))
    dt2 = dt*dt
    dt3 = dt2*dt
    dt4 = dt3*dt
    q = sigma_a
    return (q/dt)*np.array([[dt4/4, 0,     dt3/2, 0    ],
                       [0,     dt4/4, 0,     dt3/2],
                       [dt3/2, 0,     dt2,   0    ],
                       [0,     dt3/2, 0,     dt2  ]], dtype=float)

# Generate Initial Truth StateBLACKMAN3
# Constant Velocity Model
# p0 p1 v0 v1
x_0 = np.array([[32],[4],[0.1],[0.3]])

# Degrees of freedom
DOF = 4.0

# Generate observations timeline
T = 0.1
t = np.arange(N) * T
sigma = 0.1
mean = np.array([[0], [0]])
obs = []
truth = []
accelStd = 1

# SEED
seed = 42
rng = np.random.default_rng()
# rng2 = np.random.default_rng(seed+1)
accelOn = False

# Store truth
# Store observations
for o in range(len(t)):

    # a = rng2.normal(0, accelStd,1)

    F = np.array([[1, 0, t[o], 0],
                  [0, 1, 0, t[o]],
                  [0, 0, 1, 0], 
                  [0, 0, 0, 1]])
    G = np.array([  [(T**2)/2],
                    [(T**2)/2],
                    [T],
                    [T]])

    x = F @ x_0
    truth.append(x)
    z = x[:2] + np.array([rng.normal(0, sigma, 1), rng.normal(0, sigma, 1)])
    # z = x[:2] + rng.normal(mean, sigma, (2,1))
    zt = np.vstack((z, [[t[o]]]))
    obs.append(zt)


# print(obs)
# Define the initial state vector
x0 = obs[0]
x0 = x0[:2]
x_0 = np.vstack((x0, [[0],[0]]))

# Popluate estimation statistics
R = np.array([[sigma**2, 0],
              [0, sigma**2]])

beta = 10
P00 = beta * R[0][0]
P11 = 0.5**2
P_0 = np.array([[P00, 0, 0, 0],
               [0, P00, 0, 0],
               [0, 0, P11, 0], 
               [0, 0, 0, P11]])

# Tuned
alphaSq = 0.00000057

t0 = 0
t_prev = 0
t_current = 0

isOOSM = False
prevOOSM = False

Nees = []
Nis = []
estimated_P00 = []
predicted_P00 = []
pos_K = []
vel_K = []
truPos = []
measPos = []
measTime = []
estPos = []
predicted_P00.append(P00)
estimated_P00.append(P00)
oosmStamp = []
oosmStamp.append(0)

# STEP 1: Select Estimator Method
# Define the Estimator (BASELINE, BLACKMAN3, BLACKMAN4, SIMON)
state = estUtils.FilterMethod.BASELINE
state = estUtils.FilterMethod.BLACKMAN3
state = estUtils.FilterMethod.SIMON

# STEP 3: Select IN SEQUENE or OUT OF SEQUENCE
# Define the sequence method (NOOOSM, OOSM)
doOOSM = estUtils.SequenceMethod.NOOOSM
doOOSM = estUtils.SequenceMethod.OOSM

# Instantiate the Estimator
if (estUtils.FilterMethod.BASELINE == state):
    estimator_ = est.Estimator(x_0, P_0, R)

elif (estUtils.FilterMethod.BLACKMAN3 == state):
    estimator_ = bm3.BlackmanMethod3(x_0, P_0, R)

elif (estUtils.FilterMethod.SIMON == state):
    estimator_ = si.Simon(x_0, P_0, R)

if (estUtils.SequenceMethod.OOSM == doOOSM):
    obs = estUtils.convertToOOSM(obs)

# print(obs)

for ii in range(int(N)-1):

    idxIgnore0 = ii + 1
    # Pull time and location apart
    z = obs[idxIgnore0]
    z = z[:2]
    
    t_current = float(obs[idxIgnore0][2,0])
    dt = t_current - t_prev

    Q = Q_dt(np.abs(dt), alphaSq)

    if (dt < 0):
        isOOSM = True
    else: 
        isOOSM = False

    estimator_.predict(dt, Q, isOOSM)
    # estimator_.associate(z)
    estimator_.update(z, isOOSM)

    # Save off current time
    if (False == isOOSM):
        t_prev = t_current

    # Log Performance Data
    P = estimator_.get_estP()
    P_ = estimator_.get_predP()
    K = estimator_.get_K()
    S = estimator_.get_S()
    x_ = estimator_.get_predState()
    t = truth[idxIgnore0]
    # k_true = int(round(t_current / T))
    # t = truth[k_true]
    x = estimator_.get_estState()
    estimated_P00.append(P[0][0])
    predicted_P00.append(P_[0][0])
    # Calculate NEES
    e = t - x
    # print((e.T @ np.linalg.inv(P) @ e)[0,0])
    Nees.append( (e.T @ np.linalg.inv(P) @ e)[0,0] )
    # Kalman gain calcs
    pos_K.append(np.sqrt(K[0][0]**2 + K[1][1]**2))
    vel_K.append(np.sqrt(K[2][0]**2 + K[3][1]**2))
    # NIS
    if( True == estimator_.get_didAssoc() ):
        er = estimator_.get_predTrackLocation() - z
        nis = (er.T @ np.linalg.inv(S) @ er)[0,0]
        Nis.append(nis)

    # Log the Signal vs Measured vs Estimated
    # This will illustrate that these estimators function as low pass filters. 
    measPos.append(z)
    measTime.append(t_current)
    truPos.append(t[:2])
    estPos.append(x[:2])
    oosmStamp.append(isOOSM)


# Scale the process noise
nees_u = np.mean(Nees)
alphaSq_new = alphaSq * (nees_u / DOF)
print(f'Mean NEES: {nees_u}')
print(f'alpha_new: {alphaSq_new}')

# Convert P list to np array
estP00 = np.array(estimated_P00)
predP00 = np.array(predicted_P00)
posK = np.array(pos_K)
velK = np.array(vel_K)
measPos = np.array(measPos)
truPos = np.array(truPos)
estPos = np.array(estPos)
#Log the oosm frames
oosmStamp = np.array(oosmStamp)
oosm_idx = np.where(oosmStamp)[0]

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10,8))

# Plot Normalized Estiamted Error Squared
axs[0,0].plot(Nees, label="NEES", linewidth='0.1', marker='.')
axs[0,0].axhline(11.143, color='r', ls='--', lw=1, label='95% Confidence Interval')
axs[0,0].axhline(4, color='k', ls='--', lw=1, label='E[NEES]=4')
axs[0,0].axhline(0.484, color='r', ls='--', lw=1)
# axs[0,0].scatter(oosm_idx, Nees[oosm_idx], marker='*', color='orange', s=30, label='OOSM Frames')
axs[0,0].set_xlabel("Time step k")
axs[0,0].set_ylabel("NEES")
axs[0,0].set_title(f"NEES. Avg: {nees_u}")
axs[0,0].legend()

# Plot Estiamte Covariance Position Element
axs[1,0].plot(estP00, label='P_00', linewidth='0.1', marker='.', color='red')
axs[1,0].scatter(oosm_idx, estP00[oosm_idx], marker='*', color='orange', s=30, label='OOSM Frames')
axs[1,0].plot(predP00, label='Predicted P00', linewidth='0.1', marker='.', color='blue')
axs[1,0].scatter(oosm_idx, predP00[oosm_idx], marker='*', color='orange', s=30)
axs[1,0].set_xlabel('Iterations')
axs[1,0].set_title('Estimated P00')
axs[1,0].legend()

# Plot the Signal Vs Measured Vs Estimated
axs[0,2].scatter(truPos[:,0], truPos[:,1, ], label='Truth', ls='dotted', color='red')
axs[0,2].scatter(measPos[:,0], measPos[:,1] ,label='Measured', ls='dotted', color='blue')
axs[0,2].scatter(estPos[:,0], estPos[:,1], label='Filtered', ls='dotted', color='green')
axs[0,2].set_xlabel('Iterations')
axs[0,2].set_title('Filter Behavior')
axs[0,2].legend()

# Plot the Position Kalman Gain Term
axs[0,1].plot(posK, label='K pos', linewidth='0.1', marker='.', color='red')
axs[0,1].set_xlabel('Iterations')
axs[0,1].set_title('Kalman Gain Velocity')
axs[0,1].legend()

# Plot the Velocity Kalman Gain Term
axs[1,1].plot(velK, label='K vel', linewidth='0.1', marker='.', color='blue')
axs[1,1].set_xlabel('Iterations')
axs[1,1].set_title('Kalman Gain Velocity')
axs[1,1].legend()

# Plot the Velocity Kalman Gain Term
axs[1,2].hist(Nis, bins=30, density='True', color='blue', alpha=0.7, label='NIS', edgecolor='black', linewidth=0.5)
axs[1,2].set_xlabel('Iterations')
axs[1,2].set_title('NIS')
axs[1,2].legend()

fig.tight_layout()
plt.show()

# Video Player
vidPlayer_ = vid.XYStepper(truPos, measPos, estPos, measTime)
plt.show()