import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import odeint

def impulse(lenght):
    i=0
    Impulse=[]
    
    while i<lenght:
        if i==0:
            Impulse.append(1)
        else:
            Impulse.append(0)    
        i+=1

    return 0.1*np.array(Impulse)

def bump():
    i=0
    Bump=[]
    
    while i<1:
        if i<=0.5 and i>=0.25:
            Bump.append(0.05*(1-np.cos(8*np.pi*i)))
        else:
            Bump.append(0)    
        i+=0.001

    return np.array(Bump)

def noise(lenght):
    return 0.1*np.random.randn(lenght)

def step(lenght):
    return 0.1*np.ones(lenght)
    

# =============================================================================
# STATE SPACE REPRESENTATION
# x1 = x_body
# x2 = x_wheel
# x3 = x_body'
# x4 = x_wheel'
# =============================================================================
# Main spring stiffness
k_s=16000 #N/m

# Sprung mass
m_b=300 #kg

# Viscous damper
c_s=1000 #N/(m/s)

# Unsprung mass (wheel)
m_w=60 #kg

# Tyre stiffness
k_t=190000 #N/m

# Matrix definition

A=np.array([[0,0,1,0],
            [0,0,0,1],
            [-k_s/m_b, k_s/m_b, -c_s/m_b, c_s/m_b],
            [k_s/m_w, -(k_s+k_t)/m_w, c_s/m_w, -c_s/m_w]])

B=np.array([[0],
           [0],
           [0],
           [k_t/m_w]])

C=np.eye(4)

D=np.zeros((4,1))

# Input definition
Noise=noise(1000)
Step=step(1000)
Impulse=impulse(1000)
Bump=bump()

# Solving State Space
sys=signal.StateSpace(A,B,C,D)
solNoise=signal.lsim(sys,Noise,np.linspace(0,10,1000))
solStep=signal.lsim(sys,Step,np.linspace(0,10,1000))
solImpulse=signal.lsim(sys,Impulse,np.linspace(0,10,1000))
solBump=signal.lsim(sys,Bump,np.linspace(0,10,1000))

# =============================================================================
# # PLOTTING
# =============================================================================

# White noise
fig=plt.figure(1)
plt.subplot(211)
plt.plot(solNoise[0],1e3*solNoise[1][:,0],'C3')
plt.ylabel('Body displacement [mm]')
plt.subplot(212)
plt.plot(solNoise[0],solNoise[1][:,2],'C4')
plt.ylabel('Body velocity [m/s]')
fig.suptitle('Noise response',size=12)

# Step
fig=plt.figure(2)
plt.subplot(211)
plt.plot(solStep[0],1e3*solStep[1][:,0],'C3')
plt.ylabel('Body displacement [mm]')
plt.subplot(212)
plt.plot(solStep[0],solStep[1][:,2],'C4')
plt.ylabel('Body velocity [m/s]')
fig.suptitle('Step response',size=12)

# Impulse
fig=plt.figure(3)
plt.subplot(211)
plt.plot(solImpulse[0],1e3*solImpulse[1][:,0],'C3')
plt.ylabel('Body displacement [mm]')
plt.subplot(212)
plt.plot(solImpulse[0],solImpulse[1][:,2],'C4')
plt.ylabel('Body velocity [m/s]')
fig.suptitle('Impulse response',size=12)

# Bump
fig=plt.figure(4)
plt.subplot(211)
plt.plot(solBump[0],1e3*solBump[1][:,0],'C3')
plt.ylabel('Body displacement [mm]')
plt.subplot(212)
plt.plot(solBump[0],solBump[1][:,2],'C4')
plt.ylabel('Body velocity [m/s]')
fig.suptitle('Bump response',size=12)