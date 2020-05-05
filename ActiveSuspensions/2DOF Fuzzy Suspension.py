import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import FLC
import pyprind
from numpy.linalg import eig
import pandas as pd

def impulse(lenght):
    i = 0
    Impulse = []

    while i < lenght:
        if i == 99:
            Impulse.append(1)
        else:
            Impulse.append(0)
        i += 1

    return 0.1 * np.array(Impulse)


def bump():
    i = 0
    Bump = []
    while i < 1:
        if i <= 0.5625 and i >= 0.5:
            Bump.append(0.05 * (1 - np.cos(32 * np.pi * i)))
        else:
            Bump.append(0)
        i += 0.001

    return np.array(Bump)


def step(lenght):
    i = 0
    Step = []

    while i < lenght:
        if i <= 500:
            Step.append(0)
        else:
            Step.append(1)
        i += 1

    return 0.1 * np.array(Step)


def rough2(lenght):
    """Random road condition.
    Every 10 time sample a new random value is given. This simulates a car
    moving on a road at 36 km/h with roughness wide 1 cm.
    """
    i = 0
    Rough = []

    while i < lenght/10:
        j = 0
        sample = np.random.randn()  # setting correct max height
        while j < 10:               # add the same value for 10 time steps
            Rough.append(sample)
            j += 1
        i += 1

    return 0.1 * np.array(Rough) / np.max(Rough) / 2


def rough3(lenght):
    """Road condition defined by the ISO 8608 standard"""
    k = 3                                   # ISO road condition
    N = lenght + 1                          # data points
    L = 10                                  # lenght of road profile
    B = L / N                               # sampling interval
    n0 = 0.1
    dn = 1 / L                              # Frequency band
    n = np.arange(dn, N*dn, dn)             # frequency band
    phi = 2 * np.pi * (np.random.rand(len(n)))
    Amp1 = np.sqrt(dn) * (2**k) * (1e-3) * n0/n
    x = np.arange(0, L-B, B)

    hx = np.zeros(len(x))
    for i in np.arange(len(x)):
        hx[i] = np.sum(Amp1 * np.cos(2 * np.pi * n * x[i] + phi))

    return 0.1 * hx / np.max(hx)

def rough():
    """Reading values from file Rough.txt"""
    f = open('Rough.txt','r')
    RoughList = []
    for line in f:
        RoughList.append(float(line))
        
    return np.array(RoughList)


def RMS(array):
    """Calculates the root-mean-squared value of an array.
    """
    return np.sqrt(array @ array / array.size)


def derivate(array, step=100):
    """Calculates the first order derivative of an array. It differs from
    np.diff because this returns an array of the same lenght as the input one.
    It becomes useful for plotting.
    """
    deriv = np.zeros_like(array)
    deriv[0] = array[1] - array[0]
    deriv[1:] = np.diff(array)

    return deriv * step


# =============================================================================
# Importing values of PID
# =============================================================================
StepPID = pd.read_excel('Scalino.xlsx')
StepPID = np.asarray(StepPID)

ImpulsePID = pd.read_excel('impulso.xlsx')
ImpulsePID = np.asarray(ImpulsePID)

BumpPID = pd.read_excel('BumpPID.xlsx')
BumpPID = np.asarray(BumpPID)

RoughPID = pd.read_excel('Rough.xlsx')
RoughPID = np.asarray(RoughPID)

# =============================================================================
# STATE SPACE REPRESENTATION
# x1 = x_body
# x2 = x_wheel
# x3 = x_body'
# x4 = x_wheel'
# =============================================================================

# Main spring stiffness
k_s = 15000  # N/m

# Sprung mass
m_b = 250  # kg

# Viscous damper
c_s = 1000  # N/(m/s)

# Unsprung mass (wheel)
m_w = 30  # kg

# Tyre stiffness
k_t = 150000  # N/m

# Skyhook damping
c_sky = 1000  # N/(m/s)

# Different road simulations
Impulse = impulse(1000)
Step = step(1000)
Bump = bump()
Rough = rough()


def fuzzySuspensionModel(timeScale, state, road):

    x1, x2, x3, x4 = state
    fuzzyForce = FLC.FLC(x1 - x2, x3)

    xdot1 = x3
    xdot2 = x4
    xdot3 = (-k_s / m_b * x1 + k_s / m_b * x2 - c_s /
             m_b * x3 + c_s / m_b * x4 + 1 / m_b * fuzzyForce)
    xdot4 = (k_s / m_w * x1 - (k_t + k_s) / m_w * x2 + c_s / m_w * x3 -
             c_s / m_w * x4 + k_t / m_w * road - 1 / m_w * fuzzyForce)

    return np.array([xdot1, xdot2, xdot3, xdot4])


def passiveSuspensionModel(timeScale, state, road):

    x1, x2, x3, x4 = state

    xdot1 = x3
    xdot2 = x4
    xdot3 = -k_s / m_b * x1 + k_s / m_b * x2 - c_s / m_b * x3 + c_s / m_b * x4
    xdot4 = (k_s / m_w * x1 - (k_t + k_s) / m_w * x2 + c_s /
             m_w * x3 - c_s / m_w * x4 + k_t / m_w * road)

    return np.array([xdot1, xdot2, xdot3, xdot4])


def skyhookSuspensionModel(timeScale, state, road):

    x1, x2, x3, x4 = state

    xdot1 = x3
    xdot2 = x4
    xdot3 = (-k_s / m_b * x1 + k_s / m_b * x2 - c_s / m_b * x3 + c_s / m_b * x4
             - c_sky / m_b * x3)
    xdot4 = (k_s / m_w * x1 - (k_t + k_s) / m_w * x2 + c_s /
             m_w * x3 - c_s / m_w * x4 + k_t / m_w * road)

    return np.array([xdot1, xdot2, xdot3, xdot4])


# =============================================================================
# ## ODE solution - fuzzy
# =============================================================================

# Step
solStep = ode(fuzzySuspensionModel).set_integrator('dopri5',
                                                   atol=1e-6)
state0 = [0, 0, 0, 0]
solStep.set_initial_value(state0)
tFin = 10 - 0.01
dt = 0.01
Time = []
StepState = []
counter = 0

progress = pyprind.ProgBar(1000, title='Processing: Step')

while solStep.successful() and solStep.t < tFin:
    solStep.set_f_params(Step[counter])
    solStep.integrate(solStep.t + dt)
    StepState.append(solStep.y)
    Time.append(solStep.t)
    counter += 1
    progress.update()

Time = np.asarray(Time)
StepState = np.asarray(StepState)

# Impulse
solImpulse = ode(fuzzySuspensionModel).set_integrator('dopri5',
                                                      atol=1e-6)
state0 = [0, 0, 0, 0]
solImpulse.set_initial_value(state0)
tFin = 10 - 0.01
dt = 0.01
Time = []
ImpulseState = []
counter = 0

progress = pyprind.ProgBar(1000, title='Processing: Impulse')

while solImpulse.successful() and solImpulse.t < tFin:
    solImpulse.set_f_params(Impulse[counter])
    solImpulse.integrate(solImpulse.t + dt)
    ImpulseState.append(solImpulse.y)
    Time.append(solImpulse.t)
    counter += 1
    progress.update()

Time = np.asarray(Time)
ImpulseState = np.asarray(ImpulseState)

# Bump
solBump = ode(fuzzySuspensionModel).set_integrator('dopri5',
                                                   atol=1e-6)
state0 = [0, 0, 0, 0]
solBump.set_initial_value(state0)
tFin = 10 - 0.01
dt = 0.01
Time = []
BumpState = []
counter = 0

progress = pyprind.ProgBar(1000, title='Processing: Bump')

while solBump.successful() and solBump.t < tFin:
    solBump.set_f_params(Bump[counter])
    solBump.integrate(solBump.t + dt)
    BumpState.append(solBump.y)
    Time.append(solBump.t)
    counter += 1
    progress.update()

Time = np.asarray(Time)
BumpState = np.asarray(BumpState)

# Rough road
solRough = ode(fuzzySuspensionModel).set_integrator('dopri5', atol=1e-6)
state0 = [0, 0, 0, 0]
solRough.set_initial_value(state0)
tFin = 10 - 0.01
dt = 0.01
Time = []
RoughState = []
counter = 0

progress = pyprind.ProgBar(1000, title='Processing: Rough')

while solRough.successful() and solRough.t < tFin:
    solRough.set_f_params(Rough[counter])
    solRough.integrate(solRough.t + dt)
    RoughState.append(solRough.y)
    Time.append(solRough.t)
    counter += 1
    progress.update()

Time = np.asarray(Time)
RoughState = np.asarray(RoughState)

# =============================================================================
# ## ODE solution - passive
# =============================================================================

# Step
solStep2 = ode(passiveSuspensionModel).set_integrator('dopri5', atol=1e-6)
state0 = [0, 0, 0, 0]
solStep2.set_initial_value(state0)
tFin = 10 - 0.01
dt = 0.01
Time2 = []
StepState2 = []
counter = 0

progress = pyprind.ProgBar(1000, title='Processing: Step')

while solStep2.successful() and solStep2.t < tFin:
    solStep2.set_f_params(Step[counter])
    solStep2.integrate(solStep2.t + dt)
    StepState2.append(solStep2.y)
    Time2.append(solStep2.t)
    counter += 1
    progress.update()

Time2 = np.asarray(Time2)
StepState2 = np.asarray(StepState2)

# Impulse
solImpulse2 = ode(passiveSuspensionModel).set_integrator('dopri5', atol=1e-6)
state0 = [0, 0, 0, 0]
solImpulse2.set_initial_value(state0)
tFin = 10 - 0.01
dt = 0.01
Time2 = []
ImpulseState2 = []
counter = 0

progress = pyprind.ProgBar(1000, title='Processing: Impulse')

while solImpulse2.successful() and solImpulse2.t < tFin:
    solImpulse2.set_f_params(Impulse[counter])
    solImpulse2.integrate(solImpulse2.t + dt)
    ImpulseState2.append(solImpulse2.y)
    Time2.append(solImpulse2.t)
    counter += 1
    progress.update()

Time2 = np.asarray(Time2)
ImpulseState2 = np.asarray(ImpulseState2)

# Bump
solBump2 = ode(passiveSuspensionModel).set_integrator('dopri5', atol=1e-6)
state0 = [0, 0, 0, 0]
solBump2.set_initial_value(state0)
tFin = 10 - 0.01
dt = 0.01
Time2 = []
BumpState2 = []
counter = 0

progress = pyprind.ProgBar(1000, title='Processing: Bump')

while solBump2.successful() and solBump2.t < tFin:
    solBump2.set_f_params(Bump[counter])
    solBump2.integrate(solBump2.t + dt)
    BumpState2.append(solBump2.y)
    Time2.append(solBump2.t)
    counter += 1
    progress.update()

Time2 = np.asarray(Time2)
BumpState2 = np.asarray(BumpState2)

# Rough road
solRough2 = ode(passiveSuspensionModel).set_integrator('dopri5', atol=1e-6)
state0 = [0, 0, 0, 0]
solRough2.set_initial_value(state0)
tFin = 10 - 0.01
dt = 0.01
Time2 = []
RoughState2 = []
counter = 0

progress = pyprind.ProgBar(1000, title='Processing: Rough')

while solRough2.successful() and solRough2.t < tFin:
    solRough2.set_f_params(Rough[counter])
    solRough2.integrate(solRough2.t + dt)
    RoughState2.append(solRough2.y)
    Time2.append(solRough2.t)
    counter += 1
    progress.update()

Time2 = np.asarray(Time2)
RoughState2 = np.asarray(RoughState2)

# =============================================================================
# ## ODE solution - skyhook
# =============================================================================

# Step
solStep3 = ode(skyhookSuspensionModel).set_integrator('dopri5', atol=1e-6)
state0 = [0, 0, 0, 0]
solStep3.set_initial_value(state0)
tFin = 10 - 0.01
dt = 0.01
Time3 = []
StepState3 = []
counter = 0

progress = pyprind.ProgBar(1000, title='Processing: Step')

while solStep3.successful() and solStep3.t < tFin:
    solStep3.set_f_params(Step[counter])
    solStep3.integrate(solStep3.t + dt)
    StepState3.append(solStep3.y)
    Time3.append(solStep3.t)
    counter += 1
    progress.update()

Time3 = np.asarray(Time3)
StepState3 = np.asarray(StepState3)

# Impulse
solImpulse3 = ode(skyhookSuspensionModel).set_integrator('dopri5', atol=1e-6)
state0 = [0, 0, 0, 0]
solImpulse3.set_initial_value(state0)
tFin = 10 - 0.01
dt = 0.01
Time3 = []
ImpulseState3 = []
counter = 0

progress = pyprind.ProgBar(1000, title='Processing: Impulse')

while solImpulse3.successful() and solImpulse3.t < tFin:
    solImpulse3.set_f_params(Impulse[counter])
    solImpulse3.integrate(solImpulse3.t + dt)
    ImpulseState3.append(solImpulse3.y)
    Time3.append(solImpulse3.t)
    counter += 1
    progress.update()

Time3 = np.asarray(Time3)
ImpulseState3 = np.asarray(ImpulseState3)

# Bump
solBump3 = ode(skyhookSuspensionModel).set_integrator('dopri5', atol=1e-6)
state0 = [0, 0, 0, 0]
solBump3.set_initial_value(state0)
tFin = 10 - 0.01
dt = 0.01
Time3 = []
BumpState3 = []
counter = 0

progress = pyprind.ProgBar(1000, title='Processing: Bump')

while solBump3.successful() and solBump3.t < tFin:
    solBump3.set_f_params(Bump[counter])
    solBump3.integrate(solBump3.t + dt)
    BumpState3.append(solBump3.y)
    Time3.append(solBump3.t)
    counter += 1
    progress.update()

Time3 = np.asarray(Time3)
BumpState3 = np.asarray(BumpState3)

# Rough road
solRough3 = ode(skyhookSuspensionModel).set_integrator('dopri5', atol=1e-6)
state0 = [0, 0, 0, 0]
solRough3.set_initial_value(state0)
tFin = 10 - 0.01
dt = 0.01
Time3 = []
RoughState3 = []
counter = 0

progress = pyprind.ProgBar(1000, title='Processing: Rough')

while solRough3.successful() and solRough3.t < tFin:
    solRough3.set_f_params(Rough[counter])
    solRough3.integrate(solRough3.t + dt)
    RoughState3.append(solRough3.y)
    Time3.append(solRough3.t)
    counter += 1
    progress.update()

Time3 = np.asarray(Time3)
RoughState3 = np.asarray(RoughState3)


# =============================================================================
# ACCELERATION EVALUATION (AND FUZZY FORCE)
# =============================================================================

# Step
StepAcc = derivate(StepState[:, 2])
StepAcc2 = derivate(StepState2[:, 2])
StepAcc3 = derivate(StepState3[:, 2])

StepForce = (-k_s * StepState[:, 0] + k_s * StepState[:, 1] -
             c_s * StepState[:, 2] + c_s * StepState[:, 3] - StepAcc[:] * m_b)

# Impulse
ImpulseAcc = derivate(ImpulseState[:, 2])
ImpulseAcc2 = derivate(ImpulseState2[:, 2])
ImpulseAcc3 = derivate(ImpulseState3[:, 2])

ImpulseForce = (-k_s * ImpulseState[:, 0] + k_s * ImpulseState[:, 1] -
                c_s * ImpulseState[:, 2] + c_s * ImpulseState[:, 3] -
                ImpulseAcc[:] * m_b)

# Bump
BumpAcc = derivate(BumpState[:, 2])
BumpAcc2 = derivate(BumpState2[:, 2])
BumpAcc3 = derivate(BumpState3[:, 2])

BumpForce = (-k_s * BumpState[:, 0] + k_s * BumpState[:, 1] -
             c_s * BumpState[:, 2] + c_s * BumpState[:, 3] - BumpAcc[:] * m_b)

# Rough
RoughAcc = derivate(RoughState[:, 2])
RoughAcc2 = derivate(RoughState2[:, 2])
RoughAcc3 = derivate(RoughState3[:, 2])

RoughForce = (-k_s * RoughState[:, 0] + k_s * RoughState[:, 1] -
              c_s * RoughState[:, 2] + c_s * RoughState[:, 3] -
              RoughAcc[:] * m_b)

# =============================================================================
# # PLOTTING
# =============================================================================

# Step
plt.figure(1)
plt.plot(Time, 1e3 * StepState[:, 0], 'C1', label='Fuzzy')
plt.plot(Time2, 1e3 * StepState2[:, 0], 'C2', label='Passive', linewidth=1)
plt.plot(Time3, 1e3 * StepState3[:, 0], 'C3', label='Skyhook', linewidth=1)
plt.plot(StepPID[:, 0], 1e3 * StepPID[:, 1], 'C4', label='PID', linewidth=1)
plt.plot(Time, 1e3 * Step, 'C0', label='Road', linewidth=0.8)
plt.xlabel('Time [s]')
plt.ylabel('Body displacement [mm]')
plt.legend()

plt.figure(2)
plt.plot(Time, 1e3 * StepState[:, 1], 'C1', label='Fuzzy')
plt.plot(Time2, 1e3 * StepState2[:, 1], 'C2', label='Passive', linewidth=1)
plt.plot(Time3, 1e3 * StepState3[:, 1], 'C3', label='Skyhook', linewidth=1)
plt.plot(StepPID[:, 0], 1e3 * StepPID[:, 2], 'C4', label='PID', linewidth=1)
plt.xlabel('Time [s]')
plt.ylabel('Unsprung mass displacement [mm]')
plt.legend()

plt.figure(3)
plt.plot(Time, StepAcc, 'C1', label='Fuzzy')
plt.plot(Time2, StepAcc2, 'C2', label='Passive', linewidth=1)
plt.plot(Time3, StepAcc3, 'C3', label='Skyhook', linewidth=1)
plt.plot(StepPID[:, 0], StepPID[:, 3], 'C4', label='PID', linewidth=1)
# plt.plot(Time, StepForce/m_b, 'C0', label='Force', linewidth=0.8)
plt.xlabel('Time [s]')
plt.ylabel(r'Body acceleration [m/${s^2}$]')
plt.legend()

# Impulse
plt.figure(4)
plt.plot(Time, 1e3 * ImpulseState[:, 0], 'C1', label='Fuzzy')
plt.plot(Time2, 1e3 * ImpulseState2[:, 0], 'C2', label='Passive', linewidth=1)
plt.plot(Time3, 1e3 * ImpulseState3[:, 0], 'C3', label='Skyhook', linewidth=1)
plt.plot(ImpulsePID[:, 0], 1e3 * ImpulsePID[:, 1], 'C4', label='PID', linewidth=1)
plt.plot(Time, 1e3 * Impulse, 'C0', label='Road', linewidth=0.8)
plt.xlabel('Time [s]')
plt.ylabel('Body displacement [mm]')
plt.legend()

plt.figure(5)
plt.plot(Time, 1e3 * ImpulseState[:, 1], 'C1', label='Fuzzy')
plt.plot(Time2, 1e3 * ImpulseState2[:, 1], 'C2', label='Passive', linewidth=1)
plt.plot(Time3, 1e3 * ImpulseState3[:, 1], 'C3', label='Skyhook', linewidth=1)
plt.plot(ImpulsePID[:, 0], 1e3 * ImpulsePID[:, 2], 'C4', label='PID', linewidth=1)
plt.xlabel('Time [s]')
plt.ylabel('Unsprung mass displacement [mm]')
plt.legend()

plt.figure(6)
plt.plot(Time, ImpulseAcc, 'C1', label='Fuzzy')
plt.plot(Time2, ImpulseAcc2, 'C2', label='Passive', linewidth=1)
plt.plot(Time3, ImpulseAcc3, 'C3', label='Skyhook', linewidth=1)
plt.plot(ImpulsePID[:, 0], ImpulsePID[:, 3], 'C4', label='PID', linewidth=1)
# plt.plot(Time, ImpulseForce/m_b, 'C0', label='Force', linewidth=0.8)
plt.xlabel('Time [s]')
plt.ylabel(r'Body acceleration [m/${s^2}$]')
plt.legend()

# Bump
plt.figure(7)
plt.plot(Time, 1e3 * BumpState[:, 0], 'C1', label='Fuzzy')
plt.plot(Time2, 1e3 * BumpState2[:, 0], 'C2', label='Passive', linewidth=1)
plt.plot(Time3, 1e3 * BumpState3[:, 0], 'C3', label='Skyhook', linewidth=1)
plt.plot(BumpPID[:, 0], 1e3 * BumpPID[:, 1], 'C4', label='PID', linewidth=1)
plt.plot(Time, 1e3 * Bump, 'C0', label='Road', linewidth=0.8)
plt.xlabel('Time [s]')
plt.ylabel('Body displacement [mm]')
plt.legend()

plt.figure(8)
plt.plot(Time, 1e3 * BumpState[:, 1], 'C1', label='Fuzzy')
plt.plot(Time2, 1e3 * BumpState2[:, 1], 'C2', label='Passive', linewidth=1)
plt.plot(Time3, 1e3 * BumpState3[:, 1], 'C3', label='Skyhook', linewidth=1)
plt.plot(BumpPID[:, 0], 1e3 * BumpPID[:, 2], 'C4', label='PID', linewidth=1)
plt.xlabel('Time [s]')
plt.ylabel('Unsprung mass displacement [mm]')
plt.legend()

plt.figure(9)
plt.plot(Time, BumpAcc, 'C1', label='Fuzzy')
plt.plot(Time2, BumpAcc2, 'C2', label='Passive', linewidth=1)
plt.plot(Time3, BumpAcc3, 'C3', label='Skyhook', linewidth=1)
plt.plot(BumpPID[:, 0], BumpPID[:, 3], 'C4', label='PID', linewidth=1)
# plt.plot(Time, BumpForce/m_b, 'C0', label='Force', linewidth=0.8)
plt.xlabel('Time [s]')
plt.ylabel(r'Body acceleration [m/${s^2}$]')
plt.legend()

# Rough
plt.figure(10)
plt.plot(Time, 1e3 * RoughState[:, 0], 'C1', label='Fuzzy')
plt.plot(Time2, 1e3 * RoughState2[:, 0], 'C2', label='Passive', linewidth=1)
plt.plot(Time3, 1e3 * RoughState3[:, 0], 'C3', label='Skyhook', linewidth=1)
plt.plot(RoughPID[:, 0], 1e3 * RoughPID[:, 1], 'C4', label='PID', linewidth=1)
plt.plot(Time, 1e3 * Rough, 'C0', label='Road', linewidth=0.8)
plt.xlabel('Time [s]')
plt.ylabel('Body displacement [mm]')
plt.legend()

plt.figure(11)
plt.plot(Time, 1e3 * RoughState[:, 1], 'C1', label='Fuzzy')
plt.plot(Time2, 1e3 * RoughState2[:, 1], 'C2', label='Passive', linewidth=1)
plt.plot(Time3, 1e3 * RoughState3[:, 1], 'C3', label='Skyhook', linewidth=1)
plt.plot(RoughPID[:, 0], 1e3 * RoughPID[:, 2], 'C4', label='PID', linewidth=1)
plt.xlabel('Time [s]')
plt.ylabel('Unsprung mass displacement [mm]')
plt.legend()

plt.figure(12)
plt.plot(Time, RoughAcc, 'C1', label='Fuzzy')
plt.plot(Time2, RoughAcc2, 'C2', label='Passive', linewidth=1)
plt.plot(Time3, RoughAcc3, 'C3', label='Skyhook', linewidth=1)
plt.plot(RoughPID[:, 0], RoughPID[:, 3], 'C4', label='PID', linewidth=1)
# plt.plot(Time, RoughForce/m_b, 'C0', label='Force', linewidth=0.8)
plt.xlabel('Time [s]')
plt.ylabel(r'Body acceleration [m/${s^2}$]')
plt.legend()

# =============================================================================
# RESULTS
# =============================================================================
# Calculation of RMS for:
# (1) Body displacement
# (2) Body accelaration
# (3) Wheel hop (unsprung mass displacement)

#StepFuzzyRMS = np.array([
#                        RMS(StepState[:, 0]),
#                        RMS(StepAcc),
#                        RMS(StepState[:, 1])
#                        ])
#
#StepPassiveRMS = np.array([
#                        RMS(StepState2[:, 0]),
#                        RMS(StepAcc2),
#                        RMS(StepState2[:, 1])
#                        ])
#
#StepSkyhookRMS = np.array([
#                        RMS(StepState3[:, 0]),
#                        RMS(StepAcc3),
#                        RMS(StepState3[:, 1])
#                        ])
#
#StepResult = np.array([
#        (StepFuzzyRMS - StepPassiveRMS) / StepPassiveRMS,
#        (StepSkyhookRMS - StepPassiveRMS) / StepPassiveRMS,
#        ]) * 100
#
#ImpulseFuzzyRMS = np.array([
#                        RMS(ImpulseState[:, 0]),
#                        RMS(ImpulseAcc),
#                        RMS(ImpulseState[:, 1])
#                        ])
#
#ImpulsePassiveRMS = np.array([
#                        RMS(ImpulseState2[:, 0]),
#                        RMS(ImpulseAcc2),
#                        RMS(ImpulseState2[:, 1])
#                        ])
#
#ImpulseSkyhookRMS = np.array([
#                        RMS(ImpulseState3[:, 0]),
#                        RMS(ImpulseAcc3),
#                        RMS(ImpulseState3[:, 1])
#                        ])
#
#ImpulseResult = np.array([
#        (ImpulseFuzzyRMS - ImpulsePassiveRMS) / ImpulsePassiveRMS,
#        (ImpulseSkyhookRMS - ImpulsePassiveRMS) / ImpulsePassiveRMS
#        ]) * 100
#
#BumpFuzzyRMS = np.array([
#                        RMS(BumpState[:, 0]),
#                        RMS(BumpAcc),
#                        RMS(BumpState[:, 1])
#                        ])
#
#BumpPassiveRMS = np.array([
#                        RMS(BumpState2[:, 0]),
#                        RMS(BumpAcc2),
#                        RMS(BumpState2[:, 1])
#                        ])
#
#BumpSkyhookRMS = np.array([
#                        RMS(BumpState3[:, 0]),
#                        RMS(BumpAcc3),
#                        RMS(BumpState3[:, 1])
#                        ])
#
#BumpResult = np.array([
#        (BumpFuzzyRMS - BumpPassiveRMS) / BumpPassiveRMS,
#        (BumpSkyhookRMS - BumpPassiveRMS) / BumpPassiveRMS
#        ]) * 100

RoughFuzzyRMS = np.array([
                        RMS(RoughState[:, 0] - Rough),
                        RMS(RoughAcc),
                        RMS(RoughState[:, 1] - Rough)
                        ])

RoughPassiveRMS = np.array([
                        RMS(RoughState2[:, 0] - Rough),
                        RMS(RoughAcc2),
                        RMS(RoughState2[:, 1] - Rough)
                        ])

RoughSkyhookRMS = np.array([
                        RMS(RoughState3[:, 0] - Rough),
                        RMS(RoughAcc3),
                        RMS(RoughState3[:, 1] - Rough)
                        ])
    
RoughPIDRMS = np.array([
                        RMS(RoughPID[:, 1] - Rough[:-1]),
                        RMS(RoughPID[:, 3]),
                        RMS(RoughPID[:, 2] - Rough[:-1])
                        ])

RoughResult = np.array([
        (RoughFuzzyRMS - RoughPassiveRMS) / RoughPassiveRMS,
        (RoughSkyhookRMS - RoughPassiveRMS) / RoughPassiveRMS,
        (RoughPIDRMS - RoughPassiveRMS) / RoughPassiveRMS
        ]) * 100

#RoughResult = np.array([
#        (RoughFuzzyRMS - RMS(Rough)) / RMS(Rough),
#        (RoughSkyhookRMS - RMS(Rough)) / RMS(Rough),
#        (RoughPIDRMS - RMS(Rough)) / RMS(Rough)
#        ]) * 100


# =============================================================================
# FFT ANALYSIS
# =============================================================================
label = ['Fuzzy', 'Passive', 'Skyhook', 'PID']
colors = ['C1', 'C2', 'C3', 'C4']
i = 0
for acc in [StepAcc, StepAcc2, StepAcc3, StepPID[:, 3]]:
    fft = np.fft.fft(acc)
    freq = np.fft.fftfreq(len(acc), 0.01)
    plt.figure(13)
    plt.loglog(np.abs(freq), np.abs(fft), colors[i], label=label[i],
               linewidth=1)
    i += 1
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Acceleration')
plt.title('Step')

i = 0
for acc in [ImpulseAcc, ImpulseAcc2, ImpulseAcc3, ImpulsePID[:, 3]]:
    fft = np.fft.fft(acc)
    freq = np.fft.fftfreq(len(acc), 0.01)
    plt.figure(14)
    plt.loglog(np.abs(freq), np.abs(fft), colors[i], label=label[i],
               linewidth=1)
    i += 1
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Acceleration')
plt.title('Impulse')

i = 0
for acc in [BumpAcc, BumpAcc2, BumpAcc3]:
    fft = np.fft.fft(acc)
    freq = np.fft.fftfreq(len(acc), 0.01)
    plt.figure(15)
    plt.loglog(np.abs(freq), np.abs(fft), colors[i],
               label=label[i], linewidth=1)
    i += 1
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Acceleration')
plt.title('Bump')

i = 0
for acc in [RoughAcc, RoughAcc2, RoughAcc3]:
    fft = np.fft.fft(acc)
    freq = np.fft.fftfreq(len(acc), 0.01)
    plt.figure(16)
    plt.loglog(np.abs(freq), np.abs(fft),colors[i],
               label=label[i], linewidth=1)
    i += 1
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Acceleration')
plt.title('Rough')
