# =============================================================================
# This module defines some functions in order to create a fuzzy control system.
# =============================================================================

# =============================================================================
# All these functions are defined in a way that suites the efficient inference
# mechanism; this includes the following assumptions:
# (1) Additive model for superpositioning of rules
# (2) All fuzzy sets are the same
# (3) Correlation product
# =============================================================================

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


def gaussMF(domain, number, sigma=1):
    """
    gaussMF creates a number of membership function equally spaced in the
    domain. Sigma is assumed to be 1 in order to create a nice set of MF (for
    5 curves in a range from zero to ten), but it can be passed as an extra
    argument. The centers are automatically detected in order to be equally
    spaced. Domain is a np.array of shape (n,).
    Number is an integer (should be an odd number).
    MFs are returned as a 2D array: each row contains a MF.
    """
    # Create a list with center values of the MFs (except for the first and
    # last one, which values are obvious)
    step = (domain[-1] - domain[0]) / (number - 1)
    centers = []
    for i in np.arange(number):
        centers.append(domain[0] + i * step)

    # MFs creation
    mbFunc = np.empty((number, len(domain)))  # each row is a MF

    for MF in range(number):
        mbFunc[MF] = np.exp(-((domain - centers[MF]) / sigma)**2)

    return mbFunc


def triMF(domain, number):
    """
    trisMF creates a number of triangularmembership function equally spaced in
    domain. Sigma is assumed to be 1 in order to create a nice set of MF (for
    5 curves in a range from zero to ten), but it can be passed as an extra
    argument. The centers are automatically detected in order to be equally
    spaced. Domain is a np.array of shape (n,).
    Number is an integer (should be an odd number).
    MFs are returned as a 2D array: each row contains a MF.
    """
    # Create a list with center values of the MFs (except for the first and
    # last one, which values are obvious)
    step = (domain[-1] - domain[0]) / (number - 1)
    centers = []
    for i in np.arange(number):
        centers.append(domain[0] + i * step)

    # MFs creation
    mbFunc = np.empty((number, len(domain)))  # each row is a MF

    for MF in range(number):
        if MF == 0:
            mbFunc[MF] = fuzz.trimf(domain, [centers[MF], centers[MF],
                                             centers[MF + 1]])
        elif MF >= number - 1:
            mbFunc[MF] = fuzz.trimf(domain, [centers[MF - 1], centers[MF],
                                             centers[MF]])
        else:
            mbFunc[MF] = fuzz.trimf(domain, [centers[MF - 1], centers[MF],
                                             centers[MF + 1]])
    return mbFunc


def trapMF(domain, number):
    step = (domain[-1] - domain[0]) / (number - 1)
    centers = []
    for i in np.arange(number):
        centers.append(domain[0] + i * step)

    # MFs creation
    mbFunc = np.empty((number, len(domain)))  # each row is a MF

    flat = domain[-1] / 5

    for MF in range(number):
        if MF == 0:
            mbFunc[MF] = fuzz.trapmf(domain,
                                     [centers[MF], centers[MF], centers[MF] +
                                      flat, centers[MF + 1]])
        elif MF >= number - 1:
            mbFunc[MF] = fuzz.trapmf(domain,
                                     [centers[MF - 1], centers[MF] - flat,
                                      centers[MF], centers[MF]])
        else:
            mbFunc[MF] = fuzz.trapmf(domain,
                                     [centers[MF - 1], centers[MF] - flat/2,
                                      centers[MF] + flat/2, centers[MF + 1]])
    return mbFunc


# =============================================================================
# With a command line like:
#   VL,LO,ZE,HI,VH=gaussMF(domain,5,sigma=2)
# is possible to obtain the 5 MFs in separate arrays.
#
# To obtain a single MF:
#    LO=gaussMF(domain,5,sigma=2)[1]
#
# It may be useful to work with a matrix with all the MBs:
#   MF=gaussMF(domain,5,sigma=2)
# =============================================================================


def degreeOfActivation(domain, MB, value):
    """This function calculates the degree of activation for a specified value
    of the input for every MBs.
    You should pass all the MF (for the same input) in a matrix (N,M), and a
    single value is required.

    N: number of MFs
    M: number of elements (resolution)
    """

    rows = MB.shape[0]                     # number of MFs
    lam = np.zeros((rows,))

    for row in np.arange(rows):
        lam[row] = fuzz.interp_membership(domain, MB[row], value)

    return lam


def correlationProduct(consequentMF, ruleStrenght):
    """Defines the new consequent fuzzy sets based on the correlation product
    inference mechanism.
    ConsequentMF is a matrix with MFs values in each row.
    ruleStrenght is an array (N,); every element of the array multiply
    a single row of the matrix.
    """
    newMF = np.zeros_like(consequentMF)

    for i in np.arange(np.shape(consequentMF)[0]):
        newMF[i] = ruleStrenght[i] * consequentMF[i]

    return newMF


def rules(DOA_displacement, DOA_velocity):
    # =============================================================================
    # RULEBASE:
    # (R1) if (x is ) and (v is ) then (f is )
    # (R2) if (x is ) and (v is ) then (f is )
    # (R3) if (x is ) and (v is ) then (f is )
    # (R4) if (x is ) and (v is ) then (f is )
    # (R5) if (x is ) and (v is ) then (f is )
    #
    # LEGEND:
    # -----------------------------------------------------------------------------
    # VN: very negative             |         NS: negative strong
    # NE: negative                  |         NW: negative weak
    # ZE: zero                      |         ZE: zero
    # PO: positive                  |         PW: positive weak
    # VP: very positive             |         PS: positive strong
    #                               |
    # x: displacement               |
    # v: velocity                   |         f: force
    # =============================================================================
    R = np.zeros((np.shape(DOA_displacement)[0], np.shape(DOA_velocity)[0]))

    # Generates the rule base table
    for index1 in np.arange(np.shape(DOA_displacement)[0]):
        for index2 in np.arange(np.shape(DOA_velocity)[0]):
            R[index1, index2] = np.min([DOA_displacement[index1],
                                        DOA_velocity[index2]])

    # Because there are only 5 conseguent MFs, these 25 values need to be
    # manipulated. So, by a logic sense, we decide to pick the max or sum
    # values when more than one R[i,j] refers to the same consequent MF.
    # Following the table, it traduces in this:
    
#   Rulebase 2.0
    NS = np.sum([R[4, 4], R[3, 4], R[4, 3], R[4, 2],
                 R[3, 3], R[2, 4], R[1, 4], R[1, 3]])
    NW = np.sum([R[2, 3], R[3, 2], R[4, 1], R[4, 0]])
    ZE = np.sum([R[2, 2]])
    PW = np.sum([R[2, 1], R[1, 2], R[0, 3], R[0, 4]])
    PS = np.sum([R[0, 0], R[0, 1], R[1, 0], R[2, 0],
                 R[0, 2], R[1, 1], R[3, 0], R[3, 1]])

    RuleStrenght = np.array([NS, NW, ZE, PW, PS])

    return RuleStrenght


def defuzzy(domain, newConsequentMFs):
    """Calculates the output value with the centroid based method. With the
    assumptions made, it will depens only on the degrees of activation and
    on the centroids of the original MFs."""

    # Sum all the activated MFs in order to obtain one aggregated MF.
    aggregated = np.sum(newConsequentMFs, axis=0)

    # Find the output value
    output = fuzz.defuzz(domain, aggregated, 'centroid')
    return output

def soglia(variable, limit):
    out = 0
    if variable > limit:
        out = limit
    elif variable < -limit:
        out = -limit
    else:
        out = variable
        
    return out


def FLC(inputDisplacement, inputVelocity):
    # Displacement membership functions (from -0.15 m to 0.15 m)
    suspensionTravel = np.linspace(-0.15, 0.15, 1000)
    displacementMF = gaussMF(suspensionTravel, 5, sigma=0.04)
#    displacementMF = triMF(suspensionTravel, 5)

    # Velocity memebership functions (from -1.5 to 1.5 m/s)
    suspensionVelocity = np.linspace(-1.5, 1.5, 1000)
#    velocityMF = gaussMF(suspensionVelocity,5,sigma=0.4)
    velocityMF = trapMF(suspensionVelocity, 5)

    # Actuator force memebership functions (from -1500 to 1500 N)
    # the domain is -1800 to 1800 in order to obtain the centroid at 1500 N
    actuatorForce = np.linspace(-1800, 1800, 1000)
    forceMF = gaussMF(actuatorForce, 5, sigma=400)
#    forceMF = triMF(actuatorForce, 5)

#    # Threshold for avoiding error "zero area in defuzzyfication"
#    inputDisplacement = soglia(inputDisplacement, suspensionTravel[-1])
#    inputVelocity = soglia(inputVelocity, suspensionVelocity[-1])

    # Evaluation of the degrees of activation of every MF for specified values
    # of displacement and velocity.
    # Then, the strenght of the rules is computed.
    DOA_displacement = degreeOfActivation(suspensionTravel, displacementMF,
                                          inputDisplacement)
    DOA_velocity = degreeOfActivation(suspensionVelocity, velocityMF,
                                      inputVelocity)

    # Computing the strenght of each rule.
    RuleStrenght = rules(DOA_displacement, DOA_velocity)

    # Definition of the new consequent membership functions.
    # Defuzzyfication.
    newConsequent = correlationProduct(forceMF, RuleStrenght)
    fuzzyForce = defuzzy(actuatorForce, newConsequent)

    return fuzzyForce


# =============================================================================
#                              **************
#                              ***  MAIN  ***
#                              **************
# =============================================================================
#
# Defining Membership functions for:
# (1) Antecedents: displacement and velocity
# (2) Conseguent: force of the actuator
#
# =============================================================================
# Displacement membership functions
# =============================================================================
#suspensionTravel=np.linspace(-.15,.15,1000)
#displacementMF=gaussMF(suspensionTravel,5,0.04)
##Linguistic labels:
## Very-negative, negative, zero, positive, very-positive
#labels=['VN','NE','ZE','PO','VP']
#
## Plotting:
#plt.figure(1)
#for i in np.arange(displacementMF.shape[0]):
#   plt.plot(suspensionTravel,displacementMF[i],label=labels[i])
#plt.legend()
#plt.title('Displacement MFs')
##
# =============================================================================
# Velocity memebership functions
# =============================================================================
#suspensionVelocity=np.linspace(-1.5,1.5,1000)
#velocityMF=trapMF(suspensionVelocity,5)
## Linguistic labels:
## Negative-strong, negative-weak, zero, positive-weak, positive-strong
#labels=['VN','NE','ZE','PO','VP']
#
## Plotting:
#plt.figure(2)
#for i in np.arange(velocityMF.shape[0]):
#    plt.plot(suspensionVelocity,velocityMF[i],label=labels[i])
#plt.legend()
#plt.title('Velocity MFs')
#
# =============================================================================
# Actuator force memebership functions
# =============================================================================
#actuatorForce=np.linspace(-1500,1500,850)
#forceMF=gaussMF(actuatorForce,5,sigma=600)
##Linguistic labels:
## Negative-strong, negative-weak, zero, positive-weak, positive-strong
#labels=['NS','NW','ZE','PW','PS']
#
##Plotting:
#plt.figure(3)
#for i in np.arange(forceMF.shape[0]):
#    plt.plot(actuatorForce,forceMF[i],label=labels[i])
#plt.legend()
#plt.title('Actuator force MFs')
#
# =============================================================================
# =============================================================================
# Evaluation of the degrees of activation of every MF for specified values of
# displacement and velocity.
# Then, the strenght of the rules is computed.
# =============================================================================
# =============================================================================
#
# DOA_displacement=degreeOfActivation(suspensionTravel,displacementMF,
#                                    9)
# DOA_velocity=degreeOfActivation(suspensionVelocity,velocityMF,-1)
#
# Buiding a matrix as follow:
# (-) Rows: terms of the rule
# (-) Columns: DOA for a single variable (displacement or velocity)
# DOAs=np.vstack((DOA_displacement,DOA_velocity)).T
#
# Computing the strenght of each rule.
# RuleStrenght=rules(DOA_displacement,DOA_velocity)
#
# =============================================================================
# Definition of the new consequent membership functions.
# Defuzzyfication.
# =============================================================================
# newConsequent=correlationProduct(forceMF,RuleStrenght)
#
# out=defuzzy(actuatorForce,newConsequent)
