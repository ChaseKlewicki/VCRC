import numpy as np
import CoolProp.CoolProp as CP
from .cycle_functions import *
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import warnings
import pandas as pd

def make_cycle(Vars, Inputs, Param):

    # ----------------------------------------------#
    # ==------ Vars  -------==#
    P_c    = Vars[0] # Pa
    P_e    = Vars[1] # Pa
    T_SH   = Vars[2] # delta-T K
    # ----------------------------------------------#
    #==------ Inputs ------==#
    
    Q_load = Inputs[0] # W
    T_amb  = Inputs[1] # K
    T_pod  = Inputs[2] # K
    
    #----------------------------------------------#
    #==------ Param -------==#
    RPM    = Param[0]
    
    
    #----------------------------------------------#
    #==-- Init. Outputs  --==#
    P = np.zeros(9) # Pa
    T = np.zeros(9) # K
    h = np.zeros(9) # j/kg
    s = np.zeros(9) # j/kg/k
    abscissa = np.zeros(9)
    # var "abscissa" is the nondimensional 
    # Heat exchanger position 
    # for each of these stations
    # domain = [0,1]U[1,2]
    # [0,1] <-- in condensor
    # [1,2] <-- in evaporator
    
    #=========================================================================#
    # Calculate
    #=========================================================================#

    # Init state
    T_sat_e = CP.PropsSI('T', 'P', P_e, 'Q', 1, 'R410a') # K
    h_g     = CP.PropsSI('H', 'P', P_e, 'Q', 1, 'R410a') # J/kg
    
    
    P[0] = P_e - 5e4 # Pressure drop accross evap determined empirically
    T[0] = T_sat_e + T_SH
#     h[0] = h_g + SuperHT_Cp_integral( T_sat_e, T[0])
    h[0] = CP.PropsSI('H', 'P', P[0], 'T', T[0], 'R410a')
    abscissa[0] = 0
    s[0] = CP.PropsSI('S', 'P', P_e, 'H', h[0], 'R410a')
    
    STATE   = [P[0], h[0]]
    
    #   calculate compressor
    m_dot_s = compr_func( STATE, RPM )
    
    P[1] = P_c
    s[1] = s[0] - 0.8 * (s[0] - CP.PropsSI('S', 'P', P_c, 'Q', 1, 'R410a'))
    h[1] = CP.PropsSI('H', 'P', P_c, 'S', s[1], 'R410a')

    STATE = [P[1], h[1]]
    
    
    #   calculate condenser
    [P[1:5], T[1:5], h[1:5], s[1:5], abscissa[1:5]] = Condenser_Proc( STATE, 
                                                             'h', m_dot_s, T_amb)


    #   calculate expansion
#     m_dot_v = valve_func( CA, P_c, P_e, valve )
    m_dot_v = capillary_tube_func(P_c, h[4], T[4])
    
    P[5] = P_e
    # Isenthalpic expansion
    h[5] =  h[4]
    
    STATE = [ P[5], h[5]]
    

    #   calculate evap
    [P[5:9], T[5:9], h[5:9], s[5:9], abscissa[5:9]] = Evap_Proc(STATE, m_dot_v, T_pod)

    abscissa[5:9] = abscissa[5:9] + abscissa[4]

    # Energy and Mass Deficits
    Q_evap = m_dot_v * (h[8] - h[5])
    Q_absr = m_dot_v * (h[0] - h[5])

    m_def  =  (m_dot_s - m_dot_v) / m_dot_s  #Mass Deficit
    h_def  =  (Q_absr  - Q_evap) / Q_evap   #evap deficit
    Q_def  =  (Q_evap  - Q_load) / Q_load   #Pod energy deficit

    Deficit = np.array([m_def, h_def, Q_def])

    #Other Outputs
    m_dot = m_dot_v
    Q_L   = Q_evap
    Q_H   = m_dot_v * (h[1] - h[4])
    
    # Compute compressor work based on isentropic, adiabatic compressor
    W     = m_dot_s * (CP.PropsSI('H', 'P', P_c, 'S', s[0], 'R410a') - h[0])



    return [P, T, h, s, abscissa, m_dot, Q_L, Q_H, W, Deficit]


def adjust_cycle_fmin(Vars, Inputs, Param):

    assert(np.size(Vars) == 3)

    T_amb  = Inputs[1]
    T_pod  = Inputs[2]

    # bounds
    bnds = Bounds([CP.PropsSI('P', 'T', T_amb, 'Q', 1, 'R410a'), 200e3, 0], # Lower Bounds
                  [5000e3, CP.PropsSI('P', 'T', T_pod, 'Q', 0, 'R410a'), 30], True) # Upper Bounds

    #
    #
    # Make Objective Function

    def objective(Vars):
        [_, _, _, _, _, _, _, _, _, Obj] = make_cycle(Vars, Inputs, Param)
        
        Obj = 1000 * np.linalg.norm(Obj)
        
        return Obj
                        
    #
    #
    # Make Nonlinear Constraint for T_SH

    def nonlcon(Vars):
        c = (T_pod - CP.PropsSI('T', 'P', Vars[1], 'Q', 0, 'R410a')) - Vars[2] 
        return c

    cons = NonlinearConstraint(nonlcon, 0.001, np.inf)
    
    #Options
    #options = optimoptions('fmincon','Display','iter','Algorithm','sqp');

    #
    # Solve the problem.
    try:
        res = minimize(objective, Vars, bounds = bnds, constraints=cons, 
                       method = 'trust-constr', options = {'maxiter': 500})
    except ValueError:
        res = {'success': False}
    
    # ---
    if res['success']:
        Vars = res.x
#     else:
#         warnings.warn('Did not converge')
    #
    # Return Final Deficit
    #
    [_, _, _, _, _, _, _, _, _, Deficit] = make_cycle(Vars, Inputs, Param)
    #
    # ---

    return [Vars, Deficit]


def solve_cycle_shotgun(Inputs, Param):
    
    T_amb  = Inputs[1] # K
    T_pod  = Inputs[2] # K
    
    SPREAD = 4;

    #Var Extents
    lb = [CP.PropsSI('P', 'T', T_amb, 'Q', 1, 'R410a'), 400e3]
    ub = [5000e3, CP.PropsSI('P', 'T', T_pod, 'Q', 0, 'R410a')]

    #Starting points
    P_c   = lb[0] + (ub[0] - lb[0]) * np.linspace( 0.2, 0.8, SPREAD)
    P_e   = lb[1] + (ub[1] - lb[1]) * np.linspace( 0.2, 0.8, SPREAD)
    T_SH  = .5

    # Create list of possible combinations of pressures
    Vars = np.array(np.meshgrid(P_c, P_e, T_SH)).T.reshape(-1, 3)

    #Initialize Vars and Deficits
    normDeficit = np.zeros(len(Vars))
    Deficit     = np.zeros((len(Vars), 3))

    # Try different initial points
    for ind, Var in enumerate(Vars):
        #Step Vars Forward
        [Vars[ind], Deficit[ind]] = adjust_cycle_fmin( Var, Inputs, Param)
        normDeficit[ind] = np.linalg.norm(Deficit[ind])
        
    
    # find solution with lowest error
    Vars = Vars[normDeficit == np.nanmin(normDeficit)][0]
    
    # Check if error is lower than 3% 
    converged = 1
    if normDeficit[normDeficit == np.nanmin(normDeficit)] > 0.05:
        converged = 0
        warnings.warn('Warning: |Deficit| = ' + 
                      str(normDeficit[normDeficit == min(normDeficit)]))

    #Calc
    [ P, T, h, s, abcissa, m_dot, Q_L, Q_H, W, Deficit] = make_cycle(Vars, 
                                                                     Inputs,
                                                                     Param)
    Props = [P, T, h, s, abcissa]
        
    return [Props, m_dot, Q_L, Q_H, W, Deficit, converged]