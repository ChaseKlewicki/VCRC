import numpy as np
import CoolProp.CoolProp as CP
from cycle_functions import *
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import warnings
import pandas as pd
import numpy as np
import CoolProp.CoolProp as CP
from cycle_functions import *
from scipy.optimize import minimize, Bounds, NonlinearConstraint, LinearConstraint
import warnings
import pandas as pd

def make_cycle(Vars, Inputs):

    # ----------------------------------------------#
    # ==------ Vars  -------==#
    P_c    = Vars[0] # Pa
    P_e    = Vars[1] # Pa
    T_SH   = Vars[2] # delta-T K
    RPM_evap = Vars[3]
    RPM    = Vars[4]
    
    # ----------------------------------------------#
    #==------ Inputs ------==#
    
    Q_load = Inputs[0] # W
    T_amb  = Inputs[1] # K
    T_pod  = Inputs[2] # K
    RPM_cond = Inputs[3] 
    
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

    # pressure drop accross evaporator (Pa)
    delta_P_e = 5e4
    
    # pressure drop accross condenser (Pa)
    delta_P_c = 5e4
    
    P[0] = P_e - delta_P_e # Pressure drop accross evap determined empirically
    
    
    # Init state
    T_sat_e = CP.PropsSI('T', 'P', P[0], 'Q', 1, 'R410a') # K
    h_g     = CP.PropsSI('H', 'P', P[0], 'Q', 1, 'R410a') # J/kg
    T[0] = T_sat_e + T_SH
    h[0] = CP.PropsSI('H', 'P', P[0], 'T', T[0], 'R410a')
    abscissa[0] = 0
    s[0] = CP.PropsSI('S', 'P', P[0], 'H', h[0], 'R410a')
    
    STATE   = [P[0], h[0]]
    
    #   calculate compressor
    m_dot_s = compr_func(STATE, RPM, P_c / P[0])
    P[1] = P_c
    
    # Isentropic Ratio
    eta_is = 2.9
    
    if eta_is < 1:
        print([RPM, P_c, P_e])
   
    h[1] = h[0] + (CP.PropsSI('H', 'P', P_c, 'S', s[0], 'R410a') - h[0]) / eta_is
    s[1] = CP.PropsSI('S', 'P', P[1], 'H', h[1], 'R410a')

    STATE = [P[1], h[1]]
    
    
    #   calculate condenser
    [P[1:5], T[1:5], h[1:5], s[1:5], abscissa[1:5], W_fan_c] = Condenser_Proc( STATE, 
                                                             'h', m_dot_s, T_amb, RPM_cond)


    # calculate expansion mass flow rate
    m_dot_v = capillary_tube_func(P[4], h[4], T[4])
    
    P[5] = P_e
    # Isenthalpic expansion
    h[5] =  h[4]
    
    STATE = [P[5], h[5]]

    #   calculate evap
    [P[5:9], T[5:9], h[5:9], s[5:9], abscissa[5:9],  W_fan_e] = Evap_Proc(STATE, m_dot_s, T_pod, RPM_evap)

    abscissa[5:9] = abscissa[5:9] + abscissa[4]

    # Energy and Mass Deficits
    Q_evap = m_dot_s * (h[8] - h[5])
    Q_absr = m_dot_s * (h[0] - h[5])

    m_def  =  (m_dot_s - m_dot_v) / m_dot_s  #Mass Deficit
    h_def  =  (Q_absr  - Q_evap) / Q_evap   #evap deficit
    Q_def  =  (Q_evap  - Q_load) / Q_load   #Pod energy deficit

    Deficit = np.array([m_def, h_def, Q_def])

    #Other Outputs
    m_dot = [m_dot_s, m_dot_v]
    Q_L   = Q_evap
    Q_H   = m_dot_v * (h[1] - h[4])
    
    # Compute compressor work based on isentropic, adiabatic compressor
    W     = m_dot_s * (CP.PropsSI('H', 'P', P_c, 'S', s[0], 'R410a') - h[0])

    # Compute coefficient of system performance
    COSP = Q_L / (W + W_fan_c + W_fan_e)

    return [P, T, h, s, abscissa, m_dot, Q_L, Q_H, W, COSP, Deficit]


def adjust_cycle_fmin(Vars, Inputs):

    assert(np.size(Vars) == 5)

    T_amb  = Inputs[1]
    T_pod  = Inputs[2]

    #
    #
    # Make Objective Function

    def objective(Vars):
        [_, _, _, _, _, _, _, _, _, Obj, _] = make_cycle(Vars, Inputs)
        
        Obj = -Obj
        
        return Obj
    
    def nonlcon1(Vars):
        
        c = (T_pod - CP.PropsSI('T', 'P', Vars[1], 'Q', 0, 'R410a')) - Vars[2] # Superheat constraint
        
        return c
             
    def nonlcon2(Vars):
        
        [_, _, _, _, _, _, _, _, _, _, Deficit] = make_cycle(Vars, Inputs)
        c = np.linalg.norm(Deficit) # deficit constraint
        
        return c
    
    nonLinear1 = NonlinearConstraint(nonlcon1, 0, np.inf)
    nonLinear2 = NonlinearConstraint(nonlcon2, -0.05, 0.05)
    
    linear = LinearConstraint(np.identity(len(Vars)),
                              [CP.PropsSI('P', 'T', T_amb, 'Q', 1, 'R410a'), 200e3, 0.25, 0, 2000], # Lower Bounds
                              [5000e3, CP.PropsSI('P', 'T', T_pod, 'Q', 0, 'R410a'), 30, 2900, 6000] # Upper Bounds
                             )
    
    # Solve the problem.
    try:
        res = minimize(objective, Vars, constraints = [nonLinear, linear], 
                       method = 'trust-constr', options = {'maxiter': 500})
    except ValueError as e:
        print(e)
        print('initial Point: ' + str(Vars))
        res = {'success': False}
    
    # ---
    if res['success']:
        Vars = res.x
        [_, _, _, _, _, _, _, _, COSP, _] = make_cycle(Vars, Inputs)
    else:
        COSP = np.nan

    return [Vars, COSP]


def solve_cycle_shotgun(Inputs):
    
    T_amb  = Inputs[1] # K
    T_pod  = Inputs[2] # K
    
    SPREAD = 1;

    #Var Extents
    lb = [CP.PropsSI('P', 'T', T_amb, 'Q', 1, 'R410a'), 400e3]
    ub = [5000e3, CP.PropsSI('P', 'T', T_pod, 'Q', 0, 'R410a')]

    #Starting points
    P_c   = lb[0] + (ub[0] - lb[0]) * np.linspace( 0.1, 0.9, SPREAD)
    P_e   = lb[1] + (ub[1] - lb[1]) * np.linspace( 0.1, 0.9, SPREAD)
    T_SH  = .5
    RPM_evap = 2900
    RPM = 3550

    # Create list of possible combinations of pressures
    Vars = np.array(np.meshgrid(P_c, P_e, T_SH, RPM_evap, RPM)).T.reshape(-1, 5)

    #Initialize Vars and COSPs
    COSPs     = np.zeros(len(Vars))

    # Try different initial points
    for ind, Var in enumerate(Vars):
        #Step Vars Forward
        [Vars[ind], COSPs[ind]] = adjust_cycle_fmin( Var, Inputs)
        
    
    # find solution with lowest COSP
    Vars = Vars[COSPs == np.nanmin(COSPs)][0]

    #Calc
    [P, T, h, s, abcissa, m_dot, Q_L, Q_H, W, COSP, Deficit] = make_cycle(Vars, 
                                                                     Inputs,
                                                                     Param)
    Props = [P, T, h, s, abcissa]
        
    return [Props, m_dot, Q_L, Q_H, W, COSP, Deficit]