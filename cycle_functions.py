import numpy as np
import warnings
import CoolProp.CoolProp as CP
from scipy.optimize import fsolve


def SuperHT_Cp_integral(T1, T2):
    # Incorrect based on Suva thermo props sheet for ideal gas
    # Inputs should be in Kelvin.
    # output is in j/kg
    T2 = float(T2)

    # R-410a constants
    c1 = np.array([2.676084E-1, 2.115353E-3, -9.848184E-7, 6.493781E-11])

    c1 = np.array([1, 1/2, 1/3, 1/4]) * c1

    vec1 = np.matrix([[T1],  [T1**2],  [T1**3],  [T1**4]])
    vec2 = np.matrix([[T2],  [T2**2],  [T2**3],  [T2**4]])

    deltaH = c1 * vec2 - c1 * vec1

    return float(deltaH * 1000)


def compr_func( inlet_state, RPM ):

    P_e   = inlet_state[0] # Pa
    h_e_o = inlet_state[1] # j/kg

    #Param
    r_c = 2.4        # Compression Ratio
    Disp = 5.25E-6    #[m^3 per rev] #volume displacement

    h_g   = CP.PropsSI('H', 'P', P_e, 'Q', 1, 'R410a')
    if h_e_o < h_g:
        warnings.warn('Flooded Compressor, vapor quality < 1')
    
    rho_c = CP.PropsSI('D', 'P', P_e, 'H' ,h_e_o, 'R410a')

    m_dot = RPM / 60 * Disp * r_c * rho_c

    return m_dot


def Circular_Duct_Nu(Re, Pr,strg):

    if np.size(Re)!=np.size(Pr):
        raise ValueError('Re and Pr require same number of elements')

    out = np.zeros(np.size(Re))

    for j in range(0,np.size(Re)):

    #-----------------------------------#
        if Re[j] <= 2000: # Laminar Regime #
            out[j] = 3.66

    #--------------------------------------------#
        elif Re[j] >= 2300: # Turbulent Regime    
            
            if strg == 'c':
                out[j] = 0.023 * (Re[j]**(0.8)) * (Pr[j]**0.4) #Cold Side is being heated
                #Diddus and Boehler
            elif strg == 'h':
                out[j] = 0.023 * (Re[j]**(0.8)) * (Pr[j]**0.3) #Hot side being cooled
                #Diddus and Boehler
            else:
                raise NameError('String not recognized. need either' +
                                ' h for hot and c for cold-side')

    #------------------------------#
        else:  # Transition Regime    #
    #------------------------------#

            out1 = 3.66;

            if strg == 'c':
                out2 = 0.023 * (Re[j]**(0.8)) * (Pr[j]**0.4) #Cold Side is being heated
                #Diddus and Boehler
            elif strg == 'h':
                out2 = 0.023 * (Re[j]**(0.8)) * (Pr[j]**0.3) #Hot side being cooled
                #Diddus and Boehler
            else:
                raise NameError('String not recognized. need either' +
                                ' h for hot and c for cold-side')

            out[j] = (out1 * (2300-Re[j]) + out2 * (Re[j] - 2000)) / 300;

    return out


def generate_HTCOEFF(P, m_dot_g, m_dot_f, V_extr, subsys, T_air):


    if subsys == 'EVAP':
        
        # Geometric Characteristics
        
        # Fin density (fins/m) [measured 19 fins per inch]
        Nf = 19 / 0.0254

        # Outside diameter of tubing (m) [measured .31"]
        do = 0.3125 *.0254

        # Inside diameter of tubing (m) [wall thickness estimated at 0.03"]
        di = do - 2 * 0.03 *.0254

        # Transverse spacing between tubes (m) [measured 0.86"]
        xt = 0.86 * 0.0254

        # Longitudinal spacing between tubes (m) [(measured 0.994") / 2]
        xl = (0.994 * 0.0254) / 2

        # Fin thickness (m) [measured 0.004"]
        delta = 0.004 * 0.0254

        # Overall Length (m) 
        L1 = (12.5) * 0.0254
        
        # Overall depth (m) [measured 1 1/2"]
        L2 = (1.5) * 0.0254

        # Overall height (m) 
        L3 = 8.5 * 0.0254
        
        # Number of Rows 
        Nr = 3
        
        # Number of tubes
        Nt = 30

        #Interior (refrigerant side)
        A_i = np.pi * di * Nt * L1 # [m2]

        #Pipe Wall
        k_pipe = 385 # copper [W/m-K]

        R_tw = np.log(do / di) / (2 * np.pi * k_pipe * Nt * L1) # [K/W]

        #Exterior (air side)
        
        # Primary surface area (tubes and header plates)
        A_p = np.pi * do * (L1 - delta * Nf * L1) * Nt + 2 * (L3 * L2 - np.pi * do**2 / 4 * Nt)

        # Secondary surface area (fins)
        A_f = 2 * (L2 * L3 - (np.pi * do**2 /  4) * Nt) * Nf * L1 + 2 * L3 * delta * Nf * L1 
        
        A_a = A_f + A_p #[m2] #Heat transfer area airside
        
        # Volume occupied by the heat exchanger (heat exchanger total volume) (m^3)
        V_a = L1 * L2 * L3
        
        ## Outside geometric characteristics

        # Minimum free-flow area (Fundamentals of Heat Exchanger Design-Shah pg 573) 

        # 2a''
        a_prime = (xt - do) - (xt - do) * delta * Nf 

        # 2b''
        b_prime = 2 * (((xt / 2) ** 2 + xl ** 2) ** 0.5 - do - (xt - do) * delta * Nf)

        # c''
        if a_prime < b_prime:
            c_prime = a_prime
        else:
            c_prime = b_prime

        # Minimum free-flow area (m^2)
        A_o_a = ((L3 / xt - 1) * c_prime + (xt - do) - (xt - do) * delta * Nf) * L1

        # Frontal area (m^2)
        A_fr_a = L1 * L3

        # Ratio of free flow area to frontal area
        sigma_a  = A_o_a / A_fr_a

        # surface area density 
        alpha_a = A_a / V_a

        # Hydralic diameter (m)
        D_h_a = 4 * sigma_a / alpha_a

        #Need Adjusted airspeed based on obstructed area

        adjspeed = V_extr / sigma_a

        #-------------------------------------------------------------------------#
        # Refrigerant Constants (R410a) 
        #-------------------------------------------------------------------------#

        k_f  = CP.PropsSI('L', 'P', P, 'Q', 0, 'R410a') # [W/m-K] 
        k_g  = CP.PropsSI('L', 'P', P, 'Q', 1, 'R410a') # [W/m-K] 
        mu_f = CP.PropsSI('V', 'P', P, 'Q', 0, 'R410a') # [Pa-s] 
        mu_g = CP.PropsSI('V', 'P', P, 'Q', 1, 'R410a') # [Pa-s]
        c_p_f = CP.PropsSI('C', 'P', P, 'Q', 0, 'R410a') #[J/kg-K]  
        c_p_g = CP.PropsSI('C', 'P', P, 'Q', 1, 'R410a') #[J/kg-K]           

        #-------------------------------------------------------------------------#
        # Air Constants
        #-------------------------------------------------------------------------#

        k_a = CP.PropsSI('L', 'P', 101325, 'T', T_air, 'air') #[W/m-K]   
        mu_a = CP.PropsSI('V', 'P', 101325, 'T', T_air, 'air') #[Pa-s]   
        rho_a = CP.PropsSI('D', 'P', 101325, 'T', T_air, 'air') #[kg/m3] 
        c_p_a = CP.PropsSI('C', 'P', 101325, 'T', T_air, 'air') #[J/kg-K]
        Pr_a = CP.PropsSI('Prandtl', 'P', 101325, 'T', T_air, 'air') #[J/kg-K]

        #-------------------------------------------------------------------------#
        # Derived Relations
        #-------------------------------------------------------------------------#
        
        # fluid mass velocity (kg/(m^2 s))
        G_a = A_fr_a * V_extr * rho_a / A_o_a

        # Compute Reynold's number
        Re_air = G_a * D_h_a / mu_a

        # Compute j using equation 7.141 Nr >= 2 (Fundamentals of Heat Exchanger Design-Shah pg 551)

        # collar diameter (m)
        dc = do + 2 * delta

        # fluid mean axial velocity (m/s)
        u_m = adjspeed

        # Collar Reynolds number
        Re_dc = rho_a * u_m * dc / mu_a

        # fin pitch (m/fin)
        pf = 1 / Nf

        # constants from equation
        C3 = -0.361 - 0.042 * Nr / np.log(Re_dc) + 0.158 * np.log(Nr * (pf / dc) **0.41)

        C4 = -1.224 - (0.076 * (xl / D_h_a) ** 1.42) / np.log(Re_dc)

        C5 = -0.083 + 0.058 * Nr / np.log(Re_dc)

        C6 = -5.735 + 1.21 * np.log(Re_dc / Nr)

        # Compute outside heat transfer coefficeinet using colburn j factor (more accurate)
        j = 0.086 * Re_dc ** C3 * Nr ** C4 * (pf / dc) ** C5 * (pf / D_h_a) ** C6 * (pf / xt) ** -0.93

        # h = JGCp/Pr^2/3
        h_a = j * G_a * c_p_a / Pr_a**(2/3)

        # Single fin efficiency 
        # (Fundamentals of Heat Exchanger Design-Shah pg 606 eqn 9.14)
        m = (2 * h_a / k_pipe / delta) ** 0.5

        l = xl / 2 - delta 

        # Determine single fin efficiency
        eta_f = np.tanh(m * l) / (m * l)
        
        #Overall Fin efficiency
        fin_eff = 1 - (1 - eta_f) * A_f / A_a
#         print(j)
#         print(h_a)
#         print(A_a)
#         print(fin_eff)
        
        addcnst = A_i * R_tw + A_i / (h_a * fin_eff * A_a)

        #HT-coefficient, gaseous, contribution from refrigerant side
        Re_g  =  4 * m_dot_g / (np.pi * di * mu_g)
        Pr_g  =  c_p_g * mu_g / k_g
        Nu_g  =  Circular_Duct_Nu([Re_g], [Pr_g], 'c')  
        h_i_g =  k_g * Nu_g / di


        #HT-coefficient, liquid, contribution from refrigerant side
        Re_f  =  4 * m_dot_f / (np.pi * di * mu_f)
        Pr_f  =  c_p_f * mu_f / k_f
        Nu_f  =  Circular_Duct_Nu([Re_f], [Pr_f], 'c')  
        h_i_f =  k_f * Nu_f / di


        #Local overall heat transfer coefficient
        U_g = (1 / h_i_g + addcnst)**-1
        U_f = (1 / h_i_f + addcnst)**-1

        #Output UA
        UA_g = U_g*A_i
        UA_f = U_f*A_i

    elif subsys == 'COND':

        # Geometric Characteristics
        
        # Fin density (fins/m) [measured 15 fins per inch]
        Nf = 19 / 0.0254

        # Outside diameter of tubing (m) [measured .21"]
        do = 0.21 * 0.0254

        # Inside diameter of tubing (m) [wall thickness estimated at 0.03"]
        di = do - 2 * 0.03 *.0254

        # Transverse spacing between tubes (m) [measured 1.048" - do]
        xt = 1.048 * 0.0254 - do

        # Longitudinal spacing between tubes (m) [(measured 1.066" - do) / 2]
        xl = (1.066 * 0.0254 - do) / 2

        # Fin thickness (m) [measured 0.004"]
        delta = 0.004 * 0.0254

        # Overall Length (m) [measured 15 15/16"]
        L1 = (15 + 15/16) * 0.0254
        
        # Overall depth (m) [measured 1 5/16"]
        L2 = (1 + 5/16) * 0.0254

        # Overall height (m) [measured 12.5"]
        L3 = 12.5 * 0.0254
        
        # Number of Rows 
        Nr = 3
        
        # Number of tubes
        Nt = 44


        #Interior (refrigerant side)
        A_i     = np.pi * di * Nt * L1 # [m2]

        #Pipe Wall
        k_pipe = 385 # copper [W/m-K]

        R_tw = np.log(do / di) / (2 * np.pi * k_pipe * Nt * L1) # [K/W]

        #Exterior (air side)
        
        # Primary surface area (tubes and header plates)
        A_p = np.pi * do * (L1 - delta * Nf * L1) * Nt + 2 * (L3 * L2 - np.pi * do**2 / 4 * Nt)

        # Secondary surface area (fins)
        A_f = 2 * (L2 * L3 - (np.pi * do**2 / 4) * Nt) * Nf * L1 + 2 * L3 * delta * Nf * L1 
        
        A_a = A_f + A_p #[m2] #Heat transfer area airside
        
        # Volume occupied by the heat exchanger (heat exchanger total volume) (m^3)
        V_a = L1 * L2 * L3
        
        ## Outside geometric characteristics

        # Minimum free-flow area (Fundamentals of Heat Exchanger Design-Shah pg 573) 

        # 2a''
        a_prime = (xt - do) - (xt - do) * delta * Nf 

        # 2b''
        b_prime = 2 * (((xt / 2) ** 2 + xl ** 2) ** 0.5 - do - (xt - do) * delta * Nf)

        # c''
        if a_prime < b_prime:
            c_prime = a_prime
        else:
            c_prime = b_prime

        # Minimum free-flow area (m^2)
        A_o_a = ((L3 / xt - 1) * c_prime + (xt - do) - (xt - do) * delta * Nf) * L1

        # Frontal area (m^2)
        A_fr_a = L1 * L3

        # Ratio of free flow area to frontal area
        sigma_a  = A_o_a / A_fr_a

        # surface area density 
        alpha_a = A_a / V_a

        # Hydralic diameter (m)
        D_h_a = 4 * sigma_a / alpha_a

        #Need Adjusted airspeed based on obstructed area

        adjspeed = V_extr / sigma_a

        #-------------------------------------------------------------------------#
        # Refrigerant Constants (R410a) 
        #-------------------------------------------------------------------------#

        k_f  = CP.PropsSI('L', 'P', P, 'Q', 0, 'R410a') # [W/m-K] 
        k_g  = CP.PropsSI('L', 'P', P, 'Q', 1, 'R410a') # [W/m-K] 
        mu_f = CP.PropsSI('V', 'P', P, 'Q', 0, 'R410a') # [Pa-s] 
        mu_g = CP.PropsSI('V', 'P', P, 'Q', 1, 'R410a') # [Pa-s]
        c_p_f = CP.PropsSI('C', 'P', P, 'Q', 0, 'R410a') #[J/kg-K]  
        c_p_g = CP.PropsSI('C', 'P', P, 'Q', 1, 'R410a') #[J/kg-K]           

        #-------------------------------------------------------------------------#
        # Air Constants
        #-------------------------------------------------------------------------#

        k_a = CP.PropsSI('L', 'P', 101325, 'T', T_air, 'air') #[W/m-K]   
        mu_a = CP.PropsSI('V', 'P', 101325, 'T', T_air, 'air') #[Pa-s]   
        rho_a = CP.PropsSI('D', 'P', 101325, 'T', T_air, 'air') #[kg/m3] 
        c_p_a = CP.PropsSI('C', 'P', 101325, 'T', T_air, 'air') #[J/kg-K]
        Pr_a = CP.PropsSI('Prandtl', 'P', 101325, 'T', T_air, 'air') #[J/kg-K]

        #-------------------------------------------------------------------------#
        # Derived Relations
        #-------------------------------------------------------------------------#
        
        # fluid mass velocity (kg/(m^2 s))
        G_a = A_fr_a * V_extr * rho_a / A_o_a

        # Compute Reynold's number
        Re_air = G_a * D_h_a / mu_a

        # Compute j using equation 7.141 Nr >= 2 (Fundamentals of Heat Exchanger Design-Shah pg 551)

        # collar diameter (m)
        dc = do + 2 * delta

        # fluid mean axial velocity (m/s)
        u_m = adjspeed

        # Collar Reynolds number
        Re_dc = rho_a * u_m * dc / mu_a

        # fin pitch (m/fin)
        pf = 1 / Nf

        # constants from equation
        C3 = -0.361 - 0.042 * Nr / np.log(Re_dc) + 0.158 * np.log(Nr * (pf / dc) **0.41)

        C4 = -1.224 - 0.076 * (xl / D_h_a) ** 1.42 / np.log(Re_dc)

        C5 = -0.083 + 0.058 * Nr / np.log(Re_dc)

        C6 = -5.735 + 1.21 * np.log(Re_dc / Nr)

        # Compute outside heat transfer coefficeinet using coburn j factor (more accurate)
        j = 0.086 * Re_dc ** C3 * Nr ** C4 * (pf / dc) ** C5 * (pf / D_h_a) ** C6 * (pf / xt) ** -0.93

        # h = JGCp/Pr^2/3
        h_a = j * G_a * c_p_a / Pr_a ** (2/3)

        # Single fin efficiency 
        # (Fundamentals of Heat Exchanger Design-Shah pg 606 eqn 9.14)
        m = (2 * h_a / k_pipe / delta) ** 0.5

        l = xl / 2 - delta 

        # Determine single fin efficiency
        eta_f = np.tanh(m * l) / (m * l)
        
        #Overall Fin efficiency
        fin_eff = 1 - (1 - eta_f) * A_f / A_a
        
        
#         print(j)
#         print(h_a)
#         print(A_a)
#         print(fin_eff)
        
        addcnst = A_i * R_tw + A_i / (h_a * fin_eff * A_a)

        #HT-coefficient, gaseous, contribution from refrigerant side
        Re_g  =  4 * m_dot_g / (np.pi * di * mu_g)
        Pr_g  =  c_p_g * mu_g / k_g
        Nu_g  =  Circular_Duct_Nu([Re_g], [Pr_g], 'h')  
        h_i_g =  k_g * Nu_g / di


        #HT-coefficient, liquid, contribution from refrigerant side
        Re_f  =  4 * m_dot_f / (np.pi * di * mu_f)
        Pr_f  =  c_p_f * mu_f / k_f
        Nu_f  =  Circular_Duct_Nu([Re_f], [Pr_f], 'h')  
        h_i_f =  k_f * Nu_f / di


        #Local overall heat transfer coefficient
        U_g = ( 1 / h_i_g + addcnst )**-1;
        U_f = ( 1 / h_i_f + addcnst )**-1;


        #Output UA
        UA_g = U_g * A_i
        UA_f = U_f * A_i

    else:
        raise ValueError('Subsys must be "COND" or "EVAP"')


    return [UA_g, UA_f]


def Condenser_Proc(input_state, strarg, flowrate, T_amb):


    # Input state must be a row vector containing pressure 
    # and enthalpy in that order
    # input_state = [P, h]

    # Input state could be a row vector containing pressure 
    # and temperature in that order
    # input_state = [P, T]


    #Artificial Inpu
    airspeed = 5.2


    #Initialize Vars
    #----------------------
    P_in = input_state[0]
    P = P_in * np.ones(4)
    h = np.zeros(4)
    T = np.zeros(4)
    s = np.zeros(4)

    abcissa = np.zeros(4)
    dz_1 = 0
    dz_2 = 0
    dz_3 = 0

    #=========================================================================#
    # set up us the properties

    if strarg == 'h':

        h_in = input_state[1]
        T_in = CP.PropsSI('T', 'P', P_in, 'H', h_in, 'R410a')
        T_sat = CP.PropsSI('T', 'P', P_in, 'Q', 1, 'R410a')
        h_f   = CP.PropsSI('H', 'P', P_in, 'Q', 0, 'R410a')
        h_g   = CP.PropsSI('H', 'P', P_in, 'Q', 1, 'R410a')
        h_fg  = h_g - h_f    
 
    
#         T_in  = fsolve(lambda t: ((h_in-h_g) - SuperHT_Cp_integral(T_sat, t)), 
#                        T_sat + 1)


        # assign output
        #----------------
        T[0] = T_in;
        h[0] = h_in;
        #----------------

    elif strarg == 'T':

        T_in = input_state[1];
        h_in = CP.PropsSI('H', 'P', P_in, 'T', T_in, 'R410a')
        T_sat = CP.PropsSI('T', 'P', P_in, 'Q', 1, 'R410a')
        h_f   = CP.PropsSI('H', 'P', P_in, 'Q', 0, 'R410a')
        h_g   = CP.PropsSI('H', 'P', P_in, 'Q', 1, 'R410a')
        h_fg  = h_g - h_f;

#         h_in  = h_g + SuperHT_Cp_integral(T_sat, T_in);


        # assign output
        #----------------
        T[0] = T_in;
        h[0] = h_in;
        #----------------

    else:
        raise ValueError('dont recognize input property' + strarg)



    #=========================================================================#
    # Calculate Vars
    #

    [UA_1, UA_3] = generate_HTCOEFF( P_in, flowrate, flowrate, 
                                       airspeed, 'COND', T_amb)

    #Temporary
    UA_g = UA_1
    UA_f = UA_3

    #Properties
    c_p_g = 0.5 * (CP.PropsSI('C', 'P', P_in, 'T', T_in, 'R410a') + 
                   CP.PropsSI('C', 'P', P_in, 'Q', 1, 'R410a'))
    c_p_f = CP.PropsSI('C', 'P', P_in, 'Q', 0, 'R410a')

    rho_g   = CP.PropsSI('D', 'P', P_in, 'Q', 1, 'R410a')
    rho_f   = CP.PropsSI('D', 'P', P_in, 'Q', 0, 'R410a')
    #rho_fg  = rho_f - rho_g; or rho_g - rho_f?
    rho_rat = rho_g/rho_f

    #Vol Void Frac
    gamma = 1 / (1 - rho_rat) + rho_rat / (rho_rat - 1)**2 * np.log( rho_rat )

    UA_2 = UA_f * (1 - gamma) + UA_g * (gamma)




    #=========================================================================#
    #
    #  begin integration procedure, piecewise
    #
    #

    #--- Superheat-into-Saturation Process ---

    dz_1 = c_p_g  * flowrate / UA_1 * np.log((T_amb - T_in) / 
                                             (T_amb - T_sat))

    #Add exception if superheated phase takes up the
    #entire HX domain
    if (dz_1 > 1):
        T = np.nan
        h = np.nan
        P = np.nan
        abcissa = np.nan
        raise ValueError('no exception when superheated' +
                         ' phase takes up entire domain')

    # assign output
    #-----------------
    T[1] = T_sat
    h[1] = h_g
    #-----------------


    #--- SatVap-into-SatLiq Process ---

    dz_2 = flowrate * h_fg / (UA_2 * (T_sat - T_amb))

        #Begin exception if saturation phase takes up the 
        #remainder of the HX domain
    if (dz_1 + dz_2) > 1:

        dz_2   = 1 - dz_1

        #solve system 
        #gamma and delta_h are the variables\
        x = lambda dh: (dh + h_fg) / h_fg # x(var[1])
        f = lambda var: [dz_2 * (T_amb - T_sat) * 
                         (UA_f + (UA_g - UA_f) * var[0]) - 
                         (flowrate * var[1]),

                         (1 - x(var[1])) * 
                         (1 / (1 - rho_rat) - var[0]) + 
                         rho_rat / (rho_rat - 1)**2 * 
                         np.log(rho_rat + (1 - rho_rat) * 
                                x(var[1]))
                        ]

        b = fsolve( f, [gamma, -h_fg] )
        # gamma = b(1);
        dh_2  = b[1]

        #-----------------
        # Produce Output
        #
        h_out = h_g + dh_2;
        #
        # assign output
        #-----------------
        T[2] = T_sat
        h[2] = h_out
        T[3] = T[2]
        h[3] = h[2]
        #-----------------

        #Otherwise go to subcool process  
    else:

        # assign output
        #-----------------
        T[2] = T_sat
        h[2] = h_f
        #-----------------      



    #--- SatLiq-into-Subcool Process ---        

    dz_3 = 1 - dz_1 - dz_2

    T_out = (T_sat - T_amb) * np.exp(-UA_3 / (c_p_f * 
                                              flowrate) * dz_3) + T_amb
    h_out = h_f + c_p_f * (T_out - T_sat)


    # assign output
    #-----------------
    T[3] = T_out;
    h[3] = h_out;
    # Pressure drop determined empirically applied linearly
    P[1] = P[0] - 5e4 * dz_1
    P[2] = P[1] - 5e4 * dz_2
    P[3] = P[2] - 5e4 * (1 - dz_2 + dz_1)
    #-----------------


    # assign output
    #-----------------------------------
    abcissa[1] = abcissa[0] + dz_1
    abcissa[2] = abcissa[1] + dz_2
    abcissa[3] = 1
    #-----------------------------------    

    s = CP.PropsSI('S', 'P', P, 'H', h, 'R410a')


    return [P, T, h, s, abcissa]


def valve_func( CA_param, P_up, P_down, x):

    # CA_par : [m2]  dimensional parameter
    # P_up   : [kPa] upstream press
    # P_down : [kPa] downstream press
    # x      : [  ]  valve opening fraction

    # At 0.80 valve opening we have the rated value

    # Density 
    rho_v     = CP.PropsSI('D', 'P', P_up, 'Q', 0, 'R410a')

    # Mass flow rate
    m_dot = CA_param * ( x / 0.80 ) * np.sqrt( rho_v * (P_up - P_down) )

    return  m_dot


def capillary_tube_func(P_in, h_in, T_in):
# D.A. Wolf, R.R. Bittle, M.B. Pate, Adiabatic capillary tube 
# performance with alternative refrigerants, 
# ASHRAE final report No. RP-762, 1995.

    # 1/16" in OD copper tubing, .02" wall thickness
    D_c = (1/16 - 0.02 * 2) * 0.0254 
    
    # length of capillary tube.  in diameter coil, 4 loops, 2 tubes.
    L_c = 3 * 0.0254 * np.pi  * 4 * 2

    # delta subcool
    T_SC = CP.PropsSI('T', 'P', P_in, 'Q', 0, 'R410a') - T_in

    # Dynamic viscosity of r-410a fluid at inlet temperature
    mu_f = CP.PropsSI('V', 'T', T_in, 'Q', 0, 'R410a')

    # Dynamic viscosity of r-410a vapor at inlet temperature
    mu_g = CP.PropsSI('V', 'T', T_in, 'Q', 1, 'R410a')

    # Density of r-410a fluid at inlet temperature
    rho_f = CP.PropsSI('D', 'T', T_in, 'Q', 0, 'R410a')

    # Density of r-410a vapor at inlet temperature
    rho_g = CP.PropsSI('D', 'T', T_in, 'Q', 1, 'R410a')

    # Specific volume of r-410a fluid at inlet temperature
    v_f = 1 / rho_f

    # Specific volume of r-410a vapor at inlet temperature
    v_g = 1 / rho_g

    # Saturated liquid surface tension of r-410a vapor at inlet temperature
    sigma = CP.PropsSI('I', 'T', T_in, 'Q', 0, 'R410a')

    # Enthalpy of vaporization at inlet temperature
    h_fgc = (CP.PropsSI('H', 'T', T_in, 'Q', 1, 'R410a') - 
             CP.PropsSI('H', 'T', T_in, 'Q', 0, 'R410a'))

    # Enthalpy of fluid at inlet pressure
    h_f = CP.PropsSI('H', 'P', P_in, 'Q', 0, 'R410a')

    # Enthalpy of vaporization at inlet pressure
    h_fg = (CP.PropsSI('H', 'P', P_in, 'Q', 1, 'R410a') - 
             CP.PropsSI('H', 'P', P_in, 'Q', 0, 'R410a'))
    
    C_pfc = CP.PropsSI('C', 'P', P_in, 'Q', 0, 'R410a')

    # A generalized continuous empirical correlation for predicting refrigerant
    # mass flow rates through adiabatic capillary tubes

    pi_1 = L_c / D_c
    pi_2 = D_c **2 * h_fgc / v_f**2 / mu_f**2
    pi_4 = D_c**2 * P_in /  v_f / mu_f**2
    if h_in > h_f:
        pi_5 = (h_in - h_f) / h_fg
    else:    
        pi_5 = pi_2 = D_c **2 * C_pfc * T_SC / v_f**2 / mu_f**2
    pi_6 = v_g / v_f 
    pi_7 = (mu_f - mu_g) / mu_g

    # Check if it is subcooled or mixture
    # mixture
    if h_in > h_f:
        # two-phase
        pi_8 = 187.27 * (pi_1**-0.635 * pi_2**-0.189 * pi_4**0.645 * 
                         pi_5**-0.163 * pi_6**-0.213 * pi_7**-0.487)

    else: # subcooled
        pi_8 = 1.8925 * (pi_1**-0.484 * pi_2**-0.824 * pi_4**1.369 * 
                         pi_5**0.0187 * pi_6**0.773 * pi_7**0.265)
        

    
    m_dot = pi_8 * D_c * mu_f
    

    return m_dot 


def Evap_Proc(input_state, flowrate, T_pod):


    # Input state must be a row vector containing pressure 
    # and enthalpy in that order
    # input_state = [P, h]


    # Artificial Input
    airspeed = 3.2/3 #[m/s]

    #
    # Initialize Vars
    #----------------------
    P_in = input_state[0]
    P = P_in * np.ones(4)
    h = np.zeros(4)
    T = np.zeros(4)
    s = np.zeros(4)

    abcissa = np.zeros(4)
    dz_1 = 0
    dz_2 = 0
    dz_3 = 0


    #=========================================================================#
    # set up us the properties
    #
    h_in  = input_state[1]

    T_sat = CP.PropsSI('T', 'P', P_in, 'Q', 1, 'R410a')
    h_f   = CP.PropsSI('H', 'P', P_in, 'Q', 0, 'R410a')
    h_g   = CP.PropsSI('H', 'P', P_in, 'Q', 1, 'R410a')
    h_fg  = h_g - h_f    


    #=========================================================================#
    # Calculate Vars
    #

    [UA_1, UA_3] = generate_HTCOEFF( P_in, flowrate, flowrate, airspeed, 'EVAP', T_pod);

    #Temporary
    UA_g = UA_3;
    UA_f = UA_1;


    #Properties
    c_p_g = CP.PropsSI('C', 'P', P_in, 'Q', 1, 'R410a')
    c_p_f = CP.PropsSI('C', 'P', P_in, 'Q', 0, 'R410a')

    rho_g   = CP.PropsSI('D', 'P', P_in, 'Q', 1, 'R410a')
    rho_f   = CP.PropsSI('D', 'P', P_in, 'Q', 0, 'R410a')
    #rho_fg  = rho_f - rho_g; or rho_g - rho_f?
    rho_rat = rho_g / rho_f




    #=========================================================================#
    #
    #  begin integration procedure, piecewise
    #
    #=

    if h_in >= h_f:  #There is no subcooled region

        dz_1 = 0;
        # assign output
        #----------------
        T[0] = T_sat;
        h[0] = h_in;
        T[1] = T_sat;
        h[1] = h_in;
        #----------------
        
        #Vol Void Frac
        x_in  = (h_in - h_f) / h_fg
        gamma = (1 / ( 1 - rho_rat) + rho_rat / (rho_rat - 1)**2 / 
                 (1 - x_in) * np.log( x_in - rho_rat * (x_in - 1)))

        #Twophase region HT coeff
        UA_2 = UA_f * (1 - gamma) + UA_g * (gamma)



    else: #calculate subcooled region
    #--- Subcooled-into-SatLiq Process ---

        T_in = T_sat + (h_in - h_f) / c_p_f

        dh_1 = h_f - h_in;
        dz_1 = (c_p_f * flowrate / UA_1 ) * np.log( (T_pod - T_in) / (T_pod - T_sat));

        # assign output
        #----------------
        T[0] = T_in;
        h[0] = h_in;
        T[1] = T_sat;
        h[1] = h_f;
        #----------------

        #Vol Void Frac
        x_in  = 0;
        gamma = (1 / (1 - rho_rat) + rho_rat / (rho_rat - 1)**2 / 
                 (1 - x_in) * np.log( x_in - rho_rat*(x_in-1)))

        #twophase region HT coeff.
        UA_2 = UA_f * (1 - gamma) + UA_g * (gamma)


    #--- SatLiq-into-SatVap Process ---

    dh_2 = h_g - h[1]
    dz_2 = flowrate * dh_2 / (UA_2 * (T_pod - T_sat))

        #Begin exception if saturation phase takes up the 
        #remainder of the HX domain
    if (dz_2) > (1 - dz_1):
        warnings.warn('Partial Evaporation')

        dz_2 = (1 - dz_1)
            #Solve system for dh_1 and gamma
        x_out = lambda dh: (dh + h[1] - h_f) / h_fg
        f = lambda var: [dz_2 * (T_pod - T_sat) * 
                         (UA_f + ( UA_g - UA_f) * var[0]) - 
                         (flowrate * var[1]),
                         
                         (x_out(var[1]) - x_in) * 
                         (1 / (1 - rho_rat) - var[0]) - 
                         rho_rat / (rho_rat - 1)**2 * 
                         np.log((rho_rat * (x_out(var[1]) - 1) - x_out(var[1])) / 
                                (rho_rat * (x_in - 1) - x_in))
                        ]

        b = fsolve( f, [gamma, h_fg/2])
        #gamma = b(1)
        dh_2  = b[1]

        #-----------------
        # Produce Output
        #
        h_out = h_in + dh_2 
        #
        # assign output
        #-----------------
        T[2] = T_sat;
        h[2] = h_out;
        T[3] = T_sat;
        h[3] = h_out;
        #-----------------


    
    else: # Otherwise go to superheat process  
        # assign output
        #-----------------
        T[2] = T_sat
        h[2] = h_g
        #-----------------      



        #--- SatLiq-into-Subcool Process ---        

        dz_3 = 1 - dz_2 - dz_1
        T_out = (T_sat - T_pod) * np.exp(-UA_3 / (c_p_g * flowrate) * dz_3 ) + T_pod
        
#         h_out = h_g + SuperHT_Cp_integral(T_sat, T_out)
        h_out = CP.PropsSI('H', 'T', T_out, 'P', P_in, 'R410a')

        # assign output
        #-----------------
        T[3] = T_out
        h[3] = h_out
        #-----------------
    

    # assign output
    # Pressure drop determined empirically applied linearly
    P[1] = P[0] - 5e4 * (dz_1)
    P[2] = P[1] - 5e4 * (dz_2)
    P[3] = P[2] - 5e4 * (1 - dz_2 + dz_1)
    #-----------------------------------
    abcissa[1] = abcissa[0] + dz_1
    abcissa[2] = abcissa[1] + dz_2
    abcissa[3] = 1
    #-----------------------------------    
    
    s = CP.PropsSI('S', 'P', P, 'H', h, 'R410a')


    return [P, T, h, s, abcissa]