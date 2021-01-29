import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import pandas as pd
import os


# Create fit function for pressure transducers using pressure cal coefficients
pressureFit = np.poly1d([125 * 6.89476,  -62.5 * 6.89476]) * 1000 

def experimental_analysis_tunnel(file, P_amb, Input_Q):
    # A function which performs analysis of steady state values for 
    # the pressure and temperature measurements taken during the prototype 
    # experiments with wind tunnel used for air flow across condenser. 
    
    # file: Location of experimental data file.
    # P_amb: Ambient pressue measured with altimeter.
    # Input_Q: Input Power to heating element. 
    
    
    # Import Radial Test Data from experiment
    refrigData = pd.read_csv(file, sep='\t', header=21)

    # Create time column with time step and index
    refrigData['Time (s)'] = refrigData['X_Value'] * \
        refrigData['Time Step (s)'][0]

    # Create pressure columns with transducer fit function
    refrigData['Pressure Transducer 1 (PaG)'] = pressureFit(
        refrigData['Pressure  1'])
    refrigData['Pressure Transducer 2 (PaG)'] = pressureFit(
        refrigData['Pressure 2'])
    refrigData['Pressure Transducer 3 (PaG)'] = pressureFit(
        refrigData['Pressure 3'])
    refrigData['Pressure Transducer 4 (PaG)'] = pressureFit(
        refrigData['Pressure 4'])

    # Calculate mean ambient temperature
    T_amb = refrigData['Thermistor (C)'][-600:-1].mean()  + 273.15

    # Compute wind speed
    windSpeed = np.sqrt(refrigData['Pitot Tube (Torr)'].abs(
    ) * 133.322 * 2 / CP.PropsSI('D', 'T', T_amb, 'P', P_amb, 'Air.mix')).mean()

    # Compute mean steady state radial temperatures from last 600 seconds
    radialProfile = refrigData[['Thermocouple 2 (C)', 'Thermocouple 4 (C) ', 'Thermocouple 7 (C)']][-600:-1].mean()
    
    # Plot Comparison of Radial Temmperature Profiles
    locations = np.array([1.5, 7.5, 16.5])

    # Compute area for each thermocouple
    area = (np.array([4, 12, 18]) / 100)**2 - (np.array([0, 4, 12]) / 100)**2

    # compute mean internal temperature
    T_mean_pod = (radialProfile*area).sum() / \
        (locations[-1] / 100 + 1.5 / 100)**2


    # Calculate steady state mean pressures and temperatures
    pressures = refrigData[['Pressure Transducer 1 (PaG)', 'Pressure Transducer 3 (PaG)', 'Pressure Transducer 2 (PaG)',
                            'Pressure Transducer 4 (PaG)']][-600:-1].mean().values + P_amb  # Pressure from barometer

    pressures = pressures[[1, 2, 3, 0]]

    temperatures = refrigData[['Temperature  1', 'Temperature 2', 'Temperature 3', 'Temperature 4'
                               ]][-600:-1].mean().values + 273.15

    temperatures = temperatures[[1, 2, 3, 0]]
    
    # Calculate compensation due to temperature of refrigerant being measured on the outside surface of the tubing
    # http://www.burnsengineering.com/local/uploads/files/small_diameter_lines.pdf
    temperatures[0:3] = temperatures[0:3] + 0.02 * (temperatures[0:3] - T_amb)

    temperatures[3] = temperatures[3] + 0.02 * (temperatures[3] - T_mean_pod-273.15)
    
    # Look up saturated enthalpy and entropy
    fluid = 'R410a'


    cycleEnthalpy = CP.PropsSI('H', 'P', pressures, 'T', temperatures, fluid)

    cycleEntropy = CP.PropsSI('S', 'P', pressures, 'T', temperatures, fluid)

    # Compute vapor quality if pre-evap temperature below saturation temp
    if np.isinf(cycleEnthalpy[3]):
        x = (pressures[3] - CP.PropsSI('P', 'T', temperatures[3], 'Q', 1, fluid)) / \
        (CP.PropsSI('P', 'T', temperatures[3], 'Q', 0, fluid) - 
         CP.PropsSI('P', 'T', temperatures[3], 'Q', 1, fluid))
        
        cycleEnthalpy[3] = CP.PropsSI('H', 'P', pressures[3], 'Q', x, fluid)
        cycleEntropy[3] = CP.PropsSI('S', 'P', pressures[3], 'Q', x, fluid)
    
    
    # Model pod as cylindrical prism 
    # inside radius [7.75"]
    r_i = 7.75 * 0.0254
    # outside radius [8.25"]
    r_o = 8.25 * 0.0254
    
    # thermal conductivity of plywood (W/mk)
    k_ply = 0.13
    
    # effective thermal conductivity of interior convection (W/mk)
    k_eff = 4.91
    
    # length of prototype (m)
    L = 65 * 0.0254 / 2
    
    # Free stream velocity (m/s)
    U = windSpeed
    
    # absolute viscosity of air (Pa*s)
    mu = CP.PropsSI('V', 'T', T_amb, 'P', P_amb, 'Air.mix')
    
    # Density of air (kg/m^3) 
    rho = CP.PropsSI('D', 'T', T_amb, 'P', P_amb, 'Air.mix')
    
    # Prandtl number of air
    Pr = CP.PropsSI('Prandtl', 'T', T_amb, 'P', P_amb, 'Air.mix')
    
    # Reynolds number outside 
    Re = rho * U * L / mu
    
    # Temperature difference between ambient and pod (K)
    delta_T = (T_mean_pod + 273.15 - T_amb)
    
    # Incropera, Frank P.; DeWitt, David P. (2007). 
    # Fundamentals of Heat and Mass Transfer (6th ed.). 
    # Hoboken: Wiley. pp. 490, 515. ISBN 978-0-471-45728-2.
    if Re < 5e5:
        Nu = 0.664 * Re**0.5 * Pr**(1/3)
    else:
        raise ValueError('Re is turbulent')
    # Conductivity of air
    k_air  = CP.PropsSI('L', 'T', T_amb, 'P', P_amb, 'Air.mix')
    
    # convection coefficient
    h_air = Nu * k_air / L
    
    # Ambient heat load (W)
    ambient_Q = (T_mean_pod + 273.15 - T_amb) / (r_i**2 * ( 1 / 6 / k_eff + np.log(r_o/r_i) / 2 / k_ply +  1 / 2 / r_o / h_air))
    
    # Compute VCRC heat load 
    load = (Input_Q - ambient_Q)
    
    # Creae pandas dataframe for 
    experimentalData = pd.DataFrame({'Ambient P (Pa)': P_amb, 'Ambient T (K)': T_amb, 'P (Pa)': [pressures], 
                                     'T (K)': [temperatures], 'h (j/kg)':[cycleEnthalpy], 
                                     's (j/kg K)': [cycleEntropy], 'Pod T Profile (K)': [radialProfile.values], 
                                     'Pod T (K)': [T_mean_pod + 273.15], 'Wind Tunnel Velocity (m/s)': windSpeed, 
                                     'Heating Element Power (W)': Input_Q, 'Ambient Heat Load (W)': ambient_Q, 
                                     'Total Heat Load (W)': load, 'file': file,})

    

    return experimentalData

def experimental_analysis_fan(file, P_amb, Q_element, W_refrig):
    # A function which performs analysis of steady state values for 
    # the pressure and temperature measurements taken during the prototype 
    # experiments with fan used for air flow across condenser. 
    
    # file: Location of experimental data file.
    # P_amb: Ambient pressue measured with altimeter.
    # Q_element: Input Power to heating element.
    # W_refrig: Power consumed by blower and compressor
    
    
    # Import Radial Test Data from experiment
    refrigData = pd.read_csv(file, sep='\t', header=21)

    # Create time column with time step and index
    refrigData['Time (s)'] = refrigData['X_Value'] * \
        refrigData['Time Step (s)'][0]

    # Create pressure columns with transducer fit function
    refrigData['Pressure Transducer 1 (PaG)'] = pressureFit(
        refrigData['Pressure  1'])
    refrigData['Pressure Transducer 2 (PaG)'] = pressureFit(
        refrigData['Pressure 2'])
    refrigData['Pressure Transducer 3 (PaG)'] = pressureFit(
        refrigData['Pressure 3'])
    refrigData['Pressure Transducer 4 (PaG)'] = pressureFit(
        refrigData['Pressure 4'])

    # Calculate mean ambient temperature
    T_amb = refrigData['Thermistor (C)'][-600:-1].mean()  + 273.15

    # Compute air speed measured behind condenser (m/s) [not very accurate]
    windSpeed = np.sqrt(refrigData['Pitot Tube (Torr)'].abs(
    ) * 133.322 * 2 / CP.PropsSI('D', 'T', T_amb, 'P', P_amb, 'Air.mix')).mean()

    # Compute mean steady state radial temperatures from last 600 seconds
    radialProfile = refrigData[['Thermocouple 2 (C)', 'Thermocouple 4 (C) ', 'Thermocouple 7 (C)']][-600:-1].mean()
    
    # Plot Comparison of Radial Temmperature Profiles
    locations = np.array([1.5, 7.5, 16.5])

    # Compute area for each thermocouple
    area = (np.array([4, 12, 18]) / 100)**2 - (np.array([0, 4, 12]) / 100)**2

    # compute mean internal temperature
    T_mean_pod = (radialProfile*area).sum() / \
        (locations[-1] / 100 + 1.5 / 100)**2


    # Calculate steady state mean pressures and temperatures
    pressures = refrigData[['Pressure Transducer 1 (PaG)', 'Pressure Transducer 3 (PaG)', 'Pressure Transducer 2 (PaG)',
                            'Pressure Transducer 4 (PaG)']][-600:-1].mean().values + P_amb  # Pressure from barometer

    # Put in proper order: post evap, post compressor, post condenser, post capillary tubes
    pressures = pressures[[1, 2, 3, 0]]

    temperatures = refrigData[['Temperature  1', 'Temperature 2', 'Temperature 3', 'Temperature 4'
                               ]][-600:-1].mean().values + 273.15

    # Put in proper order: post evap, post compressor, post condenser, post capillary tubes
    temperatures = temperatures[[1, 2, 3, 0]]
    
    # Calculate compensation due to temperature of refrigerant being measured on the outside surface of the tubing
    # http://www.burnsengineering.com/local/uploads/files/small_diameter_lines.pdf
    temperatures[0:3] = temperatures[0:3] + 0.02 * (temperatures[0:3] - T_amb)

    temperatures[3] = temperatures[3] + 0.02 * (temperatures[3] - T_mean_pod-273.15)
    
    # Look up saturated enthalpy and entropy
    fluid = 'R410a'

    #Compute Enthalpy and entropy from P and T
    cycleEnthalpy = CP.PropsSI('H', 'P', pressures, 'T', temperatures, fluid)

    cycleEntropy = CP.PropsSI('S', 'P', pressures, 'T', temperatures, fluid)

    # Check if any positions are two phase:
    TP = np.isinf(cycleEnthalpy)
    
    for ind, point in enumerate(TP):
        if point:
            x = (pressures[ind] - CP.PropsSI('P', 'T', temperatures[ind], 'Q', 1, fluid)) / \
            (CP.PropsSI('P', 'T', temperatures[ind], 'Q', 0, fluid) - 
             CP.PropsSI('P', 'T', temperatures[ind], 'Q', 1, fluid))

            cycleEnthalpy[ind] = CP.PropsSI('H', 'P', pressures[ind], 'Q', x, fluid)
            cycleEntropy[ind] = CP.PropsSI('S', 'P', pressures[ind], 'Q', x, fluid)
    
    
    # Model pod as cylindrical prism 
    # inside radius [7.75"]
    r_i = 7.75 * 0.0254
    # outside radius [8.25"]
    r_o = 8.25 * 0.0254
    
    # thermal conductivity of plywood (W/mk)
    k_ply = 0.13
    
    # effective thermal conductivity of interior convection (W/mk)
    k_eff = 4.91
    
    # length of prototype (m)
    L = 65 * 0.0254 / 2
    
    # Free stream velocity (m/s)
    U = windSpeed
    
    # film temperature (K)
    T_f = (radialProfile[2] + 273.15 + T_amb) / 2
    
    # Temperature difference between ambient and pod (K)
    delta_T = np.abs(radialProfile[2] + 273.15 - T_amb)
    
    # Thermal conductivity (W/m K)
    k  = CP.PropsSI('L', 'T', T_f, 'P', P_amb, 'Air.mix')
    
    # isobaric Specific Heat of air (j/kg/k)
    C_p = CP.PropsSI('C', 'T', T_f, 'P', P_amb, 'Air.mix')
    
    # absolute viscosity of air (Pa*s)
    mu = CP.PropsSI('V', 'T', T_f, 'P', P_amb, 'Air.mix')
    
    # Density of air (kg/m^3) 
    rho = CP.PropsSI('D', 'T', T_f, 'P', P_amb, 'Air.mix')
    
    # Kinematic Viscosity (m^2/s)
    nu = mu / rho
    
    # Prandtl number of air
    Pr = CP.PropsSI('Prandtl', 'T', T_f, 'P', P_amb, 'Air.mix')
    
    # Isobaric expansion coefficient (1/K)
    beta = CP.PropsSI('isobaric_expansion_coefficient', 'T', T_f, 'P', P_amb, 'Air.mix')
    
    # gravity (m/s^2)
    g = 9.81
    
    # Thermal diffusivity (m^2/s)
    alpha = k / rho / C_p
    
    # Raleigh Number
    Ra = g * beta * delta_T * (2 * r_o)**3 / nu / alpha
    
    
    # Incropera, Frank P.; DeWitt, David P. (2007). 
    # Fundamentals of Heat and Mass Transfer (6th ed.). 
    # Hoboken: Wiley. pp. 580, 515. ISBN 978-0-471-45728-2.
    if Ra < 10e12:
        Nu = (0.6 + 0.387 * Ra**(1/6)/(1 + (0.559 / Pr)**(9 / 16))**(8 / 27))**2
    else:
        raise ValueError('Ra is too high')
    
    # Conductivity of air
    k_air  = CP.PropsSI('L', 'T', T_amb, 'P', P_amb, 'Air.mix')
    
    # convection coefficient
    h_air = 0.25 * Nu * k_air / L
    
    # Ambient heat load (W)
    Q_ambient = (T_mean_pod + 273.15 - T_amb) / (r_i**2 * (1 / 6 / k_eff + 
                                                                 np.log(r_o / r_i) / 2 / k_ply +  
                                                                 1 / 2 / r_o / h_air))
    
    # Compute VCRC heat load 
    load = (Q_element - Q_ambient)
    
    # Compute compressor by subrtracting blower work. Blower work from test on 01/23 0.08 KWh after 60 min. 
    W_comp = W_refrig - 0.08 * 1000 
    
    # Creae pandas dataframe for 
    experimentalData = pd.DataFrame({'Ambient P (Pa)': P_amb, 'Ambient T (K)': T_amb, 'P (Pa)': [pressures], 
                                     'T (K)': [temperatures], 'h (j/kg)':[cycleEnthalpy], 
                                     's (j/kg K)': [cycleEntropy], 'Pod T Profile (K)': [radialProfile.values], 
                                     'Pod T (K)': [T_mean_pod + 273.15], 'Air Speed Condenser (m/s)': windSpeed, 
                                     'Heating Element Power (W)': Q_element, 'Ambient Heat Load (W)': Q_ambient, 
                                     'Total Heat Load (W)': load, 'Compressor Work (W)': W_comp, 'file': file,})

    

    return experimentalData

def thermodynamic_plots(*args, lgnd = ['Vapor Dome', 'Ambient Temperature', 'Pod Temperature'], annotate = False, color = "", save = False):
    # A function which plots the T-s and P-h diagram of the experimental 
    # measurements of the VCRC and model of the VCRC. If only one arguement is given
    # the function assumes experimental data and plots T-s and P-h numbered points.
    
    # arg 1: Expereimental data in pandas dataframe
    # arg 2: data to compare in pandas dataframe
    txt = ["1'", "2'", "5'", "6'"]
    
    if len(args) == 2:
        exp = args[0] 
        model = args[1]
        fluid = 'R410a'
        # Create Vapor Dome
        vaporDomeS = np.concatenate([CP.PropsSI('S', 'T', np.linspace(200, 344, 150), 'Q', 0, fluid),
                                     CP.PropsSI('S', 'T', np.linspace(344, 200, 150), 'Q', 1, fluid)])
        vaporDomeT = np.concatenate(
            [np.linspace(200, 344, 150), np.linspace(344, 200, 150)])

        vaporDomeH = np.concatenate([CP.PropsSI('H', 'P', np.linspace(2e5, 5000e3, 150), 'Q', 0, fluid),
                                     CP.PropsSI('H', 'P', np.linspace(5000e3, 2e5, 150), 'Q', 1, fluid)])
        vaporDomeP = np.concatenate(
            [np.linspace(2e5, 5000e3, 150), np.linspace(5000e3, 2e5, 150)])


        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        plt.subplot(121)
        plt.plot(vaporDomeS, vaporDomeT, '-')
        plt.plot(model['s (j/kg K)'], model['T (K)'], color + '-o')
        plt.plot(exp['s (j/kg K)'], exp['T (K)'], 'go')
        if annotate == True:
            for ind, s in enumerate(exp['s (j/kg K)']):
                plt.annotate(txt[ind], (exp['s (j/kg K)'][ind] + 10, exp['T (K)'][ind] + 1))
            
        plt.plot([500, 2000],[exp['Ambient T (K)'], exp['Ambient T (K)']], 'c--')
        plt.plot([500, 2000],[exp['Pod T (K)'], exp['Pod T (K)']], 'r--')
        plt.ylabel('Temperature (K)')
        plt.xlabel('Entropy (j/kg/K)')
        plt.legend(lgnd)


        plt.title('Prototype Refrigeration Sys. T-s ')
        plt.ylim(exp['T (K)'].min() - 10)

        plt.subplot(122)
        plt.plot(vaporDomeH, vaporDomeP, '-')
        plt.plot(model['h (j/kg)'], model['P (Pa)'], color + '-o')
        plt.plot(exp['h (j/kg)'], exp['P (Pa)'], 'go')
        if annotate == True:
            for ind, h in enumerate(exp['h (j/kg)']):
                plt.annotate(txt[ind], (exp['h (j/kg)'][ind] + 1e3, exp['P (Pa)'][ind] + 0.1e5))
            
        plt.ylabel('Pressure (Pa)')
        plt.xlabel('Enthalpy (j/kg)')
        plt.ylim(exp['P (Pa)'].min() - 500e3)
        plt.title('Prototype Refrigeration Sys. P-h ')
        
    if len(args) == 1:
        exp = args[0] 
        fluid = 'R410a'
        # Create Vapor Dome
        vaporDomeS = np.concatenate([CP.PropsSI('S', 'T', np.linspace(200, 344, 150), 'Q', 0, fluid),
                                     CP.PropsSI('S', 'T', np.linspace(344, 200, 150), 'Q', 1, fluid)])
        vaporDomeT = np.concatenate(
            [np.linspace(200, 344, 150), np.linspace(344, 200, 150)])

        vaporDomeH = np.concatenate([CP.PropsSI('H', 'P', np.linspace(2e5, 5000e3, 150), 'Q', 0, fluid),
                                     CP.PropsSI('H', 'P', np.linspace(5000e3, 2e5, 150), 'Q', 1, fluid)])
        vaporDomeP = np.concatenate(
            [np.linspace(2e5, 5000e3, 150), np.linspace(5000e3, 2e5, 150)])

        
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        plt.subplot(121)
        plt.plot(vaporDomeS, vaporDomeT, '-')
        plt.plot([500, 2000],[exp['Ambient T (K)'], exp['Ambient T (K)']], 'c--')
        plt.plot([500, 2000],[exp['Pod T (K)'], exp['Pod T (K)']], 'r--')
        
        for ind, s in enumerate(exp['s (j/kg K)']):
            plt.annotate(txt[ind], (exp['s (j/kg K)'][ind] + 10, exp['T (K)'][ind] + 1))
            
        plt.plot(exp['s (j/kg K)'], exp['T (K)'], 'o')
        plt.ylabel('Temperature (K)')
        plt.xlabel('Entropy (j/kg/K)')
        plt.legend(lgnd)


        plt.title('Prototype Refrigeration Sys. T-s ')
        plt.ylim(exp['T (K)'].min() - 10)
        plt.subplot(122)
        plt.plot(vaporDomeH, vaporDomeP, '-')
        plt.plot(exp['h (j/kg)'], exp['P (Pa)'], 'o')
        
        for ind, h in enumerate(exp['h (j/kg)']):
            plt.annotate(txt[ind], (exp['h (j/kg)'][ind] + 1e3, exp['P (Pa)'][ind] + 0.1e5))
            
        plt.ylabel('Pressure (Pa)')
        plt.xlabel('Enthalpy (j/kg)')
        plt.ylim(exp['P (Pa)'].min() - 500e3)
        plt.title('Prototype Refrigeration Sys. P-h ')
    
    if save:
        plt.savefig(str(exp.name) + ".png")
    else:
        plt.show()
        
    return

def example_plots():
    # A function which plots example T-s and P-h diagrams of a VCRC with and without losses.
    
    fluid = 'R410a' 
    
    P_e = 1e6
    P_c = 3.5e6

    # Create Vapor Dome
    vaporDomeS = np.concatenate([CP.PropsSI('S', 'T', np.linspace(200, 344, 150), 'Q', 0, fluid),
                                 CP.PropsSI('S', 'T', np.linspace(344, 200, 150), 'Q', 1, fluid)])
    vaporDomeT = np.concatenate(
        [np.linspace(200, 344, 150), np.linspace(344, 200, 150)])

    vaporDomeH = np.concatenate([CP.PropsSI('H', 'P', np.linspace(2e5, 5000e3, 150), 'Q', 0, fluid),
                                 CP.PropsSI('H', 'P', np.linspace(5000e3, 2e5, 150), 'Q', 1, fluid)])
    vaporDomeP = np.concatenate(
        [np.linspace(2e5, 5000e3, 150), np.linspace(5000e3, 2e5, 150)])

    # Starting evaporation and condensing pressures
    idealPressure = np.array(P_e)
    idealPressure = np.append(idealPressure, P_c * np.ones(4))
    idealPressure = np.append(idealPressure, P_e * np.ones(3))


    # Starting temperature with 15 degree super heat
    idealTemperature = np.array(CP.PropsSI('T', 'P', idealPressure[0], 'Q', 1, fluid) + 15)

    # Look up entropy
    idealEntropy = np.array(CP.PropsSI('S', 'P', idealPressure[0], 'T', idealTemperature.flat[0], fluid))

    # Look up enthalpy
    idealEnthalpy = np.array(CP.PropsSI('H', 'P', idealPressure[0], 'T', idealTemperature.flat[0], fluid))

    # Isentropic Compression
    idealEntropy = np.append(idealEntropy, idealEntropy.flat[0])
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[1], 'S', idealEntropy[1], fluid))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[1], 'S', idealEntropy[1], fluid))

    # Isobaric super heat
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[2], 'Q', 1, fluid))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[2], 'Q', 1, fluid))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[2], 'Q', 1, fluid))

    # Isobaric phase change
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[3], 'Q', 0, fluid))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[3], 'Q', 0, fluid))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[3], 'Q', 0, fluid))

    # 10 degree subcool
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[4], 'Q', 0, fluid) -  10)
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[4], 'T', idealTemperature[4], fluid))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[4], 'T', idealTemperature[4], fluid))

    # Isenthalpic expansion
    idealEnthalpy = np.append(idealEnthalpy, idealEnthalpy[4])
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[5], 'H', idealEnthalpy[5], fluid))
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[5], 'H', idealEnthalpy[5], fluid))


    # Isobaric phase change
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[6], 'Q', 1, fluid))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[6], 'Q', 1, fluid))
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[6], 'Q', 1, fluid))

    # Isobaric superheat 
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[7], 'T', idealTemperature[0], fluid))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[7], 'T', idealTemperature[0], fluid))
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[7], 'T', idealTemperature[0], fluid))


    ########################### REAL Cycle ###########################

    # Starting evaporation and condensing pressures
    P_drop = 1e5
    
    actualPressure = np.array(P_e - P_drop)
    actualPressure = np.append(actualPressure, P_c)


    # Starting temperature with 15 degree super heat
    actualTemperature = np.array(CP.PropsSI('T', 'P', actualPressure[0], 'Q', 1, fluid) + 15)

    # Look up entropy
    actualEntropy = np.array(CP.PropsSI('S', 'P', actualPressure[0], 'T', actualTemperature.flat[0], fluid))

    # Look up enthalpy
    actualEnthalpy = np.array(CP.PropsSI('H', 'P', actualPressure[0], 'T', actualTemperature.flat[0], fluid))

    # Compression with 50% superheat entropy loss
    actualEntropy = np.append(actualEntropy, actualEntropy.flat[0] - 0.5 * (actualEntropy.flat[0] - 
                                                                            CP.PropsSI('S', 'P', actualPressure[1], 'Q', 1, fluid)))
    actualTemperature = np.append(actualTemperature, CP.PropsSI('T', 'P', actualPressure[1], 'S', actualEntropy[1], fluid))
    actualEnthalpy = np.append(actualEnthalpy, CP.PropsSI('H', 'P', actualPressure[1], 'S', actualEntropy[1], fluid))

    # super heat
    actualPressure = np.append(actualPressure, actualPressure[1] - 0.25 * P_drop)
    actualEntropy = np.append(actualEntropy, CP.PropsSI('S', 'P', actualPressure[2], 'Q', 1, fluid))
    actualTemperature = np.append(actualTemperature, CP.PropsSI('T', 'P', actualPressure[2], 'Q', 1, fluid))
    actualEnthalpy = np.append(actualEnthalpy, CP.PropsSI('H', 'P', actualPressure[2], 'Q', 1, fluid))

    # phase change
    actualPressure = np.append(actualPressure, actualPressure[1] - 0.5 * P_drop)
    actualEntropy = np.append(actualEntropy, CP.PropsSI('S', 'P', actualPressure[3], 'Q', 0, fluid))
    actualTemperature = np.append(actualTemperature, CP.PropsSI('T', 'P', actualPressure[3], 'Q', 0, fluid))
    actualEnthalpy = np.append(actualEnthalpy, CP.PropsSI('H', 'P', actualPressure[3], 'Q', 0, fluid))

    # 10 degree subcool
    actualPressure = np.append(actualPressure, actualPressure[1] - 1e5)
    actualTemperature = np.append(actualTemperature, CP.PropsSI('T', 'P', actualPressure[4], 'Q', 0, fluid) -  10)
    actualEntropy = np.append(actualEntropy, CP.PropsSI('S', 'P', actualPressure[4], 'T', actualTemperature[4], fluid))
    actualEnthalpy = np.append(actualEnthalpy, CP.PropsSI('H', 'P', actualPressure[4], 'T', actualTemperature[4], fluid))

    # Isenenthalpic expansion
    actualPressure = np.append(actualPressure, actualPressure[0] + P_drop)
    actualEnthalpy = np.append(actualEnthalpy, 1 * actualEnthalpy[4])
    actualTemperature = np.append(actualTemperature, CP.PropsSI('T', 'P', actualPressure[5], 'H', actualEnthalpy[5], fluid))
    actualEntropy = np.append(actualEntropy, CP.PropsSI('S', 'P', actualPressure[5], 'H', actualEnthalpy[5], fluid))


    # phase change
    actualPressure = np.append(actualPressure, actualPressure[0] + 0.25 * P_drop)
    actualEnthalpy = np.append(actualEnthalpy, CP.PropsSI('H', 'P', actualPressure[6], 'Q', 1, fluid))
    actualTemperature = np.append(actualTemperature, CP.PropsSI('T', 'P', actualPressure[6], 'Q', 1, fluid))
    actualEntropy = np.append(actualEntropy, CP.PropsSI('S', 'P', actualPressure[6], 'Q', 1, fluid))

    # superheat 
    actualPressure = np.append(actualPressure, actualPressure[0])
    actualEnthalpy = np.append(actualEnthalpy, CP.PropsSI('H', 'P', actualPressure[7], 'T', actualTemperature[0], fluid))
    actualTemperature = np.append(actualTemperature, CP.PropsSI('T', 'P', actualPressure[7], 'T', actualTemperature[0], fluid))
    actualEntropy = np.append(actualEntropy, CP.PropsSI('S', 'P', actualPressure[7], 'T', actualTemperature[0], fluid))

    insideTemp = 300
    outsideTemp = 310
    

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    plt.subplot(121)
    plt.plot(vaporDomeS, vaporDomeT, '-')
    txt = list(range(1, 7)) + list(range(8,9))

    for ind, s in enumerate(idealEntropy):
        if ind == len(idealEnthalpy) - 1:
            break
        plt.annotate(str(txt[ind]), (idealEntropy[ind] + 10, idealTemperature[ind] + 1))
    plt.plot(idealEntropy, idealTemperature, 'o-')

    for ind, s in enumerate(actualEntropy):
        if ind == len(actualEnthalpy) - 1:
            break
        plt.annotate(str(txt[ind]) + "'", (actualEntropy[ind] - 5 * 10, actualTemperature[ind] + 2))
    plt.plot(actualEntropy, actualTemperature, 'og--')
    plt.plot([500, 2000],[outsideTemp, outsideTemp], 'r--')
    plt.plot([500, 2000],[insideTemp, insideTemp], 'c--')
    plt.ylabel('Temperature (K)')
    plt.xlabel('Entropy (j/kg/K)')
    plt.legend(['Vapor Dome', 'Ideal Cycle', 'Cycle with Losses', 'Outside Temperature', 'Inside Temperature'])
    plt.title('Prototype Refrigeration Sys. T-s ')
    plt.ylim(actualTemperature.min() - 10)
    
    plt.subplot(122)
    plt.plot(vaporDomeH, vaporDomeP, '-')

    plt.plot(idealEnthalpy, idealPressure, '-o')
    for ind, h in enumerate(idealEnthalpy):
        if ind == len(idealEnthalpy) - 1:
            break
        plt.annotate(str(txt[ind]), (idealEnthalpy[ind] + 3 * 1e3, idealPressure[ind] + 3 * 0.1e5))

    plt.plot(actualEnthalpy, actualPressure, '--go')
    for ind, h in enumerate(actualEnthalpy):
        if ind == len(actualEnthalpy) - 1:
            break
        plt.annotate(str(txt[ind]) + "'", (actualEnthalpy[ind] - 1e4, actualPressure[ind] - 2 * 1e5))

    plt.ylabel('Pressure (Pa)')
    plt.xlabel('Enthalpy (j/kg)')
    plt.ylim(actualPressure.min() - 500e3)
    plt.title('Prototype Refrigeration Sys. P-h ')

    plt.show() 

    return

def ideal_cycle(P_c, P_e, T_SC, T_SH):
    # A function which plots example T-s and P-h diagrams of a VCRC with and without losses.
    
    fluid = 'R410a' 

    # Starting evaporation and condensing pressures
    idealPressure = np.array(P_e)
    idealPressure = np.append(idealPressure, P_c * np.ones(4))
    idealPressure = np.append(idealPressure, P_e * np.ones(3))


    # Starting temperature with 15 degree super heat
    idealTemperature = np.array(T_SH)

    # Look up entropy
    idealEntropy = np.array(CP.PropsSI('S', 'P', idealPressure[0], 'T', idealTemperature.flat[0], fluid))

    # Look up enthalpy
    idealEnthalpy = np.array(CP.PropsSI('H', 'P', idealPressure[0], 'T', idealTemperature.flat[0], fluid))

    # Isentropic Compression
    idealEntropy = np.append(idealEntropy, idealEntropy.flat[0])
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[1], 'S', idealEntropy[1], fluid))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[1], 'S', idealEntropy[1], fluid))

    # Isobaric super heat
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[2], 'Q', 1, fluid))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[2], 'Q', 1, fluid))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[2], 'Q', 1, fluid))

    # Isobaric phase change
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[3], 'Q', 0, fluid))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[3], 'Q', 0, fluid))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[3], 'Q', 0, fluid))

    # 10 degree subcool
    idealTemperature = np.append(idealTemperature, T_SC)
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[4], 'T', idealTemperature[4], fluid))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[4], 'T', idealTemperature[4], fluid))

    # Isenthalpic expansion
    idealEnthalpy = np.append(idealEnthalpy, idealEnthalpy[4])
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[5], 'H', idealEnthalpy[5], fluid))
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[5], 'H', idealEnthalpy[5], fluid))


    # Isobaric phase change
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[6], 'Q', 1, fluid))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[6], 'Q', 1, fluid))
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[6], 'Q', 1, fluid))

    # Isobaric superheat 
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[7], 'T', idealTemperature[0], fluid))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[7], 'T', idealTemperature[0], fluid))
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[7], 'T', idealTemperature[0], fluid))

    # Creae pandas dataframe for 
    idealData = pd.DataFrame({ 'P (Pa)': [idealPressure], 'T (K)': [idealTemperature], 'h (j/kg)':[idealEnthalpy], 
                                     's (j/kg K)': [idealEntropy]})
    
    return idealData