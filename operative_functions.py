import os, datetime, subprocess, pdb
import math, json, time, statistics, globals
import numpy as np
import pandas as pd
from datetime import timedelta,date
from random import random
from random import randrange
#from spade.message import Message

"""General Functions"""




def production_cost(coil_df, i):
    z = abs(coil_df.loc[i,'HRC Width (mm)'] - coil_df.loc[i,'FinalWidth'])
    n = abs(coil_df.loc[i,'HRC Thickness (mm)'] - coil_df.loc[i,'Final Thickness (mm)'])
    cost = float(z * 4 + n * 2)
    return cost

def transport_cost(to):
    costes_df = pd.DataFrame()
    costes_df['From'] = ['NWW1', 'NWW1', 'NWW1','NWW1','NWW1','NWW3','NWW3','NWW3','NWW3','NWW3','NWW4','NWW4','NWW4','NWW4','NWW4']
    costes_df['CrossTransport'] = [24.6, 24.6, 0, 0, 55.6, 74.8, 74.8, 50.2, 50.2, 32.3, 71.5, 71.5, 46.9,46.9, 0]
    costes_df['Supply'] = [24.6, 24.6, 21.1, 21.1, 5.7, 24.6, 24.6, 21.1, 21.1, 5.7, 24.6, 24.6, 21.1, 21.1, 5.7]
    costes_df['To'] = ['va08', 'va09', 'va10','va11','va12','va08','va09','va10','va11','va12','va08','va09','va10','va11','va12']
    costes_df = costes_df.loc[costes_df['To'] == to]
    costes_df = costes_df.reset_index(drop=True)
    return costes_df


def energy_cost(df_parameters_energy, price_energy_consumption,coil_msgs_df,i):
    speed=500  #cambiar 500 mm/min
    production_time=20 #cambiar
    to= coil_msgs_df.loc[0,'PLANT']
    if to == 'va09' or to == 'va10' or to == 'va11' or to == 'va12':
        a = df_parameters_energy.loc[to,'a']
        b = df_parameters_energy.loc[to,'b']
        c = df_parameters_energy.loc[to,'c']
        d = df_parameters_energy.loc[to,'d']
        e = df_parameters_energy.loc[to,'e']
        f = df_parameters_energy.loc[to,'f']
        melting_code = coil_msgs_df.loc[i,'Melting Code']
        tinlayerup=coil_msgs_df.loc[i,'Tin Layer Up']
        tinlayerdown=coil_msgs_df.loc[i,'Tin Layer Down']

        
        power = a + coil_msgs_df.loc[i,'FinalWidth'] * b + coil_msgs_df.loc[i,'Final Thickness (mm)'] * c + tinlayerup * d + tinlayerdown * e + speed*f
        #   Power [kw] =     a + Width-Out (VA) * b + Final Thickness * c + Tin Layer Up * d + Tin Layer Down * e + Speed (VA) * f [kw]
    else:
        power=0
    energy_demand = power * production_time / 60                                                       # KWh
    energy_cost= energy_demand * price_energy_consumption
    #print("coste energetico", energy_cost)
    return energy_cost
#this function obtains a value that is goint to multiply the "expected profit" in each coil, to reduce the profit in the case
#that the coil does not meet the rules of the plant.

def va_rules(coil_msgs_df,i, winner_df):
    to= coil_msgs_df.loc[0,'PLANT']
    accept=1
    #coil_msgs_df.loc[i,'plant_rule']= accept
    if to == 'va08':
        if len(winner_df)>0:
            if abs((winner_df.iloc[-1]['HRC Thickness (mm)']-coil_msgs_df.loc[i,'HRC Thickness (mm)'])) > 0.1:
                accept = accept*0.2
    elif to == 'va09':


        if abs((coil_msgs_df.loc[i,'HRC Width (mm)'] - coil_msgs_df.loc[i,'FinalWidth'])) > 120:                                               
            accept = accept*0.2
        #if coil_msgs_df.loc[i,'oel_sorte'] == 8:                                                                                    
            #accept = accept*0.1
        if len(winner_df)>0:
            #print(winner_df.loc[0,'HRC Thickness (mm)'])
            #print(coil_msgs_df.loc[i,'HRC Thickness (mm)'])
            if abs((winner_df.loc[0,'HRC Thickness (mm)']-coil_msgs_df.loc[i,'HRC Thickness (mm)'])) > 0.1:
                accept = accept*0.3
            '''if coil_msgs_df.loc[i,'single_reduction'] == 0:                                                                         
                
                if abs((winner_df.iloc[-1]['initial_thickness']-coil_msgs_df.loc[i,'espesor'])) > winner_df.iloc[-1]['initial_thickness']*0.04:       
                    accept = accept*0.4 
                    '''
        else:
            accept=1
    elif to == 'va10':
        if len(winner_df)>0:
            '''
            if winner_df.iloc[-1]['oel_sorte'] < coil_msgs_df.loc[i,'oel_sorte']:
                accept = accept*0.2
            
            else:
                if abs((winner_df.iloc[-1]['HRC Thickness (mm)']-coil_msgs_df.loc[i,'HRC Thickness (mm)'])) > 0.05:
                    accept = accept*0.2
                    '''
    elif to == 'va11':
        if len(winner_df)>0:
            if abs((winner_df.iloc[-1]['HRC Thickness (mm)']-coil_msgs_df.loc[i,'HRC Thickness (mm)'])) > 0.1:
                    accept = accept*0.2
    elif to == 'va12':
        '''
        if coil_msgs_df.loc[i,'assivieru_ngkz'] != 555:
            accept = accept*0.2
        else:
            if len(winner_df)>0:
                if abs((winner_df.iloc[-1]['HRC Thickness (mm)']-coil_msgs_df.loc[i,'HRC Thickness (mm)'])) > 0.1:
                    accept = accept*0.2
    '''
    return accept


def va_bid_evaluation(df_parameters_energy,coil_msgs_df, price_energy_consumption, winner_df):
    key = []
    #From_location = coil_msgs_df.loc[0,'From']
    transport_cost_df = transport_cost(coil_msgs_df.loc[0,'PLANT'])
    for i in range(transport_cost_df.shape[0]):
        m = transport_cost_df.loc[i, 'CrossTransport']
        n = transport_cost_df.loc[i, 'Supply']
        key.append(n+m)
    transport_cost_df['transport_cost'] = key
    transport_cost_df = transport_cost_df.loc[:, ['From', 'To', 'transport_cost']]
    #sergio 2007
    coil_msgs_df

    for i in range(coil_msgs_df.shape[0]):
        coil_msgs_df.at[i, 'production_cost'] = production_cost(coil_msgs_df, i)
        
        coil_msgs_df.at[i, 'energy_cost'] = energy_cost(df_parameters_energy,price_energy_consumption, coil_msgs_df,i)
        coil_msgs_df.at[i, 'plant_rule'] = va_rules(coil_msgs_df, i, winner_df)
    #coil_msgs_df = coil_msgs_df.merge(transport_cost_df, on='From', sort=False)
    results = pd.DataFrame()
    for i in range(coil_msgs_df.shape[0]):
        m = coil_msgs_df.loc[i, 'production_cost']
        #n = coil_msgs_df.loc[i, 'transport_cost']
        energy = coil_msgs_df.loc[i, 'energy_cost']
        coil_msgs_df.loc[i, 'cost'] = m  + energy
        results = coil_msgs_df
        results = results.sort_values(by=['cost'], ascending = True)
        results = results.reset_index(drop=True)
        value = []

    for i in range(results.shape[0]):
        value.append(i+1)
    results.insert(loc=0, column='position', value=value)
    #results = results.sort_values(by=['profit'], ascending = False)   
    #print("el orden por coste es:", results)   
    return results

def va_result(coil_ofertas_df, jid_list):
    #print(jid_list)
    column_names = ['Coil', 'cost', 'HRC Thickness (mm)', 'Final Thickness (mm)', 'HRC Width (mm)', 'FinalWidth']
    # Create empty DataFrame with specified columns
    df = pd.DataFrame(columns=column_names)
    '''
    if step == 'counterbid' :
        df = pd.DataFrame([], columns=['Coil', 'Minimum_price', 'Bid',\
                                   'Difference', 'Budget_remaining',\
                                   'Counterbid','Profit'])
    #
                                   '''
    for i in range(len(jid_list)):
        df.at[i, 'Coil'] = coil_ofertas_df.loc[i, 'Coil number example']
        df.at[i, 'cost'] = coil_ofertas_df.loc[i, 'cost']
        df.at[i, 'HRC Thickness (mm)'] = coil_ofertas_df.loc[i, 'HRC Thickness (mm)']
        df.at[i, 'Final Thickness (mm)'] = coil_ofertas_df.loc[i, 'Final Thickness (mm)']
        df.at[i, 'HRC Width (mm)'] = coil_ofertas_df.loc[i, 'HRC Width (mm)']
        df.at[i, 'FinalWidth'] = coil_ofertas_df.loc[i, 'FinalWidth']
        if df.at[i, 'Coil'] == 1000:
            df.at[i, 'cost'] = 1000

    #print(df)
    return df
