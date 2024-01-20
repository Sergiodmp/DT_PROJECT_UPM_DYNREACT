import gymnasium as gym
import random
import torch
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import openpyxl
import operative_functions as asf
import globals
global results_2, winner_df, all_winner_df, snapshot,i
class env_va(gym.Env):
        def __init__(self):
                self.price_energy_consumption = 0.222 #euros/KWh
                done=False
                self.index_va = ['va09', 'va10', 'va11', 'va12', 'va13', 'va14']
                self.df_parameters_energy=pd.DataFrame({
                                'a': [-4335, -4335, -8081.22, -141,-6011.6, -3855.45],
                                'b': [2.1, 2.1, 4.31, 3.27, 3.83, 2.4],
                                'c': [5405.53, 5405.53, 6826.2, 5943.73, 6742.25, 901.87],
                                'd': [191.27, 191.27, 240.12, 228.9, 195.85, 292.9],
                                'e': [212.31, 212.31, 319.5, 348.29, 264.99, 238.68],
                                'f': [9.44, 9.44, 12.68, 12.16, 11.74, 8.66]}, 
                                index=self.index_va)
                self.snapshot = pd.read_excel(io='C:/Users/Marta/OneDrive/Escritorio/sergio/poli2/production_snapshot_pruebascorto.xlsx',
                                                        sheet_name='Hoja1', header=0, names = None, index_col = None,
                                                        usecols= 'A:T', engine= 'openpyxl')
                self.jobs=len(self.snapshot.axes[0]) #number of coils that have to be processed
                #self.action_space = gym.spaces.Discrete(self.jobs) 
                #self.action_space = gym.spaces.Discrete(self.jobs+2)
                self.action_space = gym.spaces.Discrete(64)
                #self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([100000000])) #the observation space is the actual cost of the whole production
                #self.observation_space =gym.spaces.Discrete(self.jobs)
                #self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32)
                self.observation_space = gym.spaces.Box(low=np.full((62,), -np.inf), 
                                        high=np.full((62,), np.inf), 
                                        dtype=np.float32)
                self.state= 0 #overall cost (initial state)
                self.manager_lenth=50

        def reward_scaler(self,reward):
                
                return reward+0.001

        def calculate_reward(self,action):
                self.max_cost=self.results_2.iloc[-1]['cost']
                reward= 1-self.winner_df['cost'].iloc[0]/self.max_cost
                scaled_reward=self.reward_scaler(reward)

                return scaled_reward
        def reset(self):
                self.state=0
                done=False
                self.manager_lenth=50
                self.snapshot = pd.read_excel(io='C:/Users/Marta/OneDrive/Escritorio/sergio/poli2/production_snapshot_pruebascorto.xlsx',
                                                        sheet_name='Hoja1', header=0, names = None, index_col = None,
                                                        usecols= 'A:T', engine= 'openpyxl')
                self.winner_df = pd.DataFrame()     
                self.all_winner_df =   pd.DataFrame()  
                self.assess_costs_coil = asf.va_bid_evaluation(self.df_parameters_energy,self.snapshot,self.price_energy_consumption,self.winner_df)
                self.jid_list_2=self.assess_costs_coil.loc[:,'Coil number example'].tolist()
                self.results_2 = asf.va_result(self.assess_costs_coil,self.jid_list_2)
                self.costcoils_df=self.results_2.loc[:,'cost']
                self.costcoils=self.results_2['cost'].values
                self.coils_df=self.results_2.loc[:,'Coil']
                self.coils=self.results_2['Coil'].values
                self.max_cost=self.results_2.iloc[-1]['cost']
                self.coil_with_min_cost=self.results_2.iloc[0]['Coil']
                self.coils2_df=self.snapshot.loc[:,'Coil number example'] 
                self.coils2=self.snapshot['Coil number example'].values
                self.acciones_tomadas = set()
                return self.costcoils

        def step(self,action):
            if action in self.snapshot['Coil number example'].values:
                    
                    if action in self.acciones_tomadas:
                           bandera=0
                    else:
                           bandera=1
                           self.acciones_tomadas.add(action)
            else:
                    bandera=0
            
            if bandera==1:
                    self.assess_costs_coil = asf.va_bid_evaluation(self.df_parameters_energy,self.snapshot,self.price_energy_consumption,self.winner_df) 
                    self.jid_list_2=self.assess_costs_coil.loc[:,'Coil number example'].tolist()
                    self.results_2 = asf.va_result(self.assess_costs_coil,self.jid_list_2)
                    self.winner_df = self.results_2[self.results_2['Coil']==action]                        #el ganador es obviamente el que se elige
                    self.winner_df = self.winner_df.reset_index(drop=True)                
                    self.costcoils_df=self.results_2.loc[:,'cost'] 
                    self.costcoils=self.results_2['cost'].values  
                    self.coils_df=self.results_2.loc[:,'Coil']
                    self.coils=self.results_2['Coil'].values
                    self.all_winner_df = pd.concat([self.all_winner_df,self.winner_df])              
                    self.all_winner_df =self.all_winner_df.reset_index(drop=True)
                    coil_eliminada=self.snapshot[(self.snapshot['Coil number example']==action)].copy()
                    self.snapshot.drop(self.snapshot[(self.snapshot['Coil number example']==action)].index, inplace =True)
                    new_dataframe = pd.DataFrame(1000, index=[0], columns=self.snapshot.columns)
                    #new_dataframe['Coil number example']=random.randint(80,127)
                    new_dataframe['Coil number example']=0
                    self.snapshot = pd.concat([self.snapshot, new_dataframe], ignore_index=True)
                    self.snapshot = self.snapshot.reset_index(drop=True)
                    self.coil_with_min_cost=self.results_2.iloc[0]['Coil']                         
                    reward=self.calculate_reward(action)
                    self.coils2_df=self.snapshot.loc[:,'Coil number example']

                    self.manager_lenth-=1
            else:
                    reward=0    
                    self.assess_costs_coil = asf.va_bid_evaluation(self.df_parameters_energy,self.snapshot,self.price_energy_consumption,self.winner_df) 
                    self.jid_list_2=self.assess_costs_coil.loc[:,'Coil number example'].tolist()
                    self.results_2 = asf.va_result(self.assess_costs_coil,self.jid_list_2)
                    self.coil_with_min_cost=self.results_2.iloc[0]['Coil'] 
                    self.costcoils_df=self.results_2.loc[:,'cost']     
                    self.coils_df=self.results_2.loc[:,'Coil']    
            max_value = np.max(self.costcoils)
            self.state=self.costcoils[0]
            listofstates=self.costcoils

            #check if the production planner is done:
            if self.manager_lenth <= 0:
                terminado = True
                is_sorted = self.all_winner_df['cost'].is_monotonic_increasing
                done = is_sorted
            else:
                terminado = False
                done = False  # Set a default value for done when manager_lenth > 0


            info={}

            return listofstates, reward, done, terminado, listofstates
