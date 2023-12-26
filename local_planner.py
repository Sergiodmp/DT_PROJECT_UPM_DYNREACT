import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from environment import env_va
env = env_va()

def expert_policy(obs):
    if env.winner_df.empty:
        action = env.action_space.sample()
        
    else:
        action = env.coil_with_min_cost
    return action
# Inicializar DataFrame vacío
data = {
    'next_obs': [],
    'action': []
}
record_df = pd.DataFrame(data)
obs = env.reset()  
for i in range(120):
    # Reset the environment or obtain initial observation
    
        action = expert_policy(obs)
        next_obs, reward, done, terminado, total_cost = env.step(action)
        obs=next_obs
        print(i)
#print(record_df)
print(env.all_winner_df)
output_file_path = 'C:/Users/Marta/OneDrive/Escritorio/sergio/poli2/output.xlsx'  # Cambia esto por la ruta donde deseas guardar el archivo
env.all_winner_df.to_excel(output_file_path, index=False)

