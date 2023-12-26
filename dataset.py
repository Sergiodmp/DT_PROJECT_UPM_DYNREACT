import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from environment import env_va
import random
env = env_va()
observations = []
next_observations = []
actions = []
rewards = []
terminals = []

def expert_policy(obs):
    # Si no hay datos de ganadores, se elige una acción aleatoria.
    if env.winner_df.empty:
        action = env.action_space.sample()
    else:
        # Selección basada en criterios más allá del costo mínimo.
        # Por ejemplo, se podría elegir una bobina con un costo
        # un poco más alto que el mínimo para introducir variabilidad.
        sorted_coils = env.results_2.sort_values(by='cost')
        if random.random() < 0.8:  # 80% de las veces, elige entre las 3 más baratas
            choice_range = min(3, len(sorted_coils))
            action = sorted_coils.iloc[random.randint(0, choice_range - 1)]['Coil']
        else:  # 20% de las veces, elige una bobina aleatoria para explorar
            action = sorted_coils.iloc[random.randint(0, len(sorted_coils) - 1)]['Coil']
    return action
import random

def expert_policy_seq(obs):
    percent_ordered=0.8
    if env.winner_df.empty:
        # Si no hay datos de ganadores, utiliza la secuencia de acciones existente
        actions = env.coils
    else:
        # Ordena las bobinas por costo
        sorted_coils = env.results_2.sort_values(by='cost')['Coil'].tolist()

        # Decide cuántas acciones estarán ordenadas
        num_ordered = int(len(sorted_coils) * percent_ordered)

        # Añade las acciones ordenadas
        ordered_actions = sorted_coils[:num_ordered]

        # Añade las acciones restantes de manera aleatoria
        random_actions = sorted_coils[num_ordered:]
        random.shuffle(random_actions)

        # Combina las acciones ordenadas y aleatorias
        actions = ordered_actions + random_actions

    return actions

# Uso de la función




num_episodes = 200
max_timesteps = 200



trajectories = []

for episode in range(num_episodes):
    obs = env.reset()  # Reset the environment or obtain initial observation

    trajectory = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': []
        #'total_cost':[]
    }

    for timestep in range(max_timesteps):
        actions = expert_policy_seq(obs)
        action = actions[0]  # Siempre toma la primera acción de la lista
        next_obs, reward, done, terminado, total_cost = env.step(action)
        if all(val is not None for val in [next_obs, reward, done, terminado, total_cost]):
        # Collect observation, action, reward, next observation, and done flag
            trajectory['observations'].append(next_obs)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            #trajectory['total_cost'].append(total_cost)
            trajectory['dones'].append(done)
        if terminado:
            break

    trajectories.append(trajectory)
    
# Convert lists in each trajectory to numpy arrays
for trajectory in trajectories:

    trajectory['observations'] = np.array(trajectory['observations'])
    print(trajectory['observations'])
    trajectory['actions'] = np.array(trajectory['actions'])
    print(trajectory['actions'])
    trajectory['rewards'] = np.array(trajectory['rewards'])
    trajectory['dones'] = np.array(trajectory['dones'])

# Save the trajectories as a pickle file
with open('dataset_2412_17.pkl', 'wb') as f:
    pickle.dump(trajectories, f)