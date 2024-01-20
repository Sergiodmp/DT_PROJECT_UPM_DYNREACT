import random
import time
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from decision_transformer.d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS
import pandas as pd

def discount_cumsum(x, gamma):

    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum


def get_d4rl_normalized_score(score, env_name):
    env_key = env_name.split('-')[0].lower()
    assert env_key in REF_MAX_SCORE, f'no reference score for {env_key} env to calculate d4rl score'
    return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])


def get_d4rl_dataset_stats(env_d4rl_name):
    return D4RL_DATASET_STATS[env_d4rl_name]


def evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
                    num_eval_ep=1, max_test_ep_len=60,
                    state_mean=None, state_std=None, render=False):
    
    eval_batch_size = 1  # Tamaño del lote para evaluación
    results = {}
    total_reward = 0
    total_timesteps = 0
    state_dim = env.observation_space.shape[0]
    #print("state_dim", state_dim)
    #act_dim = env.action_space.n
    act_dim=62
    #state_mean = torch.zeros((state_dim,)).to(device) if state_mean is None else torch.from_numpy(state_mean).to(device)
    #state_std = torch.ones((state_dim,)).to(device) if state_std is None else torch.from_numpy(state_std).to(device)
    state_mean = torch.zeros((state_dim,)).to(device) 
    state_std = torch.ones((state_dim,)).to(device) 
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(num_eval_ep):
            acciones_tomadas = set()
            #actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim), dtype=torch.float64, device=device)
            actions = torch.zeros((eval_batch_size, max_test_ep_len), dtype=torch.long, device=device)
            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim), dtype=torch.float64, device=device)
            rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1), dtype=torch.float64, device=device)
            running_state = env.reset()
            running_state = running_state.astype(np.float64)
            running_reward = 0
            running_rtg = rtg_target / rtg_scale
            winner_df= pd.DataFrame()
            all_winner_df =   pd.DataFrame()


            i=0
            for t in range(max_test_ep_len):
                if isinstance(running_state, np.ndarray):
                    # Convertir de np.ndarray a Tensor y luego a device
                    running_state = torch.from_numpy(running_state.astype(np.float64)).to(device)
                elif isinstance(running_state, torch.Tensor):
                    running_state = running_state.to(device)

                # Asegurarse de que el tamaño del tensor sea el adecuado
                if running_state.size(0) < 62:
                    padding = torch.zeros(62 - running_state.size(0), device=device)
                    running_state = torch.cat((running_state, padding))

                states[0, t] = running_state

                #running_state = running_state.astype(np.float64)
                
                #states[0, t] = torch.from_numpy(np.array([running_state])).to(device)
                #print("Shape of states[0, t]:", states[0, t].shape)
                #print("Shape of state_mean:", state_mean.shape)
                #print("Shape of state_std:", state_std.shape)
                states[0, t] = (states[0, t] - state_mean) / state_std
                i+=1

                if t < context_len:
                    _, act_preds, _ = model.forward(timesteps[:,:context_len], states[:,:context_len], actions[:,:context_len], rewards_to_go[:,:context_len], bandera=1)
                    act = act_preds[0, t].detach()
                else:
                    _, act_preds, _ = model.forward(timesteps[:,t-context_len+1:t+1], states[:,t-context_len+1:t+1], actions[:,t-context_len+1:t+1], rewards_to_go[:,t-context_len+1:t+1], bandera=1)
                    act = act_preds[0, -1].detach()
                                                                                                    # Aquí continua el código para tomar la acción y actualizar el entorno
                                                                                                    # Primero, identificamos la acción con la puntuación más alta en la predicción
                                                                                                    # Utilizamos torch.argmax() para obtener el índice de la puntuación más alta en cada fila.
                
                predicted_action = torch.argmax(act_preds[0, -1]).item()

                # Lista de todas las acciones en orden descendente según su puntuación
                sorted_actions = torch.argsort(act_preds[0, -1], descending=True).tolist()

                # Selecciona la siguiente mejor acción que no ha sido tomada
                for action in sorted_actions:
                    if action not in acciones_tomadas:
                        predicted_action = action
                        break

                # Agrega la acción seleccionada al conjunto de acciones tomadas
                acciones_tomadas.add(predicted_action)

                # Reinicia el conjunto si todas las acciones han sido seleccionadas
                if len(acciones_tomadas) >= act_dim:
                    acciones_tomadas.clear()

                # Ejecuta la acción en el entorno si está en el conjunto de bobinas disponibles
                if predicted_action in env.snapshot.loc[:,'Coil number example'].values:
                    running_state, running_reward, done, _, total_cost = env.step(predicted_action)
                    all_winner_df = env.all_winner_df

                    print(all_winner_df)
                    actions[0, t] = predicted_action                                                 # add action in placeholder
                    #actions[0, t] = act

                    total_reward += running_reward

                    if render:
                        env.render()
                    if done:
                        break               
    print(all_winner_df)
    output_file_path = 'C:/Users/Marta/OneDrive/Escritorio/sergio/poli2/outputRL_allvas_1401.xlsx'  # Cambia esto por la ruta donde deseas guardar el archivo
    all_winner_df.to_excel(output_file_path, index=False)
    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep
    return results


class D4RLTrajectoryDataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale):

        self.context_len = context_len
        
        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        trajectories_prueba=self.trajectories
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization
        states = np.concatenate(states, axis=0)
        states = np.array(states, dtype=np.float64)

        # Ahora, verifica si hay NaN o Infinitos
        if np.isnan(states).any() or np.isinf(states).any():
            states = np.nan_to_num(states)  # Reemplaza NaN e infinitos

        # Continúa con el cálculo de la media y la desviación estándar
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std
            
        

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj['observations'] = np.array(traj['observations'], dtype=np.float32)
        traj['actions'] = np.array(traj['actions'], dtype=np.float32)
        traj_len = traj['observations'].shape[0]
        #traj_len = len(traj['observations'])

        if traj_len >= self.context_len:
            # sample random index to slice trajectory

            si = random.randint(0, traj_len - self.context_len)
            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)
            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)                          #la mascara se utiliza para diferenciar los elementos reales de los de relleno (padding)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states,
                                torch.ones(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)


            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                dtype=returns_to_go.dtype)],
                               dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        return  timesteps, states, actions, returns_to_go, traj_mask
