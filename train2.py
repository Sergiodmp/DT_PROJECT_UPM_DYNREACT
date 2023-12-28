import argparse
import os
import sys
import random
import csv
from datetime import datetime

import numpy as np
import gym

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from decision_transformer.utils2 import D4RLTrajectoryDataset, evaluate_on_env, get_d4rl_normalized_score
from decision_transformer.model import DecisionTransformer
from tqdm import tqdm


#Se debe cambiar lo de las acciones para que sea discreto siempre.
def get_model(args):
    state_dim = args.state_dim
    act_dim = args.act_dim
    n_blocks = args.n_blocks
    embed_dim = args.embed_dim
    context_len = args.context_len
    n_heads = args.n_heads
    dropout_p = args.dropout_p

    model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                n_blocks=n_blocks,
                h_dim=embed_dim,
                context_len=context_len,
                n_heads=n_heads,
                drop_p=dropout_p,
            )

    return model

def train(args):

    dataset = args.dataset          # medium / medium-replay / medium-expert
    rtg_scale = args.rtg_scale      # normalize returns to go

    # use v3 env for evaluation because
    # Decision Transformer paper evaluates results on v3 envs

    if args.env == 'env_va':
        env_name = 'env_va'
        #from prueba_env import env_va
        from environment import env_va
        env = env_va()      #instanciamos la clase
        rtg_target = 600
        env_d4rl_name = f'env_va-{dataset}-v2'

    else:
        raise NotImplementedError

    max_eval_ep_len = args.max_eval_ep_len  # max len of one episode    default 1000
    num_eval_ep = args.num_eval_ep          # num of evaluation episodes default 10

    batch_size = args.batch_size            # training batch size   default 10
    lr = args.lr                            # learning rate    default 1e-4
    wt_decay = args.wt_decay                # weight decay default 1e-4
    warmup_steps = args.warmup_steps        # warmup steps for lr scheduler default 10000

    # total updates = max_train_iters x num_updates_per_iter                        --------------> default 200x100= 20000
    max_train_iters = args.max_train_iters   #default 200
    num_updates_per_iter = args.num_updates_per_iter  #default 100

    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability

    # load data from this file
    dataset_path = 'dataset_2412_17.pkl'
    train_ds_size = int(0.9 * len(dataset_path))
    val_ds_size = len(dataset_path) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(dataset_path, [train_ds_size, val_ds_size])
    # saves model and csv in this directory
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # training and evaluation device
    device = torch.device(args.device)
    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")
    prefix = "dt_" + env_d4rl_name
    save_model_name =  prefix + "_model_" + start_time_str + ".pt"
    save_model_path = os.path.join(log_dir, save_model_name)
    save_best_model_path = save_model_path[:-3] + "_best.pt"
    log_csv_name = prefix + "_log_" + start_time_str + ".csv"
    log_csv_path = os.path.join(log_dir, log_csv_name)
    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = (["duration", "num_updates", "action_loss",
                   "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])
    csv_writer.writerow(csv_header)
    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)
    print("log csv save path: " + log_csv_path)

    traj_dataset = D4RLTrajectoryDataset(dataset_path, context_len, rtg_scale) #normaliza el dataset aunque ya lo normalizo yo antes... 
    #almacenamos estados y calculamos la media y la desviación estandar
    '''Prepara los datos para entrenar, formando minilotes...'''
    traj_data_loader = DataLoader(
                            traj_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True
                        )

    '''Cada iteración devuelve un lote de train_features y train_labels'''
    data_iter = iter(traj_data_loader)

    ## get state stats from dataset
    state_mean, state_std = traj_dataset.get_state_stats()
    '''state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]'''
    state_dim = env.observation_space.shape[0]   #it takes the shape that we have established in the environment
    #state_dim =env.observation_space.n          #si fijamos los estados como discretos
    act_dim = env.action_space.n
    
    #################################################################################################################SUPERVISED 28/06
    
    
    model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                n_blocks=n_blocks,
                h_dim=embed_dim,
                context_len=context_len,
                n_heads=n_heads,
                drop_p=dropout_p,
            ).to(device)                                                                                             #bandera=0 en model.forward

    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=lr,
                        weight_decay=wt_decay
                    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
                            optimizer,
                            lambda steps: min((steps+1)/warmup_steps, 1)
                        )

    max_d4rl_score = -1.0
    total_updates = 0

    for i_train_iter in range(max_train_iters):
        log_action_losses = []
        model.train()
###########################################################################################no entra aquiiiiiiiiii
        for timesteps, states, actions, returns_to_go, traj_mask in traj_data_loader:
            timesteps = timesteps.to(device)
            states = states.to(device)
            actions = actions.to(device)
            returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1)
            traj_mask = traj_mask.to(device)
            action_target = torch.clone(actions).detach().to(device)
            state_preds, action_preds, return_preds = model.forward(
                                                            timesteps=timesteps,
                                                            states=states,
                                                            actions=actions,
                                                            returns_to_go=returns_to_go,
                                                            bandera=0
            )
            
            action_preds_flat = action_preds.view(-1, act_dim)
            action_preds_filtered = action_preds_flat[traj_mask.view(-1,) > 0]
            action_target = action_target.long()
            action_target_flat = action_target.view(-1)
            action_target_filtered = action_target_flat[traj_mask.view(-1,) > 0]                                 #filtrado de elementos no acolchados (non-pad). Se utiliza para hacer que todas las secuencias tengan la misma longitud

            # Verificar las formas después de filtrar
            print("Forma de action_preds_filtered:", action_preds_filtered.shape)
            print("Forma de action_target_filtered:", action_target_filtered.shape)

            
            action_loss = F.cross_entropy(action_preds_filtered, action_target_filtered)                          #calculo de perdida
            action_loss = action_loss.double()
            '''action_loss = F.mse_loss(action_preds_flat, action_target_flat, reduction='mean')
            action_loss = action_loss.double()
            '''

            #Actualización de pesos en backpropagation para aprender
            #En estos pasos, se prepara el modelo para la retropropagación (backpropagation) reiniciando los gradientes, luego se retropropagan los errores (derivados de la pérdida calculada) a través del modelo para actualizar sus pesos. 
            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

            #Update the weights
            optimizer.step()
            scheduler.step()
            log_action_losses.append(action_loss.detach().cpu().item())
        # evaluate action accuracy
        results = evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
                                num_eval_ep, max_eval_ep_len, state_mean, state_std)                                  #bandera=1 en model.forward
        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        eval_d4rl_score = get_d4rl_normalized_score(results['eval/avg_reward'], env_name) * 100                      #todavia no tengo idea de datos de entrenamiento.
        print("Average reward")   
        print(eval_avg_reward)                                                                                                           #Descomentar en el futuro probando distintas "ayudas" de aprendizaje
        mean_action_loss = np.mean(log_action_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)
        total_updates += num_updates_per_iter
        log_str = ("=" * 60 + '\n' +
                "time elapsed: " + time_elapsed  + '\n' +
                "num of updates: " + str(total_updates) + '\n' +
                "action loss: " +  format(mean_action_loss, ".5f") + '\n' +
                "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
                "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' #+
                #"eval d4rl score: " + format(eval_d4rl_score, ".5f")
            )

        print(log_str)
        log_data = [time_elapsed, total_updates, mean_action_loss,
                    eval_avg_reward, eval_avg_ep_len]
                    #eval_d4rl_score]

        csv_writer.writerow(log_data)
        # save model
        print("max d4rl score: " + format(max_d4rl_score, ".5f"))
        if eval_d4rl_score >= max_d4rl_score:
            print("saving max d4rl score model at: " + save_best_model_path)
            torch.save(model.state_dict(), save_best_model_path)
            max_d4rl_score = eval_d4rl_score

        print("saving current model at: " + save_model_path)
        torch.save(model.state_dict(), save_model_path)
    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("max d4rl score: " + format(max_d4rl_score, ".5f"))
    print("saved max d4rl score model at: " + save_best_model_path)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='env_va')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--rtg_scale', type=int, default=1)
    parser.add_argument('--max_eval_ep_len', type=int, default=1000)
    parser.add_argument('--num_eval_ep', type=int, default=1)
    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')
    parser.add_argument('--context_len', type=int, default=64)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)                       
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--max_train_iters', type=int, default=5)   #previous default 5
    parser.add_argument('--num_updates_per_iter', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    train(args)
