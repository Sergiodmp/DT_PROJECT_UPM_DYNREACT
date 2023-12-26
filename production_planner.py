from pathlib import Path
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

from decision_transformer.utils import D4RLTrajectoryDataset, evaluate_on_env, get_d4rl_normalized_score
from decision_transformer.model import DecisionTransformer
from prueba_env import env_va


def production_planner(args)
#define the device and the model
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
env = env_va()      #instanciamos la clase
state_dim = env.observation_space.shape[0]   #it takes the shape that we have established in the environment
act_dim = env.action_space.n
device = torch.device("cuda" if torch.cuda is_available() else "cpu")
print("using device:", device)
model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                n_blocks=n_blocks,
                h_dim=embed_dim,
                context_len=context_len,
                n_heads=n_heads,
                drop_p=dropout_p,
            )
#load the pretrained weights
model_filename='dt_env_va-medium-v2_model_23-08-07-19-39-33.pt'
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])



#plan the production
model.eval()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='env_va')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--rtg_scale', type=int, default=1)
    parser.add_argument('--max_eval_ep_len', type=int, default=1000)
    parser.add_argument('--num_eval_ep', type=int, default=10)
    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')
    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)                       
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--max_train_iters', type=int, default=200)
    parser.add_argument('--num_updates_per_iter', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    production_planner(args)