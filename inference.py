from pathlib import Path
import os
import argparse
import torch
import torch.nn as nn
#from config import get_config, latest_weights_file_path
from train import get_model
from decision_transformer.utils import D4RLTrajectoryDataset, evaluate_on_env
#from translate import translate
#define the device
def inference (args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = get_model(args)
# Load the pretrained weights
model_filename = 'dt_env_va-medium-v2_model_23-08-07-19-39-33.pt'
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])
evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
                                num_eval_ep, max_eval_ep_len, state_mean, state_std) 

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
    inference(args)
