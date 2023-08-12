# coding: utf-8
# @email  : enoche.chow@gmail.com

"""
Utility functions
##########################
"""

import numpy as np
import pandas as pd
import torch
import importlib
import datetime
import random
import gc
from tqdm import tqdm


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')

    return cur


def get_model(model_name, saved=False):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """
    model_file_name = model_name.lower()
    module_path = '.'.join(['models', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer():
    return getattr(importlib.import_module('common.trainer'), 'Trainer')


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ': ' + '%.04f' % value + '    '
    return result_str


############ LATTICE Utilities #########

def build_knn_neighbourhood(adj, topk):
    print('build_knn_neighbourhood')
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj):
    print('compute_normalized_laplacian')
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    print('multiple d_mat_inv_sqrt, adj')
    L_norm = torch.mm(d_mat_inv_sqrt, adj)
    print('multiple d_mat_inv_sqrt, adj, d_mat_inv_sqrt')
    L_norm = torch.mm(L_norm, d_mat_inv_sqrt)
    return L_norm

def build_sim(context):
    print('build_sim')
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

def make_submit_csv(trainer, data, config, hyper_tuple, top_k) :
    trainer.config['topk'] = trainer.config['topk']+ [top_k]
    trainer.evaluator.topk = trainer.config['topk']
    _, _, df = trainer._valid_epoch(data)
    
    df.rename(columns=lambda x: int(x.replace('top_', '')) if 'top_' in x else x, inplace=True)
    melted_df = pd.melt(df, id_vars=['id'], value_name='item_id')
    sorted_df = melted_df.groupby('id').apply(lambda group: group.sort_values(by='variable'))
    save_df = sorted_df.reset_index(drop=True).reset_index(drop=True).drop('variable', axis=1)
    save_df.rename(columns={'id' : 'user_id'}, inplace=True)
    save_df['user_id'] = save_df['user_id'].astype('int32')
    save_df['item_id'] = save_df['item_id'].astype('int32')
    
    model = config['model']
    dataset = config['dataset']
    date = get_local_time()
    hyper = config['hyper_parameters']
    
    save_df.to_csv(f'../csv/{model}-{dataset}-{date}-{hyper}-{hyper_tuple}-{top_k}.csv', index=False)
    
    group_df = save_df.groupby('user_id').head(50)
    group_df.reset_index()
    group_df.to_csv(f'../csv/{model}-{dataset}-{date}-{hyper}-{hyper_tuple}-50.csv', index=False)