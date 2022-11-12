#!/usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import core


def run_experiment(args):
    out_dir = args.out/f'{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print('Flags:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    core.set_seed(args.seed)

    with open(args.config) as fh:
        conf = json.load(fh)

    root_dir = pathlib.Path(conf['img_dir'])

    adjustment = args.method in {'copa', 'gpa'}

    d_train = [core.CXR_site(root_dir, c, args.zlabels, fit_cond_prev=adjustment) for c in conf['train']]
    d_valid = [core.CXR_site(root_dir, c,  args.zlabels, fit_cond_prev=adjustment) for c in [conf['valid']]]
    d_test = [core.CXR_site(root_dir, c, args.zlabels, fit_cond_prev=adjustment) for c in conf['test']]

    print('finished data loading')

    print('Train Envs:', [len(d) for d in d_train])
    print('Valid Envs:', [len(d) for d in d_valid])
    print('Test  Envs:', [len(d) for d in d_test])

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Zs = [core.CXR_Z_DICT[zl] for zl in args.zlabels]

    trunk = core.get_trunk('r50')(args.hidden_dim)
    head = core.get_head(args.method)(args.hidden_dim, 2, Zs)
    model = core.Classifier(trunk, head).to(dev)

    best, log = core.train_wrapper(args.method, model, d_train, d_valid, d_test, args)

    log.to_csv(out_dir/f'loss_{args.method}.csv.gz', index=False)

    assert len(best['state_dict']) == 1 and best['state_dict'][0] is not None

    model.load_state_dict(best['state_dict'][0])

    rows = []
    if args.method == 'copa':
        rows.append({'N': args.method} | core.risk_round(model, d_test, args.batch_size, {'prior': 'true_prior'}))
        rows.append({'N': f'{args.method}_ast'} | core.risk_round(model, d_test, args.batch_size, {'prior': 'true_prior', 'no_Z': True}))

    elif args.method == 'gpa':
        # calibrate model prediction
        core.calibrate(model, d_valid[0])
        # estimate g for each test environment
        core.get_g_estimate_using_unlabeled_data(model, d_test, akl=core.CXR_AKL)

        rows.append({'N': args.method} | core.risk_round(model, d_test, args.batch_size, {'prior': 'em_prior'}))
        rows.append({'N': f'{args.method}_ast'} | core.risk_round(model, d_test, args.batch_size, {'prior': 'em_prior', 'no_Z': True}))

    else:
        rows.append({'N': args.method} | core.risk_round(model, d_test, args.batch_size))

    out = pd.DataFrame(rows)
    print(out[['N'] + [c for c in out.columns if c.endswith('f1')]])
    out.to_csv(out_dir/f'stats_{args.method}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--method', type=str.lower, required=True)
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--out', '-o', type=pathlib.Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--irm_lambda', type=float)

    parser.add_argument('--config', '-f', type=pathlib.Path, required=True)
    parser.add_argument('--zlabels', nargs='+', choices=['Age', 'AP/PA', 'Sex'], required=True)

    run_experiment(parser.parse_args())
