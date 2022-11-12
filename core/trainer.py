from .alg_erm import AlgERM
from .alg_irm import AlgIRM


def train_wrapper(n_method, model, d_train, d_valid, d_test, args):
    if n_method in ('erm', 'erm2', 'copa', 'gpa', 'ora'):
        return AlgERM(**vars(args))(model, d_train, d_valid)

    if n_method in ('irm', ):
        return AlgIRM(**vars(args))(model, d_train, d_valid)

    raise ValueError(f'Unsupported: {n_method=}')
