import sklearn.metrics as metrics
import torch

from .util import copy_to


def _risk_helper(logit, Y, metric):
    metric = metric.lower()
    if metric == 'f1':
        return metrics.f1_score(Y, logit.argmax(dim=1), average='macro')
    elif metric == 'auc':
        probs = torch.nn.functional.softmax(logit, dim=-1)
        if probs.shape[1] == 2:
            return metrics.roc_auc_score(Y, probs[:, 1], multi_class='ovr')
        else:
            return metrics.roc_auc_score(Y, probs, multi_class='ovr')
    elif metric == 'bac':
        return metrics.balanced_accuracy_score(Y, logit.argmax(dim=1))
    elif metric == 'acc':
        return metrics.accuracy_score(Y, logit.argmax(dim=1))

    raise ValueError(metric)


def risk(model, dataset, size, metric, fwd_args=dict()):
    model.eval()
    dev = next(model.parameters()).device
    truth, logit = [], []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(dataset, batch_size=size):
            truth.append(batch.pop('Y'))
            logit.append(model(true_prior=dataset.true_prior, em_prior=dataset.em_prior, **copy_to(batch, dev), **fwd_args).cpu())
    truth = torch.cat(truth)
    logit = torch.cat(logit, dim=0)

    if isinstance(metric, list):
        return {m: _risk_helper(logit, truth, m) for m in metric}

    return _risk_helper(logit, truth, metric)


def update_chkpnt_(best, model, slot, risk_score, iter_):
    if best['risk_score'][slot] < risk_score:
        best['risk_score'][slot] = risk_score
        best['state_dict'][slot] = copy_to(model.state_dict(), 'cpu')
        best['iter'][slot] = iter_


def validate(model_, best_, envs, batch_size, iteration):
    record = {}
    for i, E in enumerate(envs):
        value = risk(model_, E, batch_size, 'F1')
        update_chkpnt_(best_, model_, i, value, iteration)
        record[f'Val {i}th'] = value

    return record


def risk_round(model, environments, batch_size, fwd_args=dict()):
    def _key(*strings):
        return ','.join(strings)

    out = {}
    for i, data in enumerate(environments):
        scores = risk(model, data, batch_size, ['f1', 'acc'], fwd_args)
        out.update({_key(data.E, k): v for k, v in scores.items()})

    return out
