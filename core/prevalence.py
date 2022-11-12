import copy
import itertools
import torch
import torch.nn as nn


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


def prior_NLL(prior_model, log_odd, bZ):
    log_prior = nn.functional.log_softmax(prior_model(bZ), dim=-1)

    pY = nn.functional.softmax(log_odd + log_prior, dim=-1).detach()

    return -torch.sum(pY * log_prior)


def calibrate(f_net, val_env):
    prior_model = val_env.true_prior
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    f_net.to(dev)
    prior_model.to(dev)

    print([f_net.classifier.ws, f_net.classifier.bs])

    opt = torch.optim.LBFGS([f_net.classifier.ws, f_net.classifier.bs])
    feat = []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(val_env.X, batch_size=256):
            feat.append(f_net.featurizer(batch.to(dev)))
    feat = torch.cat(feat, dim=0)
    Zd, Yd =  val_env.Z.to(dev), val_env.Y.to(dev)

    def closure():
        opt.zero_grad()
        logit = f_net.classifier(feat, Zd, prior='true_prior', true_prior=prior_model)
        loss = nn.functional.cross_entropy(logit, Yd)
        loss.backward()
        return loss

    opt.step(closure)
    print([f_net.classifier.ws, f_net.classifier.bs])


def est_g_by_EM(f_net, prior_model, X, Z, EM_steps, akl=0, verbose=False):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    f_net.to(dev)
    prior_model.to(dev)
    opt = torch.optim.LBFGS(prior_model.parameters())
    feat = []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(X, batch_size=256):
            feat.append(f_net.featurizer(batch.to(dev)))
    feat = torch.cat(feat, dim=0).to(dev)

    aug_Z = []
    for mask in powerset(range(Z.shape[1])):
        in_ = Z.clone().detach().to(device=dev, dtype=torch.float)
        in_[:, mask] = torch.nan
        aug_Z.append(in_)

    def closure():
        opt.zero_grad()

        loss = 0
        head = f_net.classifier
        for Zd in aug_Z:
            log_prior = nn.functional.log_softmax(prior_model(Zd), dim=-1)

            log_odd = head._get_ratio(feat, Zd)

            pY = nn.functional.softmax((log_odd + log_prior)*head.ws + head.bs, dim=-1).detach()

            loss += -torch.sum(pY * log_prior)

            unif = torch.ones_like(log_prior) / log_prior.shape[-1]
            loss += akl * nn.functional.kl_div(log_prior, unif, reduction='sum')  # regularization

        loss.backward()
        return loss

    for step in range(EM_steps):
        opt.step(closure)
        if verbose:
            log_odd = f_net.classifier._get_ratio(feat, Z)
            print(f'\t{step=} NLL={prior_NLL(prior_model, log_odd, Z):.3f}')


def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


def get_g_estimate_using_unlabeled_data(f_net, test_envs, akl=0, EM_steps=5):
    for data_env in test_envs:
        em_prior = copy.deepcopy(data_env.true_prior)
        reset_all_weights(em_prior)
        est_g_by_EM(f_net, em_prior, data_env.X, data_env.Z, akl=akl, EM_steps=EM_steps)
        data_env.em_prior = em_prior


def est_multivariate_g(g_model, Y, Z, akl=0):
    assert len(Y.shape) == 1, Y.shape
    assert Z.dim() == 2, f'{Z.dim()=}'

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g_model.to(dev)

    aug_Z, aug_Y = [], []
    for mask in powerset(range(Z.shape[1])):
        in_ = Z.clone().detach().to(device=dev, dtype=torch.float)
        in_[:, mask] = torch.nan
        aug_Z.append(in_)
        aug_Y.append(Y)

    aug_Z = torch.cat(aug_Z, dim=0).to(dev)
    aug_Y = torch.cat(aug_Y, dim=0).to(dev)

    opt = torch.optim.LBFGS(g_model.parameters())

    def closure():
        opt.zero_grad()
        logit = g_model(aug_Z)
        loss = nn.functional.cross_entropy(logit, aug_Y)

        log_prior = nn.functional.log_softmax(logit, dim=-1)
        unif = torch.ones_like(log_prior) / log_prior.shape[-1]
        loss += akl * nn.functional.kl_div(log_prior, unif, reduction='batchmean')  # regularization

        loss.backward()
        return loss

    opt.step(closure)

    pred = torch.softmax(g_model(Z.to(dev)), dim=1)
    mse = float(((1 - pred[torch.arange(len(Y)), Y])**2).mean())

    return mse
