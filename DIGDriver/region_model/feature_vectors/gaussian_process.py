#!/usr/bin/env python

import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
import h5py
import scipy.stats
import tqdm
import argparse

class SparseGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_inducing=2000):
        super(SparseGP, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        base_cov_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            base_cov_module,
            inducing_points = train_x[:n_inducing, :],
            likelihood=likelihood
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit_params(self, train_x, train_y, likelihood, n_iter=100):
        pass

    def predict(self, test_x):
        pass

def load(fname, dataset, idx_feat=np.array([])):
    f = h5py.File(fname, 'r')

    if dataset not in f.keys():
        f.close()
        return np.array([]), np.array([]), idx_feat

    X = f[dataset]['X'][:]
    Y = f[dataset]['y'][:]
    # X = f[dataset]['features'][0, :, :]
    # Y = f[dataset]['true'][0, :]

    if not idx_feat.any():
        idx_feat = np.where(np.abs(X).mean(axis=0) > 0)[0]

    X = X[:, idx_feat]

    f.close()

    return X, Y, idx_feat

def standardize(X, Y, scaler=None, y_mean=None, y_std=None):

    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)

    if not y_mean:
        y_mean = Y.mean()
        y_std  = Y.std()

    x = scaler.transform(X)
    y = (Y - y_mean) / y_std

    return x, y, scaler, y_mean, y_std

def train_model(train_x, train_y, n_iter=100, n_inducing=2000):
    # train_x = torch.FloatTensor(train_x).contiguous().cuda()
    # train_y = torch.FloatTensor(train_y).contiguous().cuda()

    # if torch.cuda.is_available():
    #     train_x, train_y = train_x.cuda(), train_y.cuda();

    likelihood = to_gpu(
        gpytorch.likelihoods.GaussianLikelihood()
    )
    model = to_gpu(
        SparseGP(train_x, train_y, likelihood, n_inducing=n_inducing)
    )

    # if torch.cuda.is_available():
    #     model, likelihood = model.cuda(), likelihood.cuda()
    
    model.train()
    likelihood.train()

    print(f'Training model with {n_iter} iterations.')
    # model.fit_params(train_x, train_y, likelihood, n_iter=n_iter)
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.8)
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # with gpytorch.settings.max_cg_iterations(10000):
    # with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
    # with gpytorch.settings.max_preconditioner_size(80):
    iterator = tqdm.tqdm(range(n_iter), desc='GP training')
    for i in iterator:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()

    print(f"Finished training on {train_x.size(0)} samples.")
    print("Final values - Loss: %.3f   lengthscale: %.3f   outputscale: %.3f   noise: %.3f" % (
               loss.item(),
               model.covar_module.base_kernel.base_kernel.lengthscale.item(),
               model.covar_module.base_kernel.outputscale.item(),
               likelihood.noise_covar.noise.item()
    ))

    return model, likelihood, loss.item()

def predict(model, likelihood, test_x):
    model.eval()
    likelihood.eval()

    # test_x = torch.FloatTensor(test_x).contiguous()
    # if torch.cuda.is_available():
    #     print('cuda')
    #     test_x = test_x.cuda()

    print(f'Predicting over {test_x.size(0)} test samples.')
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
        # with gpytorch.settings.max_root_decomposition_size(30), gpytorch.settings.fast_pred_var():
        with gpytorch.settings.max_cg_iterations(10000):
            y_pred = model(test_x)

            y_hat = y_pred.mean.cpu().numpy()
            y_std = y_pred.stddev.cpu().numpy()

    return y_hat, y_std

def save(fname, dataset, y_hat, y_std, loss, r2, params):
    f = h5py.File(fname, 'r+')
    data = f[dataset]
    keys = [key for key in data.keys()]

    keys_mean = [key for key in keys if key.startswith('gp_mean')]
    if keys_mean:
        suffix_lst = [int(key.split('_')[-1]) for key in keys_mean]
        sfx = max(suffix_lst) + 1

    else:
        sfx = 1

    print('Saving GP results into {} gp_*_{:02d}'.format(dataset, sfx))
    data.create_dataset('gp_mean_{:02d}'.format(sfx), data=y_hat)
    data.create_dataset('gp_std_{:02d}'.format(sfx), data=y_std)
    data.create_dataset('gp_params_{:02d}'.format(sfx), data=params)
    data.attrs['gp_loss_{:02d}'.format(sfx)] = loss
    data.attrs['gp_R2_{:02d}'.format(sfx)] = r2

def to_torch(data):
    return torch.FloatTensor(data).contiguous()

def to_gpu(data):
    if torch.cuda.is_available():
        return data.cuda()

def parse_args():
    parser = argparse.ArgumentParser(description='Fit a sparse Gaussian Process')

    parser.add_argument('data', help='h5 file containing train and test data')
    parser.add_argument('--n_iter', type=int, default=100, help='number of training iterations')
    parser.add_argument('--n_inducing', type=int, default=2000, help='number of inducing points')
    parser.add_argument('--n_runs', type=int, default=5, help='number of runs to train the model')
    parser.add_argument('--save-train', action='store_true', default=False, help='save training data')

    return parser.parse_args()

def run():
    args = parse_args()

    ## Load data
    train_X, train_Y, idx_feat = load(args.data, 'train')
    test_X,  test_Y,  _        = load(args.data, 'test', idx_feat)
    held_X,  held_Y,  _        = load(args.data, 'held-out', idx_feat)
    # print(held_Y[0:5])

    ## Standardize data
    train_X, train_Y, scaler, y_mean, y_std = standardize(train_X, train_Y)
    test_X,  test_Y,  _,      _,      _     = standardize(test_X, test_Y, scaler, y_mean, y_std)

    train_x, train_y, test_x = to_torch(train_X), to_torch(train_Y), to_torch(test_X)
    train_x, train_y, test_x = to_gpu(train_x), to_gpu(train_y), to_gpu(test_x)

    ## Train model
    model, likelihood, loss = train_model(train_x, train_y, 
        n_iter=args.n_iter, n_inducing=args.n_inducing
    )

    ## Validate model
    gp_mean, gp_std = predict(model, likelihood, test_x)
    r2 = r2_score(test_Y, gp_mean)
    print(f'R^2 of model: {r2}')

    params = np.array([model.covar_module.base_kernel.base_kernel.lengthscale.item(),
                       model.covar_module.base_kernel.outputscale.item(),
                       likelihood.noise_covar.noise.item()
    ])

    save(args.data, 'test', gp_mean*y_std + y_mean, gp_std * y_std, loss, r2, params)

    if args.save_train:
        print('Saving training data')
        train_mean, train_std = predict(model, likelihood, train_x)
        r2 = r2_score(train_Y, train_mean)
        print(r2)
        save(args.data, 'train', train_mean*y_std + y_mean, train_std * y_std, loss, r2, params)

    if held_X.any():
        print('Applying GP to heldout data')
        held_X, held_Y, _, _, _ = standardize(held_X, held_Y, scaler, y_mean, y_std)
        held_x = to_gpu(to_torch(held_X))

        hld_mean, hld_std = predict(model, likelihood, held_x)
        r2 = r2_score(held_Y, hld_mean)
        print(r2)
        save(args.data, 'held-out', hld_mean*y_std + y_mean, hld_std * y_std, loss, r2, params)

if __name__ == "__main__":
    run()
