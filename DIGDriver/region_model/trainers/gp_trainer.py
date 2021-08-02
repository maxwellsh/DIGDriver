import numpy as np
import scipy
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def r2_score(y_true, y_pred):
    r2 = scipy.stats.pearsonr(y_true, y_pred)[0]**2
    return r2 if not np.isnan(r2) else 0


class SparseGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_inducing=2000):
        super(SparseGP, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        base_cov_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

        self.covar_module = gpytorch.kernels.InducingPointKernel(
            base_cov_module,
            inducing_points=train_x[:n_inducing, :],
            likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit_params(self, train_x, train_y, likelihood, n_iter=100):
        pass

    def predict(self, val_x):
        pass


class GPTrainer:
    samp_bound = int(1.5e5)

    def __init__(self, device, train_tup, val_tup, heldout_tup=None, n_iter=50, n_inducing=500):
        self.device = device
        self.n_iter = n_iter
        self.n_inducing = n_inducing
        self.org_train_x = train_tup[0]
        self.org_train_y = train_tup[1]
        self.train_chr_locations = train_tup[2]
        self.train_mappability = train_tup[3]
        self.train_quantiles = train_tup[4]
        self.org_val_x = val_tup[0]
        self.org_val_y = val_tup[1]
        self.val_chr_locations = val_tup[2]
        self.val_mappability = train_tup[3]
        self.val_quantiles = train_tup[4]

        self.train_x, self.train_y, scaler, self.y_mean, self.y_std = self.standardize(train_tup[0], train_tup[1])
        self.val_x,  self.val_y,  _, _, _ = self.standardize(val_tup[0],
                                                             val_tup[1],
                                                             scaler,
                                                             self.y_mean,
                                                             self.y_std)

        self.idx_feat = np.where(np.abs(self.train_x).mean(axis=0) > 0)[0]
        train_size = self.train_x.shape[0]
        if train_size > self.samp_bound:  # upper bound number of samples to fit on GPU memory
            samp_idxs = np.random.choice(self.train_x.shape[0], size=self.samp_bound, replace=False)
            assert len(np.unique(samp_idxs)) == len(samp_idxs)
            self.train_x = self.train_x[samp_idxs]
            self.train_y = self.train_y[samp_idxs]
            print('Reduced train set size from {} to {}, to stay within memory limits'.format(train_size, self.samp_bound))

        self.train_x = self.train_x[:, self.idx_feat]
        self.val_x = self.val_x[:, self.idx_feat]
        print('After zero features reduction feature vectors are now of size: {}'.format(self.train_x.shape[1]))

        if heldout_tup is not None:
            self.org_ho_x = heldout_tup[0]
            self.org_ho_y = heldout_tup[1]
            self.ho_chr_locations = heldout_tup[2]
            self.ho_mappability = heldout_tup[3]
            self.ho_quantiles = heldout_tup[4]
            self.held_x,  self.held_y,  _, _, _ = self.standardize(heldout_tup[0],
                                                                   heldout_tup[1],
                                                                   scaler,
                                                                   self.y_mean,
                                                                   self.y_std)
            self.held_x = self.held_x[:, self.idx_feat]
        else:
            self.held_x,  self.held_y = None, None

    def standardize(self, X, Y, scaler=None, y_mean=None, y_std=None):

        if not scaler:
            scaler = StandardScaler()
            scaler.fit(X)

        if not y_mean:
            y_mean = Y.mean()
            y_std  = Y.std()

        x = scaler.transform(X)
        y = (Y - y_mean) / y_std

        return x, y, scaler, y_mean, y_std

    def train_model(self):
        X = torch.tensor(self.train_x).float().contiguous().to(self.device)
        y = torch.tensor(self.train_y).float().contiguous().to(self.device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = SparseGP(X, y, likelihood, n_inducing=self.n_inducing).to(self.device)
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.8)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(self.n_iter):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = -mll(y_pred, y)
            loss.backward()
            optimizer.step()

        # delete variables to clear memory
        del X
        del y
        del loss
        del optimizer
        del mll
        return model, likelihood

    def predict(self, model, likelihood, x, y):
        model.eval()
        likelihood.eval()
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        X = torch.tensor(x).float().contiguous().to(self.device)
        y_true = torch.tensor(y).float().contiguous().to(self.device)
        print('Predicting over {} samples.'.format(X.size(0)))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_pred = model(X)
            loss = -mll(y_pred, y_true)
            y_hat = y_pred.mean.cpu().numpy()
            y_std = y_pred.stddev.cpu().numpy()

        # delete variables to clear memory
        del X
        return y_hat, y_std, loss.item()

    @staticmethod
    def get_results_dict(mean, std, r2, loss, params):
        return {'gp_mean': mean, 'gp_std': std, 'r2': r2, 'loss': loss, 'params': params}

    def run(self):
        torch.cuda.empty_cache()

        # Train model
        #with gpytorch.settings.cg_tolerance(1e9), gpytorch.settings.debug(False):
        model, likelihood = self.train_model()

        # Validate model
        #with gpytorch.settings.eval_cg_tolerance(1e6):
        val_mean, val_std, val_loss = self.predict(model, likelihood, self.val_x, self.val_y)
        val_r2 = r2_score(self.val_y, val_mean)
        print(bcolors.OKCYAN + 'Validation set R2: {}'.format(val_r2) + bcolors.ENDC)

        params = np.array([model.covar_module.base_kernel.base_kernel.lengthscale.item(),
                           model.covar_module.base_kernel.outputscale.item(),
                           likelihood.noise_covar.noise.item()])

        val_res = self.get_results_dict(val_mean * self.y_std + self.y_mean,
                                        val_std * self.y_std,
                                        val_r2, val_loss, params)

        if self.held_x is not None:
            #with gpytorch.settings.eval_cg_tolerance(1e6):
            hld_mean, hld_std, hld_loss = self.predict(model, likelihood, self.held_x, self.held_y)
            hld_r2 = r2_score(self.held_y, hld_mean)
            print(bcolors.OKCYAN + 'Held-out set R2: {}'.format(hld_r2) + bcolors.ENDC)
            hld_res = self.get_results_dict(hld_mean * self.y_std + self.y_mean,
                                            hld_std * self.y_std,
                                            hld_r2, hld_loss,
                                            params)
            return val_res, hld_res
        return val_res, None

    def save_results(self, val_res_dict, held_res_dict, h5_file, run_id):
        print('Saving GP {} results'.format(int(run_id) + 1))
        if 'train' not in h5_file:
            train_grp = h5_file.create_group('train')
            train_grp.create_dataset('nn_features', data=self.org_train_x)
            train_grp.create_dataset('y_true', data=self.org_train_y)
            train_grp.create_dataset('chr_locs', data=np.array(self.train_chr_locations))
            train_grp.create_dataset('mappability', data=np.array(self.train_mappability))
            train_grp.create_dataset('quantiles', data=np.array(self.train_quantiles))
        if 'val' not in h5_file:
            val_grp = h5_file.create_group('val')
            val_grp.create_dataset('nn_features', data=self.val_x)
            val_grp.create_dataset('y_true', data=self.org_val_y)
            val_grp.create_dataset('chr_locs', data=np.array(self.val_chr_locations))
            val_grp.create_dataset('mappability', data=np.array(self.val_mappability))
            val_grp.create_dataset('quantiles', data=np.array(self.val_quantiles))

        val_run_grp = h5_file['val'].create_group(run_id)
        val_run_grp.create_dataset('mean', data=val_res_dict['gp_mean'])
        val_run_grp.create_dataset('std', data=val_res_dict['gp_std'])
        val_run_grp.create_dataset('params', data=val_res_dict['params'])
        val_run_grp.attrs['R2'] = val_res_dict['r2']
        val_run_grp.attrs['loss'] = val_res_dict['loss']

        if held_res_dict is not None:
            if 'held-out' not in h5_file:
                ho_grp = h5_file.create_group('held-out')
                ho_grp.create_dataset('nn_features', data=self.org_ho_x)
                ho_grp.create_dataset('y_true', data=self.org_ho_y)
                ho_grp.create_dataset('chr_locs', data=np.array(self.ho_chr_locations))
                ho_grp.create_dataset('mappability', data=np.array(self.ho_mappability))
                ho_grp.create_dataset('quantiles', data=np.array(self.ho_quantiles))

            ho_run_grp = h5_file['held-out'].create_group(run_id)
            ho_run_grp.create_dataset('mean', data=held_res_dict['gp_mean'])
            ho_run_grp.create_dataset('std', data=held_res_dict['gp_std'])
            ho_run_grp.create_dataset('params', data=held_res_dict['params'])
            ho_run_grp.attrs['R2'] = held_res_dict['r2']
            ho_run_grp.attrs['loss'] = held_res_dict['loss']
        return val_res_dict['r2'], held_res_dict['r2']

    def compute_pretrained(self, out_h5, runs_num):
        assert 'held-out' in out_h5, 'Cannot compute pretrained model with no saved held-out set. Existing feilds are: {}'.format(out_h5.keys())
        ds = out_h5['held-out']
        chr_locs = ds['chr_locs'][:]
        mapps = ds['mappability'][:]
        quants = ds['quantiles'][:]
        y_true = ds['y_true'][:]
        mean_lst = []
        std_lst = []
        for i in np.arange(runs_num).astype(str):
            mean_lst.append(ds[i]['mean'][:])
            std_lst.append(ds[i]['std'][:])
        means = np.array(mean_lst).mean(axis=0)
        stds = np.array(std_lst).mean(axis=0)
        return chr_locs, mapps, quants, y_true, means, stds

