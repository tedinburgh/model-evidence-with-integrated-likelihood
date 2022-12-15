#!/usr/bin/env python
'''
Author: Tom Edinburgh
v1: date 14/10/2022.

This script defines the all functions needed for computation of the integrated
likelihoods involved in Bayesian model selection with multilevel linear models,
via a simulation study involving Gaussian data.
'''

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import argparse
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import model_setup_with_integrated_likelihoods as ilike
from scipy.spatial import distance
from matplotlib.backends.backend_pdf import PdfPages
from palettable.colorbrewer import sequential as sq

###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description='''
            Evaluate multilevel models using SMC with integrated and full
            likelihoods and with Akaike information criterion.
            ''')
    parser.add_argument(
        '--model_list',
        default='0, 1, 2, 3',
        help='''List of models to evaluate from those detailed in the
            simulation section of the accompanying manuscript''')
    parser.add_argument(
        '--data_list',
        default='0, 1, 2, 3',
        help='''List of simulated datasets from those detailed in the
            simulation section of the accompanying manuscript''')
    parser.add_argument(
        '--save_output',
        default=True,
        help='''Boolean to indicate whether to save output, including traces
            from PyMC3 sampling''',
        type=bool)
    parser.add_argument(
        '--data_output_file_path',
        default='simulated_data/',
        help='''File path to the directory that saves outputs from this script
            ''',
        type=str)
    parser.add_argument(
        '--table_output_file_path',
        default='figures_tables/',
        help='''
            File path to the directory that saves figures and tables from this
            script
            ''',
        type=str)
    parser.add_argument(
        '--generate_figures',
        default=True,
        help='''Generate figure containing the simulated datasets''',
        type=bool)
    parser.add_argument(
        '--compute_posterior',
        default=True,
        help='''Compute the posterior distributions for beta and eta''',
        type=bool)
    parser.add_argument(
        '--n',
        default=1000,
        help='Size of simulated datasets',
        type=int)
    parser.add_argument(
        '--n_chains',
        default=8,
        help='''Number of sequential Monte Carlo chains (random initialisation)
            ''',
        type=int)
    parser.add_argument(
        '--n_draws',
        default=2000,
        help='Length of each sequential Monte Carlo chain',
        type=int)
    parser.add_argument('--RANDOM_SEED', default=8924, type=int)
    parser.add_argument('--np_random_seed', default=286, type=int)
    args = parser.parse_args()
    return args


###############################################################################
# Functions: data simulation


def simulate_x(
        n, t=None, J=10, random_seed=7723, intercept=True,  # random_seed=7919
        d2=3, P=1, s_x=None, s_z=None, min=0, max=1):
    if s_x is None:
        s_x = [0, 0.25, 0.5, 0.75]
    if s_z is None:
        s_z = [0.5]
    d1 = len(s_x)
    np.random.seed(random_seed)
    p = np.random.dirichlet(np.arange(J) + 1)
    j = np.random.choice(np.arange(J), size=n, p=p)
    x = np.zeros((n, 0))
    z = np.zeros((n, 0))
    if intercept is True:
        x = np.ones((n, 1))
        z = np.ones((n, 1))
    if t is None:
        t = np.random.uniform(low=min, high=max, size=n)
    if len(t.shape) > 1:
        order = np.argsort(t[:, 0])
    else:
        order = np.argsort(t)
    t = t[order].reshape(n, 1)
    j = j[order]
    for s in s_x:
        x = np.hstack((x, (t - s) * (t > s)))
    for s in s_z:
        constant = 0.5 * (max - min) * (max - min - s)**2
        z = np.hstack((z, (t - s) * (t > s) - constant))
    x = np.hstack((
        x, np.hstack([np.cos(2*(m + 1)*np.pi*t / P) for m in range(d2)])))
    x = np.hstack((
        x, np.hstack([np.sin(2*(m + 1)*np.pi*t / P) for m in range(d2)])))
    output = dict(x=x, j=j, z=z, t=t, d1=d1, d2=d2)
    return output


def get_d0(
        n=5000, J=10, random_seed=7907,
        S=None, sigma2_a=3, sigma2_b=2, **x_args):
    '''Linear model'''
    x_output = simulate_x(n=n, J=J, **x_args)
    x = x_output['x']
    z = x_output['z']
    j = x_output['j']
    d = x.shape[1]
    np.random.seed(random_seed)
    if S is None:
        S = np.eye(d)
        # M = np.eye(d)
        # M[np.arange(d1) + 2, np.arange(d1) + 1] = -1
        # S = np.dot(np.dot(M, S), M.T)
    b = np.random.multivariate_normal(np.repeat(0, d), S)
    # If x ~ gamma(a, b) where a is shape and b = rate, then 1 / x ~ IG(a, b)
    # numpy gamma uses shape and scale, so for e.g. x ~ IG(a, b), sample
    # y ~ gamma(a, 1 / b), x = 1 / y
    var = 1 / np.random.gamma(sigma2_a, 1 / sigma2_b)
    sd = np.sqrt(var)
    e = np.random.normal(0, sd, size=n)  # requires standard deviation, not var
    y = np.dot(x, b) + e
    t_output = simulate_x(n=n, t=np.linspace(0, 1, n), J=1, **x_args)
    y_plot = np.dot(t_output['x'], b)
    t = t_output['t']
    output = dict(y=y, x=x, j=j, z=z, b=b, var=var, e=e, t=t, y_plot=y_plot)
    return output


def get_d1(
        n=5000, J=10, random_seed=7901, S=None,
        sigma2_a=3, sigma2_b=2, sigma2_h_a=3, sigma2_h_b=2, **x_args):
    '''Simple multilevel model'''
    x_output = simulate_x(n=n, J=J, **x_args)
    x = x_output['x']
    z = x_output['z']
    j = x_output['j']
    d = x.shape[1]
    np.random.seed(random_seed)
    if S is None:
        S = np.eye(d)
    b = np.random.multivariate_normal(np.repeat(0, d), S)
    # If x ~ gamma(a, b) where a is shape and b = rate, then 1 / x ~ IG(a, b)
    # numpy gamma uses shape and scale, so for e.g. x ~ IG(a, b), sample
    # y ~ gamma(a, 1 / b), x = 1 / y
    var_y = 1 / np.random.gamma(sigma2_a, 1 / sigma2_b)
    sd_y = np.sqrt(var_y)
    var_h = 1 / np.random.gamma(sigma2_h_a, 1 / sigma2_h_b)
    sd_h = np.sqrt(var_h)
    e = np.random.normal(0, sd_y, size=n)  # requires std, not var
    h = np.random.normal(0, sd_h, size=J)
    y = np.dot(x, b) + h[j] + e
    t_output = simulate_x(n=n, t=np.linspace(0, 1, n), J=1, **x_args)
    y_plot = np.zeros((n, J + 1))
    y_plot[:, 0] = np.dot(t_output['x'], b)
    for jj in range(J):
        y_plot[:, jj + 1] = np.dot(t_output['x'], b) + h[jj]
    t = t_output['t']
    output = dict(
        y=y, x=x, j=j, z=z, b=b, var_y=var_y, var_h=var_h, e=e, h=h,
        t=t, y_plot=y_plot)
    return output


def get_d2(
        n=5000, J=10, random_seed=7883,
        S=None, sigma2_a=3, sigma2_b=2, sigma2_h_a=3, sigma2_h_b=2,
        rho=0, **x_args):
    '''General multilevel model'''
    x_output = simulate_x(n=n, J=J, **x_args)
    x = x_output['x']
    z = x_output['z']
    j = x_output['j']
    d = x.shape[1]
    m = z.shape[1]
    np.random.seed(random_seed)
    if S is None:
        S = np.eye(d)
    b = np.random.multivariate_normal(np.repeat(0, d), S)
    # If x ~ gamma(a, b) where a is shape and b = rate, then 1 / x ~ IG(a, b)
    # numpy gamma uses shape and scale, so for e.g. x ~ IG(a, b), sample
    # y ~ gamma(a, 1 / b), x = 1 / y
    var_y = 1 / np.random.gamma(sigma2_a, 1 / sigma2_b)
    sd_y = np.sqrt(var_y)
    var_h = 1 / np.random.gamma(sigma2_h_a, 1 / sigma2_h_b, size=m)
    sd_h = np.sqrt(var_h)
    if m > 1:
        S_h = np.diag(var_h)
        if np.array(rho).size < m:
            rho = np.repeat(rho, m)[:m]
        for ii in range(m - 1):
            S_h[[ii, ii + 1], [ii + 1, ii]] = rho[ii]*sd_h[ii]*sd_h[ii + 1]
    else:
        S_h = sd_h.reshape(1, 1)
    e = np.random.normal(0, sd_y, size=n)  # requires std, not var
    h = np.random.multivariate_normal(np.repeat(0, m), S_h, size=J)
    y = np.dot(x, b) + np.sum(h[j] * z, axis=1) + e
    t_output = simulate_x(n=n, t=np.linspace(0, 1, n), J=1, **x_args)
    y_plot = np.zeros((n, J + 1))
    y_plot[:, 0] = np.dot(t_output['x'], b)
    for jj in range(J):
        y_plot[:, jj + 1] = (
            np.dot(t_output['x'], b) + np.sum(h[jj] * z, axis=1))
    t = t_output['t']
    output = dict(
        y=y, x=x, j=j, z=z, b=b, var_y=var_y, S_h=S_h,
        var_h=var_h, e=e, h=h, t=t, y_plot=y_plot)
    return output


def get_d3(
        n=5000, J=10, random_seed=7879,
        S=None, sigma2_a=3, sigma2_b=2, gamma=1, **x_args):
    '''Linear model, using normal-inverse-gamma distribution'''
    x_output = simulate_x(n=n, J=J, **x_args)
    x = x_output['x']
    z = x_output['z']
    j = x_output['j']
    d = x.shape[1]
    np.random.seed(random_seed)
    if S is None:
        S = np.eye(d)
    # If x ~ gamma(a, b) where a is shape and b = rate, then 1 / x ~ IG(a, b)
    # numpy gamma uses shape and scale, so for e.g. x ~ IG(a, b), sample
    # y ~ gamma(a, 1 / b), x = 1 / y
    var = 1 / np.random.gamma(sigma2_a, 1 / sigma2_b)
    sd = np.sqrt(var)
    b = np.random.multivariate_normal(np.repeat(0, d), var*gamma*S)
    e = np.random.normal(0, sd, size=n)  # requires standard deviation, not var
    y = np.dot(x, b) + e
    t_output = simulate_x(n=n, t=np.linspace(0, 1, n), J=1, **x_args)
    y_plot = np.dot(t_output['x'], b)
    t = t_output['t']
    output = dict(y=y, x=x, j=j, z=z, b=b, var=var, e=e, t=t, y_plot=y_plot)
    return output

###############################################################################
# Functions: model definitions


def sample_lm_model(
        m_data, m_str, n_draws, random_seed, n_chains,
        save_output, output_file_path, prior_dict=None):
    '''
    Sample from the integrated and full likelihoods
    respectively for linear model
    '''
    if prior_dict is None:
        prior_dict = dict()
    sampling_time = dict(integrated=np.nan, full=np.nan)
    logp_ = None
    # Marginal likelihood sampling
    start_time = time.time()
    with pm.Model() as smc_m:
        logp_input_dict = ilike.get_lm_integrated(smc_m, m_data, **prior_dict)
        pm.Potential('y', ilike.logp_lm(**logp_input_dict))
        trace_integrated = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        sampling_time['integrated'] = (time.time() - start_time)
        if save_output:
            filename = output_file_path + '/trace_integrated_' + m_str + '.nc'
            az.to_netcdf(
                trace_integrated.posterior,
                filename=filename)
    # Full likelihood sampling
    start_time = time.time()
    with pm.Model() as smc_m_full:
        y, sigma2 = ilike.get_lm_full(smc_m_full, m_data, **prior_dict)
        pm.Normal('y_obs', mu=y, tau=1/sigma2, observed=m_data['y'])
        trace_full = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        sampling_time['full'] = (time.time() - start_time)
        if save_output:
            az.to_netcdf(
                trace_full.posterior,
                filename=output_file_path + '/trace_full_' + m_str + '.nc')
    return trace_integrated, trace_full, sampling_time, logp_


def sample_lm_ninvg_model(
        m_data, m_str, n_draws, random_seed, n_chains,
        save_output, output_file_path, prior_dict=None):
    '''
    Sample from the integrated and full likelihoods respectively for linear
    model, using a normal-inverse-gamma prior
    '''
    if prior_dict is None:
        prior_dict = dict()
    if 'gamma' not in prior_dict.keys():
        prior_dict['gamma'] = 1
    sampling_time = dict(integrated=np.nan, full=np.nan)
    # Marginal likelihood sampling
    xy_dict = ilike.xy_sums(**m_data)
    xy_dict['d'] = m_data['x'].shape[1]
    if 'beta_Sigma_inv' in prior_dict.keys():
        xy_dict['Sigma_inv'] = prior_dict['beta_Sigma_inv']
    else:
        xy_dict['Sigma_inv'] = np.eye(xy_dict['d'])
    if 'sigma2_alpha' in prior_dict.keys():
        xy_dict['sigma2_alpha'] = prior_dict['sigma2_alpha']
    if 'sigma2_beta' in prior_dict.keys():
        xy_dict['sigma2_beta'] = prior_dict['sigma2_beta']
    xy_dict['gamma'] = prior_dict['gamma']
    logp_ = ilike.logp_evidence_ninvg_lm(**xy_dict)
    # Marginal likelihood sampling
    start_time = time.time()
    with pm.Model() as smc_m:
        prior_dict.pop('gamma', None)
        logp_input_dict = ilike.get_lm_integrated(smc_m, m_data, **prior_dict)
        prior_dict['gamma'] = xy_dict['gamma']
        logp_input_dict['gamma'] = prior_dict['gamma']
        pm.Potential('y', ilike.logp_ninvg_lm(**logp_input_dict))
        trace_integrated = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        sampling_time['integrated'] = (time.time() - start_time)
        if save_output:
            filename = output_file_path + '/trace_integrated_' + m_str + '.nc'
            az.to_netcdf(
                trace_integrated.posterior,
                filename=filename)
    # Full likelihood sampling
    start_time = time.time()
    with pm.Model() as smc_m_full:
        y, sigma2 = ilike.get_lm_ninvg_full(smc_m_full, m_data, **prior_dict)
        pm.Normal('y_obs', mu=y, tau=1/sigma2, observed=m_data['y'])
        trace_full = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        sampling_time['full'] = (time.time() - start_time)
        if save_output:
            az.to_netcdf(
                trace_full.posterior,
                filename=output_file_path + 'trace_full_' + m_str + '.nc')
    return trace_integrated, trace_full, sampling_time, logp_


def sample_mm_model(
        m_data, m_str, n_draws, random_seed, n_chains,
        save_output, output_file_path, prior_dict=None):
    '''
    Sample from the integrated and full likelihoods respectively for simple
    multilevel linear model
    '''
    if prior_dict is None:
        prior_dict = dict()
    sampling_time = dict(integrated=np.nan, full=np.nan)
    logp_ = None
    # Marginal likelihood sampling
    start_time = time.time()
    with pm.Model() as smc_m:
        logp_input_dict = ilike.get_mm_integrated(smc_m, m_data, **prior_dict)
        pm.Potential('y', ilike.logp_mm(**logp_input_dict))
        trace_integrated = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        sampling_time['integrated'] = (time.time() - start_time)
        if save_output:
            filename = output_file_path + 'trace_integrated_' + m_str + '.nc'
            az.to_netcdf(
                trace_integrated.posterior,
                filename=filename)
    # Full likelihood sampling
    start_time = time.time()
    with pm.Model() as smc_m_full:
        y, sigma2 = ilike.get_mm_full(smc_m_full, m_data, **prior_dict)
        pm.Normal('y_obs', mu=y, tau=1/sigma2, observed=m_data['y'])
        trace_full = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        sampling_time['full'] = (time.time() - start_time)
        if save_output:
            az.to_netcdf(
                trace_full.posterior,
                filename=output_file_path + 'trace_full_' + m_str + '.nc')
    return trace_integrated, trace_full, sampling_time, logp_


def sample_general_mm_model(
        m_data, m_str, n_draws, random_seed, n_chains,
        save_output, output_file_path, prior_dict=None):
    '''
    Sample from the integrated and full likelihoods respectively for more
    general multilevel linear model
    '''
    if prior_dict is None:
        prior_dict = dict()
    sampling_time = dict(integrated=np.nan, full=np.nan)
    logp_ = None
    # Marginal likelihood sampling
    start_time = time.time()
    with pm.Model() as smc_m:
        logp_input_dict = ilike.get_general_mm_integrated(
            smc_m, m_data, **prior_dict)
        pm.Potential('y', ilike.logp_general_mm(**logp_input_dict))
        trace_integrated = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        sampling_time['integrated'] = (time.time() - start_time)
        if save_output:
            filename = output_file_path + 'trace_integrated_' + m_str + '.nc'
            az.to_netcdf(
                trace_integrated.posterior,
                filename=filename)
    # Full likelihood sampling
    start_time = time.time()
    with pm.Model() as smc_m_full:
        y, sigma2 = ilike.get_general_mm_full(
            smc_m_full, m_data, **prior_dict)
        pm.Normal('y_obs', mu=y, tau=1/sigma2, observed=m_data['y'])
        trace_full = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        sampling_time['full'] = (time.time() - start_time)
        if save_output:
            az.to_netcdf(
                trace_full.posterior,
                filename=output_file_path + 'trace_full_' + m_str + '.nc')
    return trace_integrated, trace_full, sampling_time, logp_

###############################################################################
# Simulated dataset figure


def plot_simulated_data(
        data_output_file_path, figure_output_file_path, random_seed=286):
    '''Generate Figure 1, showing all simulated datasets'''
    data0 = pd.read_csv(data_output_file_path + 'data0.csv')
    data1 = pd.read_csv(data_output_file_path + 'data1.csv')
    data2 = pd.read_csv(data_output_file_path + 'data2.csv')
    data3 = pd.read_csv(data_output_file_path + 'data3.csv')

    rowcols = [[0, 0], [0, 1], [1, 0], [1, 1]]
    data_list = [data0, data1, data2, data3]

    # markers = ['o', 'v', 's', '*', 'D'] * 3
    colours = sq.Purples_7.mpl_colors[2:] + sq.Greens_7.mpl_colors[2:]
    colours = colours + sq.PuRd_7.mpl_colors
    colours = colours + sq.Blues_7.mpl_colors[2:] + sq.YlOrRd_7.mpl_colors
    np.random.seed(random_seed)
    J = np.unique(data0['j']).size
    colour_ind = np.random.choice(range(len(colours)), J)
    colours = [colours[x] for x in colour_ind]
    cmap, norm = mpl.colors.from_levels_and_colors(
        np.arange(16) - 0.5, colours)
    s = 3
    fit_j = [False, True, True, False]
    titles = [r'$D_0$', r'$D_1$', r'$D_2$', r'$D_3$']

    with PdfPages(figure_output_file_path + 'simulated_datasets.pdf') as pdf:
        fig, ax = plt.subplots(
            nrows=2, ncols=2, sharex=True, sharey=True, figsize=[8, 7])

        for kk in range(4):
            data = data_list[kk]
            rowcol = rowcols[kk]
            J = np.unique(data['j']).size
            # n = data.shape[0]
            scatter = ax[rowcol[0], rowcol[1]].scatter(
                data['x1'], data['y'],
                c=data['j'], s=s, cmap=cmap, norm=norm,
                marker='o', alpha=0.5)
            if fit_j[kk]:
                for jj in range(J):
                    ax[rowcol[0], rowcol[1]].plot(
                        data['t'], data['y_plot' + str(jj + 1)],
                        color=colours[jj], alpha=0.5)
            ax[rowcol[0], rowcol[1]].plot(
                data['t'], data['y_plot0'], color='k', lw=2)
            ax[rowcol[0], rowcol[1]].text(0.5, 2.6, titles[kk], fontsize=15)

        ax[0, 0].set_ylabel(r'$y_{ij}$')
        ax[1, 0].set_ylabel(r'$y_{ij}$')
        ax[1, 0].set_xlabel(r'$t_{ij}$')
        ax[1, 1].set_xlabel(r'$t_{ij}$')
        handles, _ = scatter.legend_elements(num=J)
        legend1 = ax[0, 1].legend(
            handles=handles, labels=['j=' + str(x + 1) for x in range(J)],
            loc='center left', title='Groups', bbox_to_anchor=(1, 0))
        ax[0, 1].add_artist(legend1)

        plt.tight_layout(rect=(0, 0, 0.9, 1))
        plt.subplots_adjust(wspace=0, hspace=0)
        tif_fig_name = figure_output_file_path + 'simulated_datasets.tif'
        plt.savefig(tif_fig_name, format='tiff')
        pdf.savefig(fig)
        plt.close()
    return


###############################################################################
# Define model data

def main():
    '''This is the main function, which runs all sampling on the data'''
    inputs = parse_args()
    n = inputs.n

    s_x = [0, 0.2, 0.4, 0.6, 0.8]
    s_z = [0, 0.4, 0.8]
    d1 = len(s_x)
    d2 = 20
    d = 1 + d1 + 2*d2
    m = 1 + len(s_z)
    S1 = np.diag([1, 4, 5, 10, 5, 6])
    S1[[1, 1, 2, 2, 3, 4], [2, 3, 3, 4, 4, 5]] = [-3, -1, -4, 2, -4, 2]
    S1[[2, 3, 3, 4, 4, 5], [1, 1, 2, 2, 3, 4]] = [-3, -1, -4, 2, -4, 2]
    # S1 = np.diag([0.5, 2, 2.5, 6, 2.5, 5])
    # S1[[1, 1, 2, 2, 3, 4], [2, 3, 3, 4, 4, 5]] = [-1.5, -0.5, -2, 1, -3, 2]
    # S1[[2, 3, 3, 4, 4, 5], [1, 1, 2, 2, 3, 4]] = [-1.5, -0.5, -2, 1, -3, 2]
    S = np.eye(d) * 0.001
    S[:(1 + d1), :(1 + d1)] = S1
    J = 15
    sigma2_a = 3
    sigma2_b = 0.4
    sigma2_y_b = 0.3
    sigma2_h_b = 0.1
    rho = [0, 0.2, 0.2]
    rho_fixed = np.diag(rho, 1)[np.triu_indices(np.diag(rho, 1).shape[0])]
    gamma = (sigma2_a - 1) / sigma2_b

    data_inputs = dict(
        n=n, J=J, S=S, sigma2_a=sigma2_a, P=1, s_x=s_x, s_z=s_z, d2=d2)
    data0 = get_d0(**data_inputs, sigma2_b=sigma2_b, random_seed=6997)
    data1 = get_d1(
        **data_inputs, sigma2_b=sigma2_y_b,
        sigma2_h_b=sigma2_h_b, random_seed=6883)
    data2 = get_d2(
        **data_inputs, sigma2_b=sigma2_y_b, sigma2_h_b=sigma2_h_b,
        rho=rho, random_seed=6773)
    data3 = get_d3(
        **data_inputs, sigma2_b=sigma2_b, gamma=gamma, random_seed=6473)
    d_dict_list = [data0, data1, data2, data3]

    if inputs.save_output:
        d = np.max([d_dict['x'].shape[1] for d_dict in d_dict_list])
        var_params = [
            x for d_dict in d_dict_list
            for x in d_dict.keys() if x.startswith('var')]
        var_params = list(np.unique(var_params))
        data_params = pd.DataFrame(
            columns=['d0', 'd1', 'd2', 'd3'],
            index=['b' + str(x) for x in range(d)] + var_params)
        for ii in range(4):
            d_dict = d_dict_list[ii]
            d = d_dict['x'].shape[1]
            df_columns = ['y'] + ['x' + str(x) for x in range(d)]
            m = d_dict['z'].shape[1]
            df_columns += ['z' + str(x) for x in range(m)]
            df_columns += ['j', 'e']
            if 'h' in d_dict.keys():
                m = d_dict['h'].reshape(J, -1).shape[1]
                df_columns += ['h' + str(x) for x in range(m)]
            q = d_dict['y_plot'].reshape(n, -1).shape[1]
            df_columns += ['t'] + ['y_plot' + str(x) for x in range(q)]
            df = pd.DataFrame(index=range(n), columns=df_columns)
            df['y'] = d_dict['y']
            df['j'] = d_dict['j']
            df['e'] = d_dict['e']
            if 'z' in d_dict.keys():
                df.loc[:, df.columns.str.startswith('z')] = d_dict['z']
            if 'h' in d_dict.keys():
                df.loc[:, df.columns.str.startswith('h')] = (
                    d_dict['h'][d_dict['j']])
            df.loc[:, df.columns.str.startswith('x')] = d_dict['x']
            df['t'] = d_dict['t']
            df.loc[:, df.columns.str.startswith('y_plot')] = d_dict['y_plot']
            var_params = [x for x in d_dict.keys() if x.startswith('var')]
            data_params.loc[(
                    data_params.index.str.startswith('b')), 'd' + str(ii)] = (
                d_dict['b'])
            for var_param in var_params:
                data_params.loc[var_param, 'd' + str(ii)] = d_dict[var_param]
            df.to_csv(
                inputs.data_output_file_path + 'data' + str(ii) + '.csv',
                index=False)
        data_params.to_csv(inputs.data_output_file_path + 'data_params.csv')

    sample_inputs = dict(
        n_draws=inputs.n_draws,
        random_seed=inputs.RANDOM_SEED,
        n_chains=inputs.n_chains,
        save_output=inputs.save_output,
        output_file_path=inputs.data_output_file_path)

    model_list = [int(x) for x in inputs.model_list.strip(' ').split(',')]
    data_list = [int(x) for x in inputs.data_list.strip(' ').split(',')]

    sample_model_list = [sample_lm_model, sample_mm_model]
    sample_model_list += [sample_general_mm_model, sample_lm_ninvg_model]
    sample_model_list = [sample_model_list[m] for m in model_list]

    # Sigma retains only the diagonal elements of S
    Sigma_inv = np.diag(1 / np.diag(S))
    mu = np.zeros(Sigma_inv.shape[0])

    prior_dict_list = [
        dict(
            sigma2_alpha=sigma2_a, sigma2_beta=sigma2_b,
            beta_mu=mu, beta_Sigma_inv=Sigma_inv),
        dict(
            sigma2_y_alpha=sigma2_a, sigma2_y_beta=sigma2_y_b,
            sigma2_eta_alpha=sigma2_a, sigma2_eta_beta=sigma2_h_b,
            beta_mu=mu, beta_Sigma_inv=Sigma_inv),
        dict(
            sigma2_y_alpha=sigma2_a, sigma2_y_beta=sigma2_y_b,
            sigma2_eta_alpha=sigma2_a, sigma2_eta_beta=sigma2_h_b,
            beta_mu=mu, beta_Sigma_inv=Sigma_inv,
            include_rho=True, rho_fixed=rho_fixed),
        dict(
            sigma2_alpha=sigma2_a, sigma2_beta=sigma2_b,
            gamma=gamma, beta_mu=mu, beta_Sigma_inv=Sigma_inv)]
    prior_dict_list = [prior_dict_list[m] for m in model_list]

    np.random.seed(inputs.np_random_seed)
    # Set up SMC sampling results table
    smc_column_list = [
        type + '_m' + str(m) + '-d' + str(d)
        for d in data_list for m in model_list
        for type in ['integrated', 'full']]
    smc_column_list += ['analytic_m3-d' + str(d) for d in data_list]
    smc_lml = pd.DataFrame(
        columns=smc_column_list, index=range(inputs.n_chains))

    smc_time = pd.DataFrame(columns=smc_column_list, index=range(1))

    for m in model_list:
        for d in data_list:
            print('Starting: model m' + str(m) + ' and data d' + str(d))
            d_dict = d_dict_list[d]
            prior_dict = prior_dict_list[m]
            k_include = ['y', 'x']
            if m == 1:
                k_include += ['j']
            if m == 2:
                k_include += ['j', 'z']
            d_dict = dict((k, d_dict[k]) for k in k_include)
            sample_model = sample_model_list[m]
            m_str = 'd' + str(d) + '-m' + str(m)
            trace_integrated, trace_full, sampling_time, logp_ = sample_model(
                d_dict, m_str, **sample_inputs, prior_dict=prior_dict)
            smc_time['integrated_m' + str(m) + '-d' + str(d)] = (
                sampling_time['integrated'])
            smc_time['full_m' + str(m) + '-d' + str(d)] = (
                sampling_time['full'])
            if m == 3:
                # The direct logp_ for the linear model with NIG prior
                smc_lml['analytic_m3-d' + str(d)] = logp_
            # Marginal likelihood
            lml = trace_integrated['sample_stats']['log_marginal_likelihood']
            if trace_integrated['sample_stats']['chain'].size == 1:
                lml = [
                    lml[:, ii].values[0][-1]
                    for ii in range(inputs.n_chains)]
            else:
                lml = lml[:, -1].values
            smc_lml['integrated_m' + str(m) + '-d' + str(d)] = lml
            # Full likelihood
            lml = trace_full['sample_stats']['log_marginal_likelihood']
            if trace_full['sample_stats']['chain'].size == 1:
                lml = [
                    lml[:, ii].values[0][-1]
                    for ii in range(inputs.n_chains)]
            else:
                lml = lml[:, -1].values
            smc_lml['full_m' + str(m) + '-d' + str(d)] = lml
            if inputs.save_output:
                smc_lml.to_csv(
                    inputs.data_output_file_path + 'smc_likelihoods.csv',
                    index=False)
                smc_time.to_csv(
                    inputs.data_output_file_path + 'smc_time.csv', index=False)

    # Table with model evidence
    smc_table_columns = [
        'logp(D' + str(d) + '|M) integrated' for d in data_list]
    smc_table_columns += ['Time, s integrated']
    smc_table_columns += [
        'logp(D' + str(d) + '|M) full' for d in data_list]
    smc_table_columns += ['Time, s full']
    smc_table_columns += [
        'logp(D' + str(d) + '|M) analytic' for d in data_list]
    smc_table = pd.DataFrame(
        columns=smc_table_columns,
        index=['M' + str(m) for m in model_list])
    for d in data_list:
        for y in ['integrated', 'full']:
            asterisk = smc_lml.loc[:, (
                    smc_lml.columns.str.contains('d' + str(d)) &
                    smc_lml.columns.str.contains(y))
                ].mean()
            asterisk = (asterisk == asterisk.max())
            asterisk = asterisk.replace({True: '*', False: ''})
            a_row = [
                x.strip(y + '_').rsplit('-d' + str(d), 1)[0].upper()
                for x in asterisk.index]
            asterisk = asterisk.rename(index=dict(zip(asterisk.index, a_row)))
            for m in model_list:
                smc_lml_column = y + '_m' + str(m) + '-d' + str(d)
                smc_table.loc['M' + str(m), 'logp(D' + str(d) + '|M) ' + y] = (
                    str(np.round(smc_lml[smc_lml_column].mean(), 2)) +
                    ' (' +
                    str(np.round(smc_lml[smc_lml_column].std(), 2)) + ')' +
                    asterisk.loc['M' + str(m)])
    for d in data_list:
        smc_lml_column = 'analytic_m3-d' + str(d)
        smc_table.loc['M3', 'logp(D' + str(d) + '|M) analytic'] = (
            str(np.round(smc_lml[smc_lml_column].mean(), 2)))
    for m in model_list:
        smc_table.loc['M' + str(m), 'Time, s integrated'] = smc_time.loc[:, (
                smc_time.columns.str.contains('m' + str(m)) &
                smc_time.columns.str.contains('integrated'))
            ].mean(axis=1).round(1).astype(str).values[0]
    for m in model_list:
        smc_table.loc['M' + str(m), 'Time, s full'] = smc_time.loc[:, (
                smc_time.columns.str.contains('m' + str(m)) &
                smc_time.columns.str.contains('full'))
            ].mean(axis=1).round(1).astype(str).values[0]
    if inputs.save_output:
        smc_table.to_csv(inputs.table_output_file_path + 'tableI.csv')

    # Posterior
    if inputs.compute_posterior:
        mdist_df = pd.DataFrame(
            index=range(len(model_list) * len(data_list)),
            columns=['m', 'd', 'integrated', 'full'])
        mdist_df['m'] = np.repeat(model_list, len(data_list))
        mdist_df['d'] = np.tile(data_list, len(model_list))

        xy_fun_list = [ilike.xy_sums, ilike.xy_j_sums, ilike.xyz_j_sums]
        xy_fun_list += [ilike.xy_sums]
        xy_fun_list = [xy_fun_list[m] for m in model_list]
        mu_fun_list = [ilike.mu_tilde_lm, ilike.mu_hat_mm]
        mu_fun_list += [ilike.mu_hat_general_mm, ilike.mu_tilde_ninvg_lm]
        mu_fun_list = [mu_fun_list[m] for m in model_list]

        d_param = data_params['coef'].str.startswith('b').sum()
        mu_df = pd.DataFrame(columns=smc_lml.columns, index=range(d_param))

        for ii in mdist_df.index:
            m = mdist_df.loc[ii, 'm']
            d = mdist_df.loc[ii, 'd']
            print('Starting: model m' + str(m) + ' and data d' + str(d))
            d_dict = d_dict_list[d]
            prior_dict = prior_dict_list[m]
            k_include = ['y', 'x']
            if m == 1:
                k_include += ['j']
            if m == 2:
                k_include += ['j', 'z']
            d_dict = dict((k, d_dict[k]) for k in k_include)
            xy_fun = xy_fun_list[m]
            mu_fun = mu_fun_list[m]
            xy_dict = xy_fun(**d_dict)
            b = data_params.loc[
                data_params['coef'].str.startswith('b'), 'd' + str(d)].values
            filename = inputs.data_output_file_path
            filename += 'trace_integrated_d' + str(d) + '-m' + str(m) + '.nc'
            trace_integrated = az.from_netcdf(filename)
            tm_pos = trace_integrated['posterior']
            mu_tildes = np.zeros((
                tm_pos.coords.dims['chain'], tm_pos.coords.dims['draw'],
                d_param))
            Sigma_tilde_invs = np.zeros((
                tm_pos.coords.dims['chain'], tm_pos.coords.dims['draw'],
                d_param, d_param))
            for jj in range(tm_pos.coords.dims['chain']):
                for kk in range(tm_pos.coords.dims['draw']):
                    if m in [0, 3]:
                        sigma2_dict = dict(
                            sigma2=tm_pos['sigma2'][jj, kk].values)
                    elif m == 1:
                        sigma2_dict = dict(
                            sigma2_y=tm_pos['sigma2_y'][jj, kk].values,
                            sigma2_eta=tm_pos['sigma2_eta'][jj, kk].values)
                    elif m == 2:
                        sigma2_dict = dict(
                            sigma2_y=tm_pos['sigma2_y'][jj, kk].values,
                            Sigma_eta_inv=(
                                tm_pos['Sigma_eta_inv'][jj, kk].values))
                    if m == 3:
                        sigma2_dict['gamma'] = prior_dict['gamma']
                    sigma2_dict['beta_mu'] = mu
                    sigma2_dict['beta_Sigma_inv'] = Sigma_inv
                    output = mu_fun(**sigma2_dict, **xy_dict)
                    mu_tildes[jj, kk, :] = output['mu_tilde']
                    Sigma_tilde_invs[jj, kk, :] = output['Sigma_tilde_inv']
            Sigma_tilde_inv_integrated = Sigma_tilde_invs.mean(axis=(0, 1))
            mu_tilde_integrated = mu_tildes.mean(axis=(0, 1))
            mu_df['integrated_m' + str(m) + '-d' + str(d)] = (
                mu_tilde_integrated)
            mdist_df.loc[ii, 'integrated'] = distance.mahalanobis(
                b, mu_tilde_integrated, Sigma_tilde_inv_integrated)
            filename = inputs.data_output_file_path + 'trace_full_d' + str(d)
            filename += '-m' + str(m) + '.nc'
            trace_full = az.from_netcdf(filename)
            Sigma_tilde_full = np.cov(
                trace_full['posterior']['beta'].values.reshape(-1, d_param).T)
            Sigma_tilde_inv_full = np.linalg.inv(Sigma_tilde_full)
            mu_tilde_full = trace_full['posterior']['beta'].mean(axis=(0, 1))
            mu_tilde_full = mu_tilde_full.values
            mu_df['full_m' + str(m) + '-d' + str(d)] = mu_tilde_full
            mdist_df.loc[ii, 'full'] = distance.mahalanobis(
                b, mu_tilde_full, Sigma_tilde_inv_full)

        mdist_table = pd.DataFrame(
            columns=[
                'D' + str(d) + ', b: ' + y
                for y in ['integrated', 'full'] for d in data_list],
            index=['M' + str(m) for m in model_list])
        for ii in mdist_df.index:
            m = mdist_df.loc[ii, 'm']
            d = mdist_df.loc[ii, 'd']
            for y in ['integrated', 'full']:
                mdist_table.loc['M' + str(m), 'D' + str(d) + ', b: ' + y] = (
                    mdist_df.round(2).loc[ii, y])

        if inputs.save_output:
            mdist_table.to_csv(
                inputs.table_output_file_path + 'tableII.csv', index=True)
            mdist_df.round(2).to_csv(
                inputs.data_output_file_path + 'mdist_df.csv', index=False)

    # Figure
    if inputs.generate_figures:
        plot_simulated_data(
            inputs.data_output_file_path, inputs.table_output_file_path)
    return


if __name__ == '__main__':
    main()
