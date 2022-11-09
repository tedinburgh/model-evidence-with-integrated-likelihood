#!/usr/bin/env python
'''
Author: Tom Edinburgh
v2: date 14/10/2022.

This script defines the all functions needed for computation of the integrated
likelihoods involved in Bayesian model selection with multilevel linear models,
using example data from the PyMC package.
'''

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import argparse
import time
import statsmodels.api as sm
import model_setup_with_integrated_likelihoods as msm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description='''
            Evaluate multilevel models using SMC with integrated and full
            likelihoods and with Akaike information criterion.
            ''')
    parser.add_argument(
        '--model_list',
        default='0, 1, 2, 3, 4, 5',
        help='''List of models to evaluate from those detailed in the
            accompanying manuscript''')
    parser.add_argument(
        '--save_output',
        default=True,
        help='''Boolean to indicate whether to save output, including traces
            from PyMC3 sampling''',
        type=bool)
    parser.add_argument(
        '--print_output',
        default=True,
        help='''Boolean to indicate whether to print output table''',
        type=bool)
    parser.add_argument(
        '--data_output_file_path',
        default='radon_data/',
        help='''File path to the directory that saves outputs from this script
            ''',
        type=str)
    parser.add_argument(
        '--table_output_file_path',
        default='figures_tables/',
        help='''
            File path to the directory that saves tables and figures
            from this script
            ''',
        type=str)
    parser.add_argument(
        '--generate_figures',
        default=True,
        help='''Generate figure containing the simulated datasets''',
        type=bool)
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
# Functions: process data, model definitions


def process_data():
    # Import radon data
    srrs2 = pd.read_csv(pm.get_data('srrs2.dat'))
    srrs2.columns = srrs2.columns.map(str.strip)
    srrs_mn = srrs2[srrs2.state == "MN"].copy()

    srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips
    cty = pd.read_csv(pm.get_data("cty.dat"))
    cty_mn = cty[cty.st == "MN"].copy()
    cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips

    srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips")
    srrs_mn = srrs_mn.drop_duplicates(subset="idnum")
    srrs_mn['log_uranium'] = np.log(srrs_mn['Uppm'])

    srrs_mn.county = srrs_mn.county.map(str.strip)
    mn_counties = srrs_mn.county.unique()
    counties = len(mn_counties)
    county_lookup = dict(zip(mn_counties, range(counties)))

    srrs_mn["county_code"] = srrs_mn.county.replace(county_lookup).values
    srrs_mn["log_radon"] = np.log(srrs_mn['activity'] + 0.1).values
    return srrs_mn


def get_m0_vars(df):
    '''Complete pooling, linear model'''
    y = df['log_radon'].values
    y = (y - np.mean(y)) / np.std(y)
    x = np.array([1 - df['floor'].values, df['floor'].values]).T
    output = dict(y=y, x=x)
    return output


def get_m1_vars(df):
    '''Complete pooling with group-level predictors, linear model'''
    y = df['log_radon'].values
    y = (y - np.mean(y)) / np.std(y)
    x = np.array([1 - df['floor'].values, df['floor'].values]).T
    v = df['log_uranium'].values.reshape(-1, 1)
    v = (v - np.mean(v)) / np.std(v)
    x = np.hstack([x, v])
    output = dict(y=y, x=x)
    return output


def get_m2_vars(df):
    '''Unpooled intercept and pooled predictors, linear model'''
    y = df['log_radon'].values
    y = (y - np.mean(y)) / np.std(y)
    x = np.array([(df['county'] == c).values for c in df['county'].unique()]).T
    x = np.hstack([x, (1 - df['floor'].values).reshape(-1, 1)])
    x = np.hstack([x, df['floor'].values.reshape(-1, 1)])
    output = dict(y=y, x=x)
    return output


def get_m3_vars(df, discard_missing=True):
    '''No pooling (no group-level covariates), linear model'''
    y = df['log_radon'].values
    y = (y - np.mean(y)) / np.std(y)
    x = np.array([(df['county'] == c).values for c in df['county'].unique()]).T
    x = np.hstack([
        x * (1 - df['floor'].values).reshape(-1, 1),
        x * df['floor'].values.reshape(-1, 1)])
    if discard_missing:
        x = x[:, x.sum(axis=0) > 0]
    output = dict(y=y, x=x)
    return output


def get_m4_vars(df):
    '''Partial pooling, multilevel model'''
    y = df['log_radon'].values
    y = (y - np.mean(y)) / np.std(y)
    x = np.array([1 - df['floor'].values, df['floor'].values]).T
    v = df['log_uranium'].values.reshape(-1, 1)
    v = (v - np.mean(v)) / np.std(v)
    x = np.hstack([x, v])
    j = df['county'].values
    output = dict(y=y, x=x, j=j)
    return output


def get_m5_vars(df):
    '''Varying slopes and intercepts'''
    y = df['log_radon'].values
    y = (y - np.mean(y)) / np.std(y)
    x = np.array([1 - df['floor'].values, df['floor'].values]).T
    v = df['log_uranium'].values.reshape(-1, 1)
    v = (v - np.mean(v)) / np.std(v)
    x = np.hstack([x, v])
    z = np.array([1 - df['floor'].values, df['floor'].values]).T
    j = df['county'].values
    output = dict(y=y, x=x, z=z, j=j)
    return output


def sample_lm_model(
        m_data, m_str, n_draws, random_seed, n_chains,
        save_output, output_file_path):
    '''
    Sample from the integrated and full likelihoods
    respectively for linear model
    '''
    # Marginal likelihood sampling
    print(time.ctime())
    with pm.Model() as smc_m:
        logp_input_dict = msm.get_lm_integrated(smc_m, m_data)
        pm.Potential('y', msm.logp_lm(**logp_input_dict))
        trace_integrated = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        if save_output:
            filename = output_file_path + 'trace_integrated_' + m_str + '.nc'
            az.to_netcdf(
                trace_integrated.posterior,
                filename=filename)
    # Full likelihood sampling
    print(time.ctime())
    with pm.Model() as smc_m_full:
        y, sigma2 = msm.get_lm_full(smc_m_full, m_data)
        pm.Normal('y_obs', mu=y, tau=1/sigma2, observed=m_data['y'])
        trace_full = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        if save_output:
            az.to_netcdf(
                trace_full.posterior,
                filename=output_file_path + 'trace_full_' + m_str + '.nc')
    return trace_integrated, trace_full


def sample_mm_model(
        m_data, m_str, n_draws, random_seed, n_chains,
        save_output, output_file_path):
    '''
    Sample from the integrated and full likelihoods respectively for simple
    multilevel linear model
    '''
    # Marginal likelihood sampling
    print(time.ctime())
    with pm.Model() as smc_m:
        logp_input_dict = msm.get_mm_integrated(smc_m, m_data)
        pm.Potential('y', msm.logp_mm(**logp_input_dict))
        trace_integrated = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        if save_output:
            filename = output_file_path + 'trace_integrated_' + m_str + '.nc'
            az.to_netcdf(
                trace_integrated.posterior,
                filename=filename)
    # Full likelihood sampling
    print(time.ctime())
    with pm.Model() as smc_m_full:
        y, sigma2 = msm.get_mm_full(smc_m_full, m_data)
        pm.Normal('y_obs', mu=y, tau=1/sigma2, observed=m_data['y'])
        trace_full = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        if save_output:
            az.to_netcdf(
                trace_full.posterior,
                filename=output_file_path + 'trace_full_' + m_str + '.nc')
    return trace_integrated, trace_full


def sample_general_mm_model(
        m_data, m_str, n_draws, random_seed, n_chains,
        save_output, output_file_path):
    '''
    Sample from the integrated and full likelihoods respectively for more
    general multilevel linear model
    '''
    # Marginal likelihood sampling
    print(time.ctime())
    with pm.Model() as smc_m:
        logp_input_dict = msm.get_general_mm_integrated(smc_m, m_data)
        pm.Potential('y', msm.logp_general_mm(**logp_input_dict))
        trace_integrated = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        if save_output:
            filename = output_file_path + 'trace_integrated_' + m_str + '.nc'
            az.to_netcdf(
                trace_integrated.posterior,
                filename=filename)
    # Full likelihood sampling
    print(time.ctime())
    with pm.Model() as smc_m_full:
        y, sigma2 = msm.get_general_mm_full(
            smc_m_full, m_data, include_rho=True)
        pm.Normal('y_obs', mu=y, tau=1/sigma2, observed=m_data['y'])
        trace_full = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        if save_output:
            az.to_netcdf(
                trace_full.posterior,
                filename=output_file_path + 'trace_full_' + m_str + '.nc')
    return trace_integrated, trace_full


###############################################################################
# Frequentist approaches: AIC


def aic_lm_fit(m_data):
    '''
    Frequentist model fit for the linear model, to facilitate AIC computation
    '''
    fit = sm.GLM(
            endog=m_data['y'], exog=m_data['x'], family=sm.families.Gaussian()
        ).fit(reml=False)
    return fit


def aic_mm_fit(m_data):
    '''
    Frequentist model fit for the simple multilevel model, to facilitate AIC
    computation
    '''
    fit = sm.MixedLM(
            endog=m_data['y'], exog=m_data['x'], groups=m_data['j']
        ).fit(reml=False, method='lbfgs')
    return fit


def aic_general_mm_fit(m_data):
    '''
    Frequentist model fit for the more general multilevel model, to facilitate
    AIC computation
    '''
    fit = sm.MixedLM(
            endog=m_data['y'], exog=m_data['x'],
            exog_re=m_data['z'], groups=m_data['j']
        ).fit(reml=False, method='lbfgs')
    # # Note: to remove off-diagonal covariance, use instead:
    # import statsmodels.regression.mixed_linear_model as mlm
    # m5_free = mlm.MixedLMParams.from_components(
    #     fe_params=np.ones(m_data['x'].shape[1]),
    #     cov_re=np.eye(m_data['z'].shape[1]))
    # m5_fit = sm.MixedLM(
    #         endog=m_data['y'], exog=m_data['x'],
    #         exog_re=m_data['z'], groups=m_data['j']
    #     ).fit(reml=False, method='lbfgs', free=m_free)
    return fit


def model_fits_and_plot():
    inputs = parse_args()
    np.random.seed(inputs.np_random_seed)
    srrs_mn = process_data()
    get_vars_list = [
        get_m0_vars, get_m1_vars, get_m2_vars,
        get_m3_vars, get_m4_vars, get_m5_vars]
    model_list = [int(x) for x in inputs.model_list.strip(' ').split(',')]

    d_max = 0
    J_max = 0
    for m in model_list:
        get_vars = get_vars_list[m]
        m_data = get_vars(srrs_mn.copy())
        d_param = m_data['x'].shape[1]
        d_max = max((d_max, d_param))
        if m in [4, 5]:
            J_max = max((np.unique(m_data['j']).size, J_max))

    beta_estimate_integrated_df = pd.DataFrame(
        columns=[
            y + '_m' + str(m) for m in model_list for y in ['mean', 'var']],
        index=range(d_max))
    eta_estimate_integrated_df = pd.DataFrame(
        columns=[
            y + '_m' + str(m)
            for m in ['4', '5-0', '5-1'] for y in ['mean', 'var']],
        index=range(J_max))
    beta_estimate_full_df = pd.DataFrame(
        columns=[
            y + '_m' + str(m) for m in model_list for y in ['mean', 'var']],
        index=range(d_max))
    eta_estimate_full_df = pd.DataFrame(
        columns=[
            y + '_m' + str(m)
            for m in ['4', '5-0', '5-1'] for y in ['mean', 'var']],
        index=range(J_max))

    xy_fun_list = [msm.xy_sums] * 4 + [msm.xy_j_sums, msm.xyz_j_sums]
    xy_fun_list = [xy_fun_list[m] for m in model_list]
    mu_fun_list = [msm.mu_tilde_lm] * 4
    mu_fun_list += [msm.mu_hat_mm, msm.mu_hat_general_mm]
    mu_fun_list = [mu_fun_list[m] for m in model_list]

    for m in model_list:
        if inputs.print_output:
            print('Starting sampling with model M' + str(m))
        get_vars = get_vars_list[m]
        m_data = get_vars(srrs_mn.copy())
        d_param = m_data['x'].shape[1]
        xy_fun = xy_fun_list[m]
        mu_fun = mu_fun_list[m]
        xy_dict = xy_fun(**m_data)
        filename = inputs.data_output_file_path + 'trace_integrated_' + str(m)
        filename += '.nc'
        trace_integrated = az.from_netcdf(filename)
        tm_pos = trace_integrated['posterior']
        mu_tildes = np.zeros((
            tm_pos.coords.dims['chain'], tm_pos.coords.dims['draw'], d_param))
        Sigma_tilde_invs = np.zeros((
            tm_pos.coords.dims['chain'], tm_pos.coords.dims['draw'],
            d_param, d_param))
        m_param = 0
        J = 0
        if m == 4:
            m_param = 1
            J = xy_dict['J']
        elif m == 5:
            m_param = tm_pos.coords.dims['sigma2_eta_dim_0']
            J = xy_dict['J']
        eta_hats = np.zeros((
            tm_pos.coords.dims['chain'], tm_pos.coords.dims['draw'],
            J, m_param))
        Sigma_hat_eta_invs = np.zeros((
            tm_pos.coords.dims['chain'], tm_pos.coords.dims['draw'],
            J, m_param, m_param))
        beta_mu = np.zeros(d_param)
        beta_Sigma_inv = np.eye(d_param)
        for jj in range(tm_pos.coords.dims['chain']):
            for kk in range(tm_pos.coords.dims['draw']):
                if m in [0, 1, 2, 3]:
                    sigma2_dict = dict(sigma2=tm_pos['sigma2'][jj, kk].values)
                elif m == 4:
                    sigma2_dict = dict(
                        sigma2_y=tm_pos['sigma2_y'][jj, kk].values,
                        sigma2_eta=tm_pos['sigma2_eta'][jj, kk].values)
                elif m == 5:
                    sigma2_dict = dict(
                        sigma2_y=tm_pos['sigma2_y'][jj, kk].values,
                        Sigma_eta_inv=tm_pos['Sigma_eta_inv'][jj, kk].values)
                sigma2_dict['beta_mu'] = beta_mu
                sigma2_dict['beta_Sigma_inv'] = beta_Sigma_inv
                if m == 4:
                    xy_extra_dict = msm.xy_j_extra_sums(**m_data)
                    output = mu_fun(**sigma2_dict, **xy_dict, **xy_extra_dict)
                else:
                    output = mu_fun(**sigma2_dict, **xy_dict)
                mu_tildes[jj, kk, :] = output['mu_tilde']
                Sigma_tilde_invs[jj, kk, :] = output['Sigma_tilde_inv']
                if m == 4:
                    eta_hats[jj, kk, :] = output['eta_hat']
                    Sigma_hat_eta_invs[jj, kk, :] = output['sigma2_hat_eta']
                if m == 5:
                    eta_hats[jj, kk, :] = output['eta_hat']
                    Sigma_hat_eta_invs[jj, kk, :] = output['Sigma_hat_eta_inv']
        Sigma_tilde_inv_integrated = Sigma_tilde_invs.mean(axis=(0, 1))
        mu_tilde_integrated = mu_tildes.mean(axis=(0, 1))
        beta_estimate_integrated_df.loc[range(d_param), 'mean_m' + str(m)] = (
            mu_tilde_integrated)
        beta_estimate_integrated_df.loc[range(d_param), 'var_m' + str(m)] = (
            np.diag(np.linalg.inv(Sigma_tilde_inv_integrated)))
        if m == 4:
            Sigma_hat_eta_inv_integrated = (
                Sigma_hat_eta_invs.mean(axis=(0, 1)))
            eta_hat_integrated = eta_hats.mean(axis=(0, 1))
            eta_estimate_integrated_df.loc[
                    range(J), 'mean_m' + str(m)] = (
                eta_hat_integrated[:, 0])
            eta_estimate_integrated_df.loc[
                    range(J), 'var_m' + str(m)] = (
                Sigma_hat_eta_inv_integrated[:, 0, 0])
        elif m == 5:
            for kk in range(eta_hats.shape[2]):
                Sigma_hat_eta_estimate = np.linalg.inv(
                    Sigma_hat_eta_invs[:, :, kk].mean(axis=(0, 1)))
                eta_hat_estimate = eta_hats[:, :, kk].mean(axis=(0, 1))
                temp_columns = [
                    'mean_m' + str(m) + '-' + x for x in ['0', '1']]
                eta_estimate_integrated_df.loc[kk, temp_columns] = (
                    eta_hat_estimate)
                temp_columns = ['var_m' + str(m) + '-' + x for x in ['0', '1']]
                eta_estimate_integrated_df.loc[kk, temp_columns] = np.diag(
                    Sigma_hat_eta_estimate)
        filename = inputs.data_output_file_path
        filename += 'trace_full_' + str(m) + '.nc'
        trace_full = az.from_netcdf(filename)
        tf_pos = trace_full['posterior']
        Sigma_tilde_full = np.cov(tf_pos['beta'].values.reshape(-1, d_param).T)
        mu_tilde_full = tf_pos['beta'].mean(axis=(0, 1))
        mu_tilde_full = mu_tilde_full.values
        beta_estimate_full_df.loc[range(d_param), 'mean_m' + str(m)] = (
            mu_tilde_full)
        beta_estimate_full_df.loc[range(d_param), 'var_m' + str(m)] = (
            np.diag(Sigma_tilde_full))
        if m == 4:
            eta_cov_full = np.cov(tf_pos['eta'].values.reshape(-1, J).T)
            eta_estimate_full = tf_pos['eta'].mean(axis=(0, 1))
            eta_estimate_full = eta_estimate_full.values
            eta_estimate_full_df.loc[range(J), 'mean_m' + str(m)] = (
                eta_estimate_full)
            eta_estimate_full_df.loc[range(J), 'var_m' + str(m)] = (
                np.diag(eta_cov_full))
        if m == 5:
            for kk in range(tf_pos['eta'].shape[3]):
                vals = tf_pos['eta'].values[:, :, :, kk]
                eta_cov_full = np.cov(vals.reshape(-1, J).T)
                eta_estimate_full = vals.mean(axis=(0, 1))
                eta_estimate_full_df.loc[
                        range(J), 'mean_m' + str(m) + '-' + str(kk)] = (
                    eta_estimate_full)
                eta_estimate_full_df.loc[
                        range(J), 'var_m' + str(m) + '-' + str(kk)] = (
                    np.diag(eta_cov_full))

    if inputs.save_output:
        beta_estimate_integrated_df.to_csv(
            inputs.data_output_file_path + 'beta_estimates.csv', index=False)
        eta_estimate_integrated_df.to_csv(
            inputs.data_output_file_path + 'eta_estimates.csv', index=False)
        beta_estimate_full_df.to_csv(
            inputs.data_output_file_path + 'beta_estimates_full.csv',
            index=False)
        eta_estimate_full_df.to_csv(
            inputs.data_output_file_path + 'eta_estimates_full.csv',
            index=False)

    beta_estimates = beta_estimate_integrated_df
    eta_estimates = eta_estimate_integrated_df

    m1_data = get_m1_vars(srrs_mn.copy())
    m4_data = get_m4_vars(srrs_mn.copy())
    m5_data = get_m5_vars(srrs_mn.copy())
    j_ind = [35, 0, 34, 20, 13, 70, 60, 69]  # Following PyMC tutorial
    j_list = [np.unique(m5_data['j'])[x] for x in j_ind]
    j_unique = np.unique(m5_data['j'])
    model_fits_columns = [
        'm' + str(m) + '_f' + str(y) + z
        for m in model_list for y in [0, 1] for z in ['_mean', '_var']]
    model_fits = pd.DataFrame(columns=model_fits_columns, index=j_list)
    for m in model_list:
        if m == 0:
            model_fits['m0_f0_mean'] = beta_estimates.loc[0, 'mean_m0']
            model_fits['m0_f0_var'] = beta_estimates.loc[0, 'var_m0']
            model_fits['m0_f1_mean'] = beta_estimates.loc[1, 'mean_m0']
            model_fits['m0_f1_var'] = beta_estimates.loc[1, 'var_m0']
        if m == 1:
            model_fits['m1_f0_mean'] = beta_estimates.loc[0, 'mean_m1']
            model_fits['m1_f0_mean'] += np.array([
                m1_data['x'][:, 2][m5_data['j'] == j_unique[j]][0]
                for j in j_ind]) * beta_estimates.loc[2, 'mean_m1']
            model_fits['m1_f0_var'] = beta_estimates.loc[0, 'var_m1']
            model_fits['m1_f0_var'] += np.array([
                m1_data['x'][:, 2][m5_data['j'] == j_unique[j]][0]**2
                for j in j_ind]) * beta_estimates.loc[2, 'var_m1']
            model_fits['m1_f1_mean'] = beta_estimates.loc[1, 'mean_m1']
            model_fits['m1_f1_mean'] += np.array([
                m1_data['x'][:, 2][m5_data['j'] == j_unique[j]][0]
                for j in j_ind]) * beta_estimates.loc[2, 'mean_m1']
            model_fits['m1_f1_var'] = beta_estimates.loc[1, 'var_m1']
            model_fits['m1_f1_var'] += np.array([
                m1_data['x'][:, 2][m5_data['j'] == j_unique[j]][0]**2
                for j in j_ind]) * beta_estimates.loc[2, 'var_m1']
        if m == 2:
            model_fits['m2_f0_mean'] = (
                beta_estimates.loc[j_ind, 'mean_m2'].values)
            model_fits['m2_f0_mean'] += beta_estimates.loc[J_max, 'mean_m2']
            model_fits['m2_f0_var'] = (
                beta_estimates.loc[j_ind, 'var_m2'].values)
            model_fits['m2_f0_var'] += beta_estimates.loc[J_max, 'var_m2']
            model_fits['m2_f1_mean'] = (
                beta_estimates.loc[j_ind, 'mean_m2'].values)
            model_fits['m2_f1_mean'] += (
                beta_estimates.loc[J_max + 1, 'mean_m2'])
            model_fits['m2_f1_var'] = (
                beta_estimates.loc[j_ind, 'var_m2'].values)
            model_fits['m2_f1_var'] += beta_estimates.loc[J_max + 1, 'var_m2']
        if m == 3:
            m3_data_v1 = get_m3_vars(srrs_mn.copy(), False)
            x = pd.DataFrame(
                m3_data_v1['x'],
                columns=[x + '_f' + y for y in ['0', '1'] for x in j_unique])
            x_cols = x.loc[:, x.sum(axis=0) > 0].columns
            m3_f0_ind = [
                np.where(x_cols == (z + '_f0'))[0][0] for z in j_unique[j_ind]]
            m3_f1_ind = [
                np.where(x_cols == (z + '_f1'))[0][0] for z in j_unique[j_ind]]
            model_fits['m3_f0_mean'] = (
                beta_estimates.loc[m3_f0_ind, 'mean_m3'].values)
            model_fits['m3_f0_var'] = (
                beta_estimates.loc[m3_f0_ind, 'var_m3'].values)
            model_fits['m3_f1_mean'] = (
                beta_estimates.loc[m3_f1_ind, 'mean_m3'].values)
            model_fits['m3_f1_var'] = (
                beta_estimates.loc[m3_f1_ind, 'var_m3'].values)
        if m == 4:
            model_fits['m4_f0_mean'] = beta_estimates.loc[0, 'mean_m4']
            model_fits['m4_f0_mean'] += np.array([
                m4_data['x'][:, 2][m4_data['j'] == j_unique[j]][0]
                for j in j_ind]) * beta_estimates.loc[2, 'mean_m4']
            model_fits['m4_f0_mean'] += (
                eta_estimates.loc[j_ind, 'mean_m4'].values)
            model_fits['m4_f0_var'] = beta_estimates.loc[0, 'var_m4']
            model_fits['m4_f0_var'] += np.array([
                m4_data['x'][:, 2][m4_data['j'] == j_unique[j]][0]**2
                for j in j_ind]) * beta_estimates.loc[2, 'var_m4']
            model_fits['m4_f0_var'] += (
                eta_estimates.loc[j_ind, 'var_m4'].values)
            model_fits['m4_f1_mean'] = beta_estimates.loc[1, 'mean_m4']
            model_fits['m4_f1_mean'] += np.array([
                m4_data['x'][:, 2][m4_data['j'] == j_unique[j]][0]
                for j in j_ind]) * beta_estimates.loc[2, 'mean_m4']
            model_fits['m4_f1_mean'] += (
                eta_estimates.loc[j_ind, 'mean_m4'].values)
            model_fits['m4_f1_var'] = beta_estimates.loc[1, 'var_m4']
            model_fits['m4_f0_var'] += np.array([
                m4_data['x'][:, 2][m4_data['j'] == j_unique[j]][0]**2
                for j in j_ind]) * beta_estimates.loc[2, 'var_m4']
            model_fits['m4_f0_var'] += (
                eta_estimates.loc[j_ind, 'var_m4'].values)
        if m == 5:
            model_fits['m5_f0_mean'] = beta_estimates.loc[0, 'mean_m5']
            model_fits['m5_f0_mean'] += np.array([
                m5_data['x'][:, 2][m4_data['j'] == j_unique[j]][0]
                for j in j_ind]) * beta_estimates.loc[2, 'mean_m5']
            model_fits['m5_f0_mean'] += (
                eta_estimates.loc[j_ind, 'mean_m5-0'].values)
            model_fits['m5_f0_var'] = beta_estimates.loc[0, 'var_m5']
            model_fits['m5_f0_var'] += np.array([
                m5_data['x'][:, 2][m4_data['j'] == j_unique[j]][0]**2
                for j in j_ind]) * beta_estimates.loc[2, 'var_m5']
            model_fits['m5_f0_var'] += (
                eta_estimates.loc[j_ind, 'var_m5-0'].values)
            model_fits['m5_f1_mean'] = beta_estimates.loc[1, 'mean_m5']
            model_fits['m5_f1_mean'] += np.array([
                m5_data['x'][:, 2][m4_data['j'] == j_unique[j]][0]
                for j in j_ind]) * beta_estimates.loc[2, 'mean_m5']
            model_fits['m5_f1_mean'] += (
                eta_estimates.loc[j_ind, 'mean_m5-1'].values)
            model_fits['m5_f1_var'] = beta_estimates.loc[1, 'var_m5']
            model_fits['m5_f0_var'] += np.array([
                m5_data['x'][:, 2][m4_data['j'] == j_unique[j]][0]**2
                for j in j_ind]) * beta_estimates.loc[2, 'var_m5']
            model_fits['m5_f0_var'] += (
                eta_estimates.loc[j_ind, 'var_m5-1'].values)

    rowcols = [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3]]
    for m in model_list:
        fig_name = inputs.table_output_file_path
        fig_name += 'radon_m' + str(m) + '.pdf'
        with PdfPages(fig_name) as pdf:
            fig, ax = plt.subplots(
                nrows=2, ncols=4, sharex=True, sharey=True, figsize=[10, 7])
            for ii in range(len(rowcols)):
                row, col = rowcols[ii]
                j = j_unique[j_ind[ii]]
                ax[row, col].plot(
                    m5_data['x'][m5_data['j'] == j, 1],
                    m5_data['y'][m5_data['j'] == j],
                    'o', ms=5)
                ax[row, col].set_title(j)
                ax[row, col].set_xticks(
                    [0, 1], ['Basement', 'Ground floor'], rotation=90)
                y_line = [
                    model_fits.loc[j, 'm' + str(m) + '_f0_mean'],
                    model_fits.loc[j, 'm' + str(m) + '_f1_mean']]
                ax[row, col].plot([0, 1], y_line, lw=2, c='k')
                y_line = [
                    (model_fits.loc[j, 'm' + str(m) + '_f0_mean'] -
                        np.sqrt(model_fits.loc[j, 'm' + str(m) + '_f0_var'])),
                    (model_fits.loc[j, 'm' + str(m) + '_f1_mean'] -
                        np.sqrt(model_fits.loc[j, 'm' + str(m) + '_f0_var']))]
                ax[row, col].plot([0, 1], y_line, ls='--', c='k')
                y_line = [
                    (model_fits.loc[j, 'm' + str(m) + '_f0_mean'] +
                        np.sqrt(model_fits.loc[j, 'm' + str(m) + '_f0_var'])),
                    (model_fits.loc[j, 'm' + str(m) + '_f1_mean'] +
                        np.sqrt(model_fits.loc[j, 'm' + str(m) + '_f0_var']))]
                ax[row, col].plot([0, 1], y_line, ls='--', c='k')
            for ii in range(ax.shape[0]):
                ax[ii, 0].set_ylabel('Log radon level')
            plt.tight_layout()
            plt.subplots_adjust(wspace=0)
            pdf.savefig(fig)
            plt.close()
    return


###############################################################################
# Misc function: for output tables


def print_msd(x, dp=2):
    format_str = '%.' + str(dp) + 'f'
    output = format_str % x.mean()
    output += ' (' + format_str % x.std() + ')'
    return output


###############################################################################
# Define model data

def main():
    '''This is the main function, which runs all sampling on the data'''
    inputs = parse_args()
    np.random.seed(inputs.np_random_seed)
    srrs_mn = process_data()
    get_vars_list = [
        get_m0_vars, get_m1_vars, get_m2_vars,
        get_m3_vars, get_m4_vars, get_m5_vars]
    sample_model_list = [sample_lm_model] * 4
    sample_model_list += [sample_mm_model, sample_general_mm_model]
    aic_fit_list = [aic_lm_fit] * 4 + [aic_mm_fit, aic_general_mm_fit]

    sample_inputs = dict(
        n_draws=inputs.n_draws,
        random_seed=inputs.RANDOM_SEED,
        n_chains=inputs.n_chains,
        save_output=inputs.save_output,
        output_file_path=inputs.data_output_file_path)

    model_list = [int(x) for x in inputs.model_list.strip(' ').split(',')]

    # Set up SMC sampling results table
    smc_column_list = [
        type + '_m' + str(ii)
        for ii in model_list for type in ['integrated', 'full']]
    smc_lml = pd.DataFrame(
        columns=smc_column_list, index=range(inputs.n_chains))

    for m in model_list:
        if inputs.print_output:
            print('Starting sampling with model M' + str(m))
        get_vars = get_vars_list[m]
        sample_model = sample_model_list[m]
        m_data = get_vars(srrs_mn.copy())
        if inputs.save_output:
            d_param = m_data['x'].shape[1]
            n = m_data['x'].shape[0]
            df_columns = ['y'] + ['x' + str(x) for x in range(d_param)]
            if 'z' in m_data.keys():
                m_param = m_data['z'].shape[1]
                df_columns += ['z' + str(x) for x in range(m_param)]
            if 'j' in m_data.keys():
                df_columns += ['j']
            df = pd.DataFrame(index=range(n), columns=df_columns)
            df['y'] = m_data['y']
            if 'j' in m_data.keys():
                df['j'] = m_data['j']
            if 'z' in m_data.keys():
                df.loc[:, df.columns.str.startswith('z')] = m_data['z']
            df.loc[:, df.columns.str.startswith('x')] = m_data['x']
            df.to_csv(
                inputs.data_output_file_path + 'data' + str(m) + '.csv',
                index=False)
        m_str = str(m)
        trace_integrated, trace_full = sample_model(
            m_data, m_str, **sample_inputs)
        lml = trace_integrated['sample_stats']['log_marginal_likelihood']
        if trace_integrated['sample_stats']['chain'].size == 1:
            lml = [
                lml[:, ii].values[0][-1] for ii in range(inputs.n_chains)]
        else:
            lml = lml[:, -1].values
        smc_lml['integrated_m' + str(m)] = lml
        lml = trace_full['sample_stats']['log_marginal_likelihood']
        if trace_full['sample_stats']['chain'].size == 1:
            lml = [
                lml[:, ii].values[0][-1]
                for ii in range(inputs.n_chains)]
        else:
            lml = lml[:, -1].values
        smc_lml['full_m' + str(m)] = lml
        if inputs.save_output:
            smc_lml.to_csv(
                inputs.data_output_file_path + 'smc_likelihoods.csv',
                index=False)

    # Set up final results table
    smc_table_columns = ['Model', 'k', 'AIC', 'rank 1']
    smc_table_columns += ['Model evidence with integrated likelihood']
    smc_table_columns += ['rank 2']
    smc_table_columns += ['Model evidence with full likelihood', 'rank 3']
    smc_table = pd.DataFrame(
        columns=smc_table_columns, index=range(len(model_list)))
    smc_table['Model'] = ['M' + str(x) for x in model_list]
    for ii in range(len(model_list)):
        m = model_list[ii]
        get_vars = get_vars_list[m]
        aic_fit_fun = aic_fit_list[m]
        m_data = get_vars(srrs_mn.copy())
        aic_fit = aic_fit_fun(m_data)
        smc_table.loc[ii, 'k'] = np.floor(
            aic_fit.aic / 2 + aic_fit.llf).astype(int)
        smc_table.loc[ii, 'AIC'] = np.round(aic_fit.aic, 2)
        smc_table.loc[ii, 'Model evidence with integrated likelihood'] = (
            print_msd(smc_lml['integrated_m' + str(m)]))
        smc_table.loc[ii, 'Model evidence with full likelihood'] = print_msd(
            smc_lml['full_m' + str(m)])
    smc_table = smc_table.sort_values(by=['AIC'], ascending=True)
    smc_table['rank 1'] = np.arange(len(model_list)) + 1
    smc_table = smc_table.sort_values(
        by=['Model evidence with integrated likelihood'], ascending=True)
    smc_table['rank 2'] = np.arange(len(model_list)) + 1
    smc_table = smc_table.sort_values(
        by=['Model evidence with full likelihood'], ascending=True)
    smc_table['rank 3'] = np.arange(len(model_list)) + 1
    smc_table = smc_table.sort_index()

    if inputs.print_output:
        print(smc_table)
    if inputs.save_output:
        smc_table.to_csv(
            inputs.table_output_file_path + 'tableIII.csv', index=False)

    if inputs.generate_figures:
        model_fits_and_plot()
    return


if __name__ == '__main__':
    main()
