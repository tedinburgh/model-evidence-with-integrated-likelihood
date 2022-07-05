#!/usr/bin/env python
'''
Author: Tom Edinburgh
v1: date 04/07/2022.

This script defines the all functions needed for computation of the marginal
likelihoods involved in Bayesian model selection with multilevel linear models.
'''
import numpy as np
import pandas as pd
import pymc3 as pm
import argparse
import statsmodels.api as sm
import model_setup_with_marginals as msm

###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description='''
            Evaluate multilevel models using SMC with marginal and full
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
        '--output_file_path',
        default='./',
        help='''File path to the directory that saves outputs from this script
            ''',
        type=str)
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
    x = x * (1 - df['floor'].values).reshape(-1, 1)
    x = np.hstack([x, df['floor'].values.reshape(-1, 1)])
    output = dict(y=y, x=x)
    return output


def get_m3_vars(df):
    '''No pooling (no group-level covariates), linear model'''
    y = df['log_radon'].values
    y = (y - np.mean(y)) / np.std(y)
    x = np.array([(df['county'] == c).values for c in df['county'].unique()]).T
    x = np.hstack([
        x * (1 - df['floor'].values).reshape(-1, 1),
        x * df['floor'].values.reshape(-1, 1)])
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
        m_data, m_number, n_draws, random_seed, n_chains,
        save_output, output_file_path):
    '''
    Sample from the marginal and full likelihoods respectively for linear model
    '''
    # Marginal likelihood sampling
    with pm.Model() as smc_m:
        logp_input_dict = msm.get_lm_marginal(smc_m, m_data)
        pm.Potential('y', msm.logp_lm(**logp_input_dict))
        trace_marginal = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        if save_output:
            dir = output_file_path + 'marginal_m' + str(m_number)
            pm.save_trace(trace_marginal, directory=dir, overwrite=True)
    # Full likelihood sampling
    with pm.Model() as smc_m_full:
        y, sigma2 = msm.get_lm_full(smc_m_full, m_data)
        pm.Normal('y_obs', mu=y, tau=1/sigma2, observed=m_data['y'])
        trace_full = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        if save_output:
            dir = output_file_path + 'full_m' + str(m_number)
            pm.save_trace(trace_full, directory=dir, overwrite=True)
    return trace_marginal, trace_full


def sample_mm_model(
        m_data, m_number, n_draws, random_seed, n_chains,
        save_output, output_file_path):
    '''
    Sample from the marginal and full likelihoods respectively for simple
    multilevel linear model
    '''
    # Marginal likelihood sampling
    with pm.Model() as smc_m:
        logp_input_dict = msm.get_mm_marginal(smc_m, m_data)
        pm.Potential('y', msm.logp_mm(**logp_input_dict))
        trace_marginal = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        if save_output:
            dir = output_file_path + 'marginal_m' + str(m_number)
            pm.save_trace(trace_marginal, directory=dir, overwrite=True)
    # Full likelihood sampling
    with pm.Model() as smc_m_full:
        y, sigma2 = msm.get_mm_full(smc_m_full, m_data)
        pm.Normal('y_obs', mu=y, tau=1/sigma2, observed=m_data['y'])
        trace_full = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        if save_output:
            dir = output_file_path + 'full_m' + str(m_number)
            pm.save_trace(trace_full, directory=dir, overwrite=True)
    return trace_marginal, trace_full


def sample_general_mm_model(
        m_data, m_number, n_draws, random_seed, n_chains,
        save_output, output_file_path):
    '''
    Sample from the marginal and full likelihoods respectively for more
    general multilevel linear model
    '''
    # Marginal likelihood sampling
    with pm.Model() as smc_m:
        logp_input_dict = msm.get_general_mm_marginal(smc_m, m_data)
        pm.Potential('y', msm.logp_general_mm(**logp_input_dict))
        trace_marginal = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        if save_output:
            dir = output_file_path + 'marginal_m' + str(m_number)
            pm.save_trace(trace_marginal, directory=dir, overwrite=True)
    # Full likelihood sampling
    with pm.Model() as smc_m_full:
        y, sigma2 = msm.get_general_mm_full(
            smc_m_full, m_data, include_rho=True)
        pm.Normal('y_obs', mu=y, tau=1/sigma2, observed=m_data['y'])
        trace_full = pm.sample_smc(
            draws=n_draws, random_seed=random_seed, chains=n_chains)
        if save_output:
            dir = output_file_path + 'full_m' + str(m_number)
            pm.save_trace(trace_full, directory=dir, overwrite=True)
    return trace_marginal, trace_full


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
    get_vars_all_list = [
        get_m0_vars, get_m1_vars, get_m2_vars,
        get_m3_vars, get_m4_vars, get_m5_vars]
    sample_model_all_list = [sample_lm_model] * 4
    sample_model_all_list += [sample_mm_model, sample_general_mm_model]
    aic_fit_all_list = [aic_lm_fit] * 4 + [aic_mm_fit, aic_general_mm_fit]

    sample_inputs = dict(
        n_draws=inputs.n_draws,
        random_seed=inputs.RANDOM_SEED,
        n_chains=inputs.n_chains,
        save_output=inputs.save_output,
        output_file_path=inputs.output_file_path)

    model_list = [int(x) for x in inputs.model_list.strip(' ').split(',')]

    # Set up SMC sampling results table
    smc_column_list = [
        type + '_m' + str(ii)
        for ii in model_list for type in ['marginal', 'full']]
    smc_lml = pd.DataFrame(
        columns=smc_column_list, index=range(inputs.n_chains))

    for m in model_list:
        if inputs.print_output:
            print('Starting sampling with model M' + str(m))
        get_vars = get_vars_all_list[m]
        sample_model = sample_model_all_list[m]
        m_data = get_vars(srrs_mn.copy())
        trace_marginal, trace_full = sample_model(m_data, m, **sample_inputs)
        smc_lml['marginal_m' + str(m)] = (
            trace_marginal.report.log_marginal_likelihood)
        smc_lml['full_m' + str(m)] = trace_full.report.log_marginal_likelihood

    if inputs.save_output:
        smc_lml.to_csv(
            inputs.output_file_path + 'smc_likelihoods.csv', index=False)

    # Set up final results table
    tableI_cols = ['Model', 'k', 'AIC']
    tableI_cols += ['Model evidence with marginal likelihood']
    tableI_cols += ['Model evidence with full likelihood']
    tableI = pd.DataFrame(
        columns=tableI_cols, index=range(len(model_list)))
    tableI['Model'] = ['M' + str(x) for x in model_list]
    for ii in range(len(model_list)):
        m = model_list[ii]
        get_vars = get_vars_all_list[m]
        aic_fit_fun = aic_fit_all_list[m]
        m_data = get_vars(srrs_mn.copy())
        aic_fit = aic_fit_fun(m_data)
        tableI.loc[ii, 'k'] = np.floor(
            aic_fit.aic / 2 + aic_fit.llf).astype(int)
        tableI.loc[ii, 'AIC'] = aic_fit.aic
        tableI.loc[ii, 'Model evidence with marginal likelihood'] = print_msd(
            smc_lml['marginal_m' + str(m)])
        tableI.loc[ii, 'Model evidence with full likelihood'] = print_msd(
            smc_lml['full_m' + str(m)])

    if inputs.print_output:
        print(tableI)
    if inputs.save_output:
        tableI.to_csv(inputs.output_file_path + 'tableI.csv', index=False)
    return


if __name__ == '__main__':
    main()
