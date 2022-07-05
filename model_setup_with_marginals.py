#!/usr/bin/env python
'''
Author: Tom Edinburgh
v1: date 04/07/2022.

This script defines the all functions needed for computation of the marginal
likelihoods involved in Bayesian model selection with multilevel linear models.

It also contains functions for PyMC3 model formulation for running an MCMC
sampling algorithm, either using the full likelihood or marginal likelihood,
in this model selection problem. A general problem can be set up as follows:

    m_data = dict(x=..., y=..., j=...)
    with pm.Model() as m:
        logp_input_dict = get_mm_marginal(m, m_data)
        pm.Potential('y', logp_mm(**logp_input_dict))
        ...

Further examples of usage are given in the accompanying script
`evaluate_models_radon.py`.
'''
import numpy as np
import pandas as pd
import pymc3 as pm
from theano import tensor as tt

###############################################################################
# Functions: marginal likelihoods


def logp_lm(sigma2, mu, Sigma_inv, SxxT, Sxy, Syy, n):
    '''
    Log marginal for the linear model, given pre-defined functions of the data
    '''
    Sigma_tilde_inv = Sigma_inv + 1 / sigma2 * SxxT
    Sigma_tilde = tt.nlinalg.matrix_inverse(Sigma_tilde_inv)
    mu_tilde = tt.dot(Sigma_tilde, tt.dot(Sigma_inv, mu))
    mu_tilde += 1 / sigma2 * tt.dot(Sigma_tilde, Sxy)
    logp_ = tt.log(tt.nlinalg.det(Sigma_tilde))
    logp_ += tt.log(np.linalg.det(Sigma_inv))
    logp_ += -n * tt.log(2 * np.pi * sigma2)
    logp_ += tt.dot(mu_tilde, tt.dot(Sigma_tilde_inv, mu_tilde.T))
    logp_ += -tt.dot(mu, tt.dot(Sigma_inv, mu.T))
    logp_ += -1 / sigma2 * Syy
    logp_ = 0.5 * logp_
    return logp_


def logp_mm(
        sigma2_y, sigma2_eta, mu, Sigma_inv,
        SxSxT_j, SxSy_j, SySy_j, SxxT, Sxy, Syy, n_j, J):
    frac_j = sigma2_eta / (sigma2_y + n_j * sigma2_eta)
    Sigma_hat_inv = Sigma_inv + 1 / sigma2_y * SxxT
    Sigma_hat_inv += (
        -1 / sigma2_y * (frac_j.reshape([-1, 1, 1]) * SxSxT_j).sum(axis=0))
    Sigma_hat_inv_mu_hat = (
        tt.dot(Sigma_inv, mu) +
        1 / sigma2_y * (Sxy - (frac_j.reshape([-1, 1]) * SxSy_j).sum(axis=0)))
    mu_hat = tt.dot(
        tt.nlinalg.matrix_inverse(Sigma_hat_inv), Sigma_hat_inv_mu_hat)
    logp_ = -tt.log(tt.nlinalg.det(Sigma_hat_inv))
    logp_ += tt.log(np.linalg.det(Sigma_inv))
    logp_ += -tt.sum(n_j) * tt.log(2 * np.pi * sigma2_y)
    logp_ += tt.sum(tt.log(sigma2_y / (sigma2_y + n_j * sigma2_eta)))
    logp_ += tt.dot(mu_hat.T, Sigma_hat_inv_mu_hat)
    logp_ += -tt.dot(mu.T, tt.dot(Sigma_inv, mu))
    logp_ += -1 / sigma2_y * Syy
    logp_ += 1 / sigma2_y * (frac_j * SySy_j).sum(axis=0)
    logp_ = 0.5 * logp_
    return logp_


def logp_general_mm(
        sigma2_y, Sigma_eta_inv, mu, Sigma_inv,
        SzzT_j, SzxT_j, Szy_j, SxxT, Sxy, Syy, n_j, J):
    Sigma_hat_eta_j_inv = [
        Sigma_eta_inv + 1 / sigma2_y * SzzT_j[jj] for jj in range(J)]
    Sigma_hat_eta_j = [
        tt.nlinalg.matrix_inverse(S) for S in Sigma_hat_eta_j_inv]
    Sigma_hat_inv = Sigma_inv + 1 / sigma2_y * SxxT
    Sigma_hat_inv += (
        -1 / sigma2_y**2 * sum([
            tt.dot(SzxT_j[jj].T, tt.dot(Sigma_hat_eta_j[jj], SzxT_j[jj]))
            for jj in range(J)]))
    Sigma_hat_inv_mu_hat = (
        tt.dot(Sigma_inv, mu) +
        1 / sigma2_y * Sxy - 1 / sigma2_y**2 * sum([
            tt.dot(SzxT_j[jj].T, tt.dot(Sigma_hat_eta_j[jj], Szy_j[jj]))
            for jj in range(J)]))
    mu_hat = tt.dot(
        tt.nlinalg.matrix_inverse(Sigma_hat_inv), Sigma_hat_inv_mu_hat)
    logp_ = -tt.log(tt.nlinalg.det(Sigma_hat_inv))
    logp_ += tt.log(np.linalg.det(Sigma_inv))
    logp_ += -sum(n_j) * tt.log(2 * np.pi * sigma2_y)
    logp_ += J * tt.log(tt.nlinalg.det(Sigma_eta_inv))
    logp_ += -sum([
        tt.log(tt.nlinalg.det(Sigma_hat_eta_j_inv[jj])) for jj in range(J)])
    logp_ += tt.dot(mu_hat.T, Sigma_hat_inv_mu_hat)
    logp_ += -tt.dot(mu.T, tt.dot(Sigma_inv, mu))
    logp_ += -1 / sigma2_y * Syy
    logp_ += 1 / sigma2_y**2 * sum([
        tt.dot(Szy_j[jj].T, tt.dot(Sigma_hat_eta_j[jj], Szy_j[jj]))
        for jj in range(J)])
    logp_ = 0.5 * logp_
    return logp_


def xy_sums(x, y):
    '''
    Generate fixed products and sums of data used in the marginal likelihood
    for the linear model
    '''
    n = x.shape[0]
    SxxT = sum(
        [x[ii, :].reshape(-1, 1) @ x[ii, :].reshape(1, -1) for ii in range(n)])
    Sxy = np.sum(x * y.reshape(-1, 1), axis=0)
    Syy = np.sum(y**2, axis=0)
    output = dict()
    output['SxxT'] = SxxT
    output['Sxy'] = Sxy
    output['Syy'] = Syy
    output['n'] = n
    return output


def xy_j_sums(x, y, j):
    '''
    Generate fixed products and sums of data used in the marginal likelihood
    for the multilevel model
    '''
    J = np.unique(j).size
    n = x.shape[0]
    x_ij = [x[j == j_val, :] for j_val in np.unique(j)]
    y_ij = [y[j == j_val] for j_val in np.unique(j)]
    n_j = np.array([x.shape[0] for x in x_ij])
    # \Sum_i x_{ij}
    Sx_j = np.array([
        sum([x_ij[jj][ii, :] for ii in range(n_j[jj])]) for jj in range(J)])
    # \Sum_i y_{ij}
    Sy_j = np.array([np.sum(y_ij[jj], axis=0) for jj in range(J)])
    # (\Sum_i x_{ij})(\Sum_k x_{kj}^T)
    SxSxT_j = np.array([
        Sx_j[jj].reshape(-1, 1) @ Sx_j[jj].reshape(1, -1) for jj in range(J)])
    # (\Sum_i x_{ij})(\Sum_k y_{kj})
    SxSy_j = np.array([Sx_j[jj] * Sy_j[jj] for jj in range(J)])
    # (\Sum_i y_{ij})(\Sum_k y_{kj})
    SySy_j = np.array([Sy_j[jj]**2 for jj in range(J)])
    # \Sum_{i,j} x_{ij} x_{ij}^T
    SxxT = np.array([
        x[ii, :].reshape(-1, 1) @ x[ii, :].reshape(1, -1)
        for ii in range(n)]).sum(axis=0)
    # \Sum_{i,j} y_{ij} x_{ij}
    Sxy = (x * y.reshape(-1, 1)).sum(axis=0)
    # \Sum_{i,j} y_{ij}^2
    Syy = (y**2).sum(axis=0)
    output = dict()
    output['SxSxT_j'] = SxSxT_j
    output['SxSy_j'] = SxSy_j
    output['SySy_j'] = SySy_j
    output['SxxT'] = SxxT
    output['Sxy'] = Sxy
    output['Syy'] = Syy
    output['n_j'] = n_j
    output['J'] = J
    return output


def xyz_j_sums(x, y, z, j):
    '''
    Generate fixed products and sums of data used in the marginal likelihood
    for the general multilevel model
    '''
    J = np.unique(j).size
    n = x.shape[0]
    x_ij = [x[j == j_val, :] for j_val in np.unique(j)]
    z_ij = [z[j == j_val, :] for j_val in np.unique(j)]
    y_ij = [y[j == j_val] for j_val in np.unique(j)]
    n_j = np.array([x.shape[0] for x in x_ij])
    # \Sum_i z_{ij} z_{ij}^T
    SzzT_j = np.array([
        sum([z_ij[jj][ii, :].reshape(-1, 1) @ z_ij[jj][ii, :].reshape(1, -1)
            for ii in range(n_j[jj])]) for jj in range(J)])
    # \Sum_i z_{ij} x_{ij}^T
    SzxT_j = np.array([
        sum([z_ij[jj][ii, :].reshape(-1, 1) @ x_ij[jj][ii, :].reshape(1, -1)
            for ii in range(n_j[jj])]) for jj in range(J)])
    # \Sum_i z_{ij} y_{ij}
    Szy_j = np.array([
        sum([z_ij[jj][ii, :] * y_ij[jj][ii]
            for ii in range(n_j[jj])]) for jj in range(J)])
    # \Sum_{i,j} x_{ij} x_{ij}^T
    SxxT = np.array([
        x[ii, :].reshape(-1, 1) @ x[ii, :].reshape(1, -1)
        for ii in range(n)]).sum(axis=0)
    # \Sum_{i,j} y_{ij} x_{ij}
    Sxy = (x * y.reshape(-1, 1)).sum(axis=0)
    # \Sum_{i,j} y_{ij}^2
    Syy = (y**2).sum(axis=0)
    output = dict()
    output['SzzT_j'] = SzzT_j
    output['SzxT_j'] = SzxT_j
    output['Szy_j'] = Szy_j
    output['SxxT'] = SxxT
    output['Sxy'] = Sxy
    output['Syy'] = Syy
    output['n_j'] = n_j
    output['J'] = J
    return output


###############################################################################
# Functions: model set up


def get_lm_marginal(
        pm_model, m_data, model_str='',
        sigma2_alpha=3, sigma2_beta=1, beta_mu=None, beta_Sigma_inv=None):
    '''Define the PyMC3 potential for a linear model, for MCMC sampling'''
    xy_dict = xy_sums(**m_data)
    d = m_data['x'].shape[1]
    if beta_mu is None:
        beta_mu = np.zeros(d)
    if beta_Sigma_inv is None:
        beta_Sigma_inv = np.eye(d)
    with pm_model:
        sigma2 = pm.InverseGamma(
            'sigma2' + model_str, alpha=sigma2_alpha, beta=sigma2_beta)
    logp_input_dict = dict(
        sigma2=sigma2, mu=beta_mu,
        Sigma_inv=beta_Sigma_inv, **xy_dict)
    return logp_input_dict


def get_mm_marginal(
        pm_model, m_data, model_str='',
        sigma2_y_alpha=3, sigma2_y_beta=1,
        sigma2_eta_alpha=3, sigma2_eta_beta=1,
        beta_mu=None, beta_Sigma_inv=None):
    '''
    Define the PyMC3 potential for a simple multilevel linear model (with
    group-varying intercept only), for MCMC sampling
    '''
    xy_dict = xy_j_sums(**m_data)
    d = m_data['x'].shape[1]
    if beta_mu is None:
        beta_mu = np.zeros(d)
    if beta_Sigma_inv is None:
        beta_Sigma_inv = np.eye(d)
    with pm_model:
        sigma2_y = pm.InverseGamma(
            'sigma2_y' + model_str, alpha=sigma2_y_alpha, beta=sigma2_y_beta)
        sigma2_eta = pm.InverseGamma(
            'sigma2_eta' + model_str,
            alpha=sigma2_eta_alpha, beta=sigma2_eta_beta)
    logp_input_dict = dict(
        sigma2_y=sigma2_y, sigma2_eta=sigma2_eta,
        mu=beta_mu, Sigma_inv=beta_Sigma_inv, **xy_dict)
    return logp_input_dict


def get_general_mm_marginal(
        pm_model, m_data, model_str='',
        sigma2_y_alpha=3, sigma2_y_beta=1,
        sigma2_eta_alpha=3, sigma2_eta_beta=1,
        beta_mu=None, beta_Sigma_inv=None,
        include_rho=False, rho_eta_mu=0, rho_eta_var=1):
    '''
    Define the PyMC3 potential for a general multilevel linear model (with
    multiple group-varying covariates), for MCMC sampling
    '''
    xy_dict = xyz_j_sums(**m_data)
    d = m_data['x'].shape[1]
    m = m_data['z'].shape[1]
    rho_eta_shape = int(m * (m - 1) / 2)
    if beta_mu is None:
        beta_mu = np.zeros(d)
    if beta_Sigma_inv is None:
        beta_Sigma_inv = np.eye(d)
    if include_rho:
        triu_index = np.zeros((m, m), dtype=int)
        triu_index[np.triu_indices(m, 1)] = np.arange(rho_eta_shape)
        triu_index[np.triu_indices(m, 1)[::-1]] = np.arange(rho_eta_shape)
    with pm_model:
        sigma2_y = pm.InverseGamma(
            'sigma2_y' + model_str, alpha=sigma2_y_alpha, beta=sigma2_y_beta)
        sigma2_eta = pm.InverseGamma(
            'sigma2_eta' + model_str,
            alpha=sigma2_eta_alpha, beta=sigma2_eta_beta, shape=m)
        if include_rho:
            # Include an off-diagonal term in the random effect covariance
            rho_eta = pm.TruncatedNormal(
                'rho_eta' + model_str, mu=rho_eta_mu, tau=1/rho_eta_var,
                lower=-1, upper=1, shape=rho_eta_shape)
            corr_matrix = tt.diag(np.repeat(1 - rho_eta[0], m))
            corr_matrix += rho_eta[triu_index]
            Sigma_eta_inv = pm.Deterministic(
                'Sigma_eta_inv' + model_str,
                tt.nlinalg.matrix_inverse(tt.nlinalg.matrix_dot(
                    tt.diag(sigma2_eta), corr_matrix, tt.diag(sigma2_eta))))
        else:
            Sigma_eta_inv = pm.Deterministic(
                'Sigma_eta_inv' + model_str, tt.nlinalg.diag(1 / sigma2_eta))
    logp_input_dict = dict(
        sigma2_y=sigma2_y, Sigma_eta_inv=Sigma_eta_inv,
        mu=beta_mu, Sigma_inv=beta_Sigma_inv, **xy_dict)
    return logp_input_dict


def get_lm_full(
        pm_model, m_data, model_str='',
        sigma2_alpha=3, sigma2_beta=1, beta_mu=0, beta_var=1):
    '''Define the PyMC3 potential for a linear model, for MCMC sampling'''
    d = m_data['x'].shape[1]
    with pm_model:
        beta = pm.Normal(
            'beta' + model_str, mu=beta_mu, tau=1/beta_var, shape=d)
        sigma2 = pm.InverseGamma(
            'sigma2' + model_str, alpha=sigma2_alpha, beta=sigma2_beta)
        y = tt.dot(m_data['x'], beta)
    return y, sigma2


def get_mm_full(
        pm_model, m_data, model_str='',
        sigma2_y_alpha=3, sigma2_y_beta=1,
        sigma2_eta_alpha=3, sigma2_eta_beta=1,
        beta_mu=0, beta_var=1):
    '''Define the PyMC3 potential for a linear model, for MCMC sampling'''
    d = m_data['x'].shape[1]
    unique_j = np.unique(m_data['j'])
    J = unique_j.size
    j = pd.Series(m_data['j']).replace(dict(zip(unique_j, range(J)))).values
    with pm_model:
        beta = pm.Normal(
            'beta' + model_str, mu=beta_mu, tau=1/beta_var, shape=d)
        sigma2_y = pm.InverseGamma(
            'sigma2_y' + model_str, alpha=sigma2_y_alpha, beta=sigma2_y_beta)
        sigma2_eta = pm.InverseGamma(
            'sigma2_eta' + model_str,
            alpha=sigma2_eta_alpha, beta=sigma2_eta_beta)
        eta = pm.Normal('eta' + model_str, mu=0, tau=1/sigma2_eta, shape=J)
        y = tt.dot(m_data['x'], beta) + eta[j]
    return y, sigma2_y


def get_general_mm_full(
        pm_model, m_data, model_str='',
        sigma2_y_alpha=3, sigma2_y_beta=1,
        sigma2_eta_alpha=3, sigma2_eta_beta=1,
        beta_mu=0, beta_var=1,
        include_rho=False, rho_eta_mu=0, rho_eta_var=1):
    '''
    Define the PyMC3 potential for a general multilevel linear model (with
    multiple group-varying covariates), for MCMC sampling
    '''
    d = m_data['x'].shape[1]
    m = m_data['z'].shape[1]
    unique_j = np.unique(m_data['j'])
    J = unique_j.size
    j = pd.Series(m_data['j']).replace(dict(zip(unique_j, range(J)))).values
    rho_eta_shape = int(m * (m - 1) / 2)
    if include_rho:
        triu_index = np.zeros((m, m), dtype=int)
        triu_index[np.triu_indices(m, 1)] = np.arange(rho_eta_shape)
        triu_index[np.triu_indices(m, 1)[::-1]] = np.arange(rho_eta_shape)
    with pm_model:
        beta = pm.Normal(
            'beta' + model_str, mu=beta_mu, tau=1/beta_var, shape=d)
        sigma2_y = pm.InverseGamma(
            'sigma2_y' + model_str, alpha=sigma2_y_alpha, beta=sigma2_y_beta)
        sigma2_eta = pm.InverseGamma(
            'sigma2_eta' + model_str,
            alpha=sigma2_eta_alpha, beta=sigma2_eta_beta, shape=m)
        if include_rho:
            # Include an off-diagonal term in the random effect covariance
            rho_eta = pm.TruncatedNormal(
                'rho_eta' + model_str, mu=rho_eta_mu, tau=1/rho_eta_var,
                lower=-1, upper=1, shape=rho_eta_shape)
            corr_matrix = tt.diag(np.repeat(1 - rho_eta[0], m))
            corr_matrix += rho_eta[triu_index]
            Sigma_eta_inv = tt.nlinalg.matrix_inverse(tt.nlinalg.matrix_dot(
                    tt.diag(sigma2_eta), corr_matrix, tt.diag(sigma2_eta)))
        else:
            Sigma_eta_inv = tt.nlinalg.diag(1 / sigma2_eta)
        eta = pm.MvNormal(
            'eta' + model_str, mu=0, tau=Sigma_eta_inv, shape=(J, m))
        y = tt.dot(m_data['x'], beta)
        y += tt.sum(m_data['z'] * eta[j, :], axis=1)
    return y, sigma2_y
