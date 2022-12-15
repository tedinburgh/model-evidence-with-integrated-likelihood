#!/usr/bin/env python
'''
Author: Tom Edinburgh
v2: date 20/10/2022.

This script defines the all functions needed for computation of the integrated
likelihoods involved in Bayesian model selection with multilevel linear models.

It also contains functions for PyMC3 model formulation for running an MCMC
sampling algorithm, either using the full likelihood or integrated likelihood,
in this model selection problem. A general problem can be set up as follows:

    m_data = dict(x=..., y=..., j=...)
    with pm.Model() as m:
        logp_input_dict = get_mm_integrated(m, m_data)
        pm.Potential('y', logp_mm(**logp_input_dict))
        ...

Further examples of usage are given in the accompanying script
`evaluate_models_radon.py`.
'''

import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import loggamma

###############################################################################
# Functions: integrated likelihoods


def logp_lm(sigma2, mu, Sigma_inv, SxxT, Sxy, Syy, n):
    '''
    Log integrated for the linear model,
    given pre-defined functions of the data
    '''
    Sigma_tilde_inv = Sigma_inv + 1 / sigma2 * SxxT
    Sigma_tilde = pm.math.matrix_inverse(Sigma_tilde_inv)
    mu_tilde = pm.math.dot(Sigma_tilde, np.dot(Sigma_inv, mu))
    mu_tilde += 1 / sigma2 * pm.math.dot(Sigma_tilde, Sxy)
    logp_ = pm.math.log(pm.math.det(Sigma_tilde_inv))
    logp_ += -np.log(np.linalg.det(Sigma_inv))
    logp_ += n * pm.math.log(2 * np.pi * sigma2)
    logp_ += -pm.math.dot(mu_tilde, pm.math.dot(Sigma_tilde_inv, mu_tilde))
    logp_ += np.dot(mu, np.dot(Sigma_inv, mu))
    logp_ += 1 / sigma2 * Syy
    logp_ = -0.5 * logp_
    return logp_


def mu_tilde_lm(sigma2, beta_mu, beta_Sigma_inv, SxxT, Sxy, Syy, n):
    '''
    Mu tilde for the linear model, given pre-defined functions of the data
    '''
    Sigma_tilde_inv = beta_Sigma_inv + 1 / sigma2 * SxxT
    Sigma_tilde = np.linalg.inv(Sigma_tilde_inv)
    mu_tilde = np.dot(Sigma_tilde, np.dot(beta_Sigma_inv, beta_mu))
    mu_tilde += 1 / sigma2 * np.dot(Sigma_tilde, Sxy)
    output = dict(mu_tilde=mu_tilde, Sigma_tilde_inv=Sigma_tilde_inv)
    return output


def logp_ninvg_lm(sigma2, mu, Sigma_inv, SxxT, Sxy, Syy, n, gamma):
    '''
    Log integrated for the linear model,
    given pre-defined functions of the data
    '''
    Sigma_tilde_inv = 1 / gamma * Sigma_inv + SxxT
    Sigma_tilde = pm.math.matrix_inverse(Sigma_tilde_inv)
    mu_tilde = pm.math.dot(Sigma_tilde, np.dot(Sigma_inv, mu))
    mu_tilde += pm.math.dot(Sigma_tilde, Sxy)
    logp_ = pm.math.log(pm.math.det(Sigma_tilde_inv))
    logp_ += -np.log(np.linalg.det(1 / gamma * Sigma_inv))
    logp_ += n * pm.math.log(2 * np.pi * sigma2)
    logp_ += -1 / sigma2 * pm.math.dot(
        mu_tilde, pm.math.dot(Sigma_tilde_inv, mu_tilde))
    logp_ += 1 / sigma2 * np.dot(mu, np.dot(Sigma_inv, mu))
    logp_ += 1 / sigma2 * Syy
    logp_ = -0.5 * logp_
    return logp_


def mu_tilde_ninvg_lm(
        sigma2, beta_mu, beta_Sigma_inv, SxxT, Sxy, Syy, n, gamma):
    '''
    Mu tilde for the linear model, given pre-defined functions of the data
    '''
    Sigma_tilde_inv = 1 / gamma * beta_Sigma_inv + SxxT
    Sigma_tilde = np.linalg.inv(Sigma_tilde_inv)
    mu_tilde = np.dot(Sigma_tilde, np.dot(beta_Sigma_inv, beta_mu))
    mu_tilde += np.dot(Sigma_tilde, Sxy)
    output = dict(mu_tilde=mu_tilde, Sigma_tilde_inv=Sigma_tilde_inv)
    return output


def logp_evidence_ninvg_lm(
        SxxT, Sxy, Syy, n, d, Sigma_inv,
        sigma2_alpha=3, sigma2_beta=2, gamma=1):
    '''
    Log integrated for the linear model with prior NIG(a, b, 0, Sigma_inv),
    given pre-defined functions of the data.
    '''
    Sigma_tilde_inv = 1 / gamma * Sigma_inv + SxxT
    mu_tilde = np.linalg.solve(Sigma_tilde_inv, Sxy)
    beta = sigma2_beta + 0.5*(Syy - np.dot(Sxy, mu_tilde))
    logp_ = np.log(np.linalg.det(Sigma_tilde_inv))
    logp_ += -np.log(np.linalg.det(1 / gamma * Sigma_inv))
    logp_ += n * np.log(2*np.pi)
    logp_ += (2*sigma2_alpha + n) * np.log(beta)
    logp_ += -2*sigma2_alpha * np.log(sigma2_beta)
    logp_ += -2 * loggamma(sigma2_alpha + 0.5*n)
    logp_ += 2 * loggamma(sigma2_alpha)
    logp_ = -0.5 * logp_
    return logp_


def logp_mm(
        sigma2_y, sigma2_eta, mu, Sigma_inv,
        SxSxT_j, SxSy_j, SySy_j, SxxT, Sxy, Syy, n_j, J):
    frac_eta = sigma2_eta / (sigma2_y + n_j * sigma2_eta)
    Sigma_hat_inv = Sigma_inv + 1 / sigma2_y * SxxT
    Sigma_hat_inv += (
        -1 / sigma2_y * (frac_eta.reshape([-1, 1, 1]) * SxSxT_j).sum(axis=0))
    Sigma_hat_inv_mu_hat = (
        np.dot(Sigma_inv, mu) + 1 / sigma2_y * (
            Sxy - (frac_eta.reshape([-1, 1]) * SxSy_j).sum(axis=0)))
    mu_hat = pm.math.dot(
        pm.math.matrix_inverse(Sigma_hat_inv), Sigma_hat_inv_mu_hat)
    logp_ = -pm.math.log(pm.math.det(Sigma_hat_inv))
    logp_ += np.log(np.linalg.det(Sigma_inv))
    logp_ += -np.sum(n_j) * pm.math.log(2 * np.pi * sigma2_y)
    logp_ += pm.math.sum(pm.math.log(sigma2_y / (sigma2_y + n_j * sigma2_eta)))
    logp_ += pm.math.dot(mu_hat, Sigma_hat_inv_mu_hat)
    logp_ += -np.dot(mu, np.dot(Sigma_inv, mu))
    logp_ += -1 / sigma2_y * Syy
    logp_ += 1 / sigma2_y * (frac_eta * SySy_j).sum(axis=0)
    logp_ = 0.5 * logp_
    return logp_


def mu_hat_mm(
        sigma2_y, sigma2_eta, beta_mu, beta_Sigma_inv,
        SxSxT_j, SxSy_j, SySy_j, SxxT, Sxy, Syy, Syy_j, Sxy_j, SxxT_j, n_j, J):
    frac_eta = sigma2_eta / (sigma2_y + n_j * sigma2_eta)
    Sigma_hat_inv = beta_Sigma_inv + 1 / sigma2_y * SxxT
    Sigma_hat_inv += (
        -1 / sigma2_y * (frac_eta.reshape([-1, 1, 1]) * SxSxT_j).sum(axis=0))
    Sigma_hat_inv_mu_hat = (
        np.dot(beta_Sigma_inv, beta_mu) + 1 / sigma2_y * (
            Sxy - (frac_eta.reshape([-1, 1]) * SxSy_j).sum(axis=0)))
    mu_hat = np.dot(np.linalg.inv(Sigma_hat_inv), Sigma_hat_inv_mu_hat)
    eta_hat = frac_eta * (
        Syy_j - 2*np.dot(Sxy_j, mu_hat) +
        np.dot(np.dot(SxxT_j, mu_hat), mu_hat))
    sigma2_eta_hat = sigma2_y * frac_eta
    output = dict(
        mu_tilde=mu_hat, Sigma_tilde_inv=Sigma_hat_inv,
        eta_hat=eta_hat, sigma2_hat_eta=sigma2_eta_hat)
    return output


def logp_hierarchical_mm(
        sigma2_y, sigma2_eta, sigma2_zeta, mu, Sigma_inv, SxSxT_jk,
        SxSy_jk, SySy_jk, SxxT, Sxy, Syy, Sx_jk, Sy_jk, n_k, m_jk, m_k, K):
    frac_eta = [
        [sigma2_eta / (sigma2_y + m_jk[kk][jj] * sigma2_eta)
            for jj in range(n_k[kk])]
        for kk in range(K)]
    frac_eta_zeta = [
        sigma2_zeta / (sigma2_y + sigma2_zeta * sum([
            (sigma2_y * m_jk[kk][jj]) / (sigma2_y + sigma2_eta * m_jk[kk][jj])
            for jj in range(n_k[kk])]))
        for kk in range(K)]
    # frac_zeta = [
    #     sigma2_zeta / (sigma2_y + m_k[kk] * sigma2_zeta) for kk in range(K)]
    Sigma_hat_inv = Sigma_inv + 1 / sigma2_y * SxxT
    Sigma_hat_inv += -1 / sigma2_y * sum([sum([
            frac_eta[kk][jj] * SxSxT_jk[kk][jj] for jj in range(n_k[kk])])
        for kk in range(K)])
    Sigma_hat_inv += -1 / sigma2_y * sum([(
            frac_eta_zeta[kk] *
            sum([sigma2_y / sigma2_eta * frac_eta[kk][jj] * Sx_jk[kk][jj]
                for jj in range(n_k[kk])]) *
            sum([sigma2_y / sigma2_eta * frac_eta[kk][jj] * Sx_jk[kk][jj]
                for jj in range(n_k[kk])]).reshape([-1, 1]))
        for kk in range(K)])
    # need square brackets for tensor reshape in the above!
    Sigma_hat_inv_mu_hat = np.dot(Sigma_inv, mu) + 1 / sigma2_y * Sxy
    Sigma_hat_inv_mu_hat += -1 / sigma2_y * sum([sum([
            frac_eta[kk][jj] * SxSy_jk[kk][jj] for jj in range(n_k[kk])])
        for kk in range(K)])
    Sigma_hat_inv_mu_hat += -1 / sigma2_y * sum([(
            frac_eta_zeta[kk] *
            sum([sigma2_y / sigma2_eta * frac_eta[kk][jj] * Sx_jk[kk][jj]
                for jj in range(n_k[kk])]) *
            sum([sigma2_y / sigma2_eta * frac_eta[kk][jj] * Sy_jk[kk][jj]
                for jj in range(n_k[kk])]))
        for kk in range(K)])
    mu_hat = pm.math.dot(
        pm.math.matrix_inverse(Sigma_hat_inv), Sigma_hat_inv_mu_hat)
    logp_ = -pm.math.log(pm.math.det(Sigma_hat_inv))
    logp_ += np.log(np.linalg.det(Sigma_inv))
    logp_ += -sum(m_k) * pm.math.log(2 * np.pi * sigma2_y)
    logp_ += sum(pm.math.log([
        sigma2_y / sigma2_zeta * frac_eta_zeta[kk] for kk in range(K)]))
    logp_ += sum(pm.math.log([(
            sigma2_y / sigma2_eta * frac_eta[kk][jj])
        for kk in range(K) for jj in range(n_k[kk])]))
    logp_ += -np.dot(mu, np.dot(Sigma_inv, mu))
    logp_ += pm.math.dot(mu_hat, Sigma_hat_inv_mu_hat)
    logp_ += -1 / sigma2_y * Syy
    logp_ += 1 / sigma2_y * sum([(
            sum([frac_eta[kk][jj] * SySy_jk[kk][jj] for jj in range(n_k[kk])]))
        for kk in range(K)])
    logp_ += 1 / sigma2_y * sum([
        frac_eta_zeta[kk] * sum([(
                sigma2_y / sigma2_eta * frac_eta[kk][jj] * Sy_jk[kk][jj])
            for jj in range(n_k[kk])])**2
        for kk in range(K)])
    logp_ = 0.5 * logp_
    return logp_


def mu_hat_hierarchical_mm(
        sigma2_y, sigma2_eta, sigma2_zeta, beta_mu, beta_Sigma_inv,
        SxSxT_jk, SxSy_jk, SySy_jk, SxxT, Sxy, Syy, Sx_jk, Sy_jk, n_k, m_jk,
        m_k, K):
    frac_eta = [
        [sigma2_eta / (sigma2_y + m_jk[kk][jj] * sigma2_eta)
            for jj in range(n_k[kk])]
        for kk in range(K)]
    frac_eta_zeta = [
        sigma2_zeta / (sigma2_y + sigma2_zeta * sum([
            (sigma2_y * m_jk[kk][jj]) / (sigma2_y + sigma2_eta * m_jk[kk][jj])
            for jj in range(n_k[kk])]))
        for kk in range(K)]
    frac_zeta = [
        sigma2_zeta / (sigma2_y + m_k[kk] * sigma2_zeta) for kk in range(K)]
    Sigma_hat_inv = beta_Sigma_inv + 1 / sigma2_y * SxxT
    Sigma_hat_inv += -1 / sigma2_y * sum([sum([
            frac_eta[kk][jj] * SxSxT_jk[kk][jj] for jj in range(n_k[kk])])
        for kk in range(K)])
    Sigma_hat_inv += -1 / sigma2_y * sum([(
            frac_eta_zeta[kk] *
            sum([sigma2_y / sigma2_eta * frac_eta[kk][jj] * Sx_jk[kk][jj]
                for jj in range(n_k[kk])]) *
            sum([sigma2_y / sigma2_eta * frac_eta[kk][jj] * Sx_jk[kk][jj]
                for jj in range(n_k[kk])]).reshape(-1, 1))
        for kk in range(K)])
    Sigma_hat_inv_mu_hat = np.dot(beta_Sigma_inv, beta_mu) + 1 / sigma2_y * Sxy
    Sigma_hat_inv_mu_hat += -1 / sigma2_y * sum([sum([
            frac_eta[kk][jj] * SxSy_jk[kk][jj] for jj in range(n_k[kk])])
        for kk in range(K)])
    Sigma_hat_inv_mu_hat += -1 / sigma2_y * sum([(
            frac_eta_zeta[kk] *
            sum([sigma2_y / sigma2_eta * frac_eta[kk][jj] * Sx_jk[kk][jj]
                for jj in range(n_k[kk])]) *
            sum([sigma2_y / sigma2_eta * frac_eta[kk][jj] * Sy_jk[kk][jj]
                for jj in range(n_k[kk])]))
        for kk in range(K)])
    mu_hat = np.dot(np.linalg.inv(Sigma_hat_inv), Sigma_hat_inv_mu_hat)
    Sb_jk = [[(
        Sy_jk[kk][jj] - np.dot(Sx_jk[kk][jj], mu_hat))
        for jj in range(n_k[kk])]
        for kk in range(K)]
    eta_hat = [[
        frac_eta[kk][jj] * Sb_jk[kk][jj] -
        frac_eta_zeta[kk] * frac_eta[kk][jj] * m_jk[kk][jj] * sum([(
                sigma2_y / sigma2_eta *
                frac_eta[kk][ll] * Sb_jk[kk][ll])
            for ll in range(n_k[kk])])
        for jj in range(n_k[kk])]
        for kk in range(K)]
    # sigma2_eta_hat = [[
    #     sigma2_y * frac_eta[kk][jj] + (
    #         1 + frac_eta_zeta[kk] * frac_eta[kk][jj] * m_jk[kk][jj]**2)
    #     for jj in range(n_k[kk])]
    #     for kk in range(K)]
    S_k = [
        sigma2_y * np.diag(frac_eta[kk]) + frac_eta_zeta[kk] * np.dot(
            np.array(
                [sigma2_y * frac_eta[kk][jj] * m_jk[kk][jj]
                    for jj in range(n_k[kk])]).reshape(-1, 1),
            np.array(
                [frac_eta[kk][jj] * m_jk[kk][jj]
                    for jj in range(n_k[kk])]).reshape(1, -1))
        for kk in range(K)]
    sigma2_eta_hat = [list(np.diag(S_k[kk])) for kk in range(K)]
    zeta_hat = [frac_zeta[kk] * (
        sum(Sy_jk[kk]) - np.dot(sum(Sx_jk[kk]), mu_hat) -
        sum([eta_hat[kk][jj] * m_jk[kk][jj] for jj in range(n_k[kk])]))
        for kk in range(K)]
    sigma2_zeta_hat = [sigma2_y * frac_zeta[kk] for kk in range(K)]
    output = dict(
        mu_tilde=mu_hat, Sigma_tilde_inv=Sigma_hat_inv,
        eta_hat=eta_hat, sigma2_hat_eta=sigma2_eta_hat,
        zeta_hat=zeta_hat, sigma2_hat_zeta=sigma2_zeta_hat)
    return output


def logp_general_mm(
        sigma2_y, Sigma_eta_inv, mu, Sigma_inv,
        SzzT_j, SzxT_j, Szy_j, SxxT, Sxy, Syy, n_j, J):
    Sigma_hat_eta_j_inv = [
        Sigma_eta_inv + 1 / sigma2_y * SzzT_j[jj] for jj in range(J)]
    Sigma_hat_eta_j = [pm.math.matrix_inverse(S) for S in Sigma_hat_eta_j_inv]
    Sigma_hat_inv = Sigma_inv + 1 / sigma2_y * SxxT
    Sigma_hat_inv += (
        -1 / sigma2_y**2 * sum([(
                pm.math.dot(SzxT_j[jj].T, pm.math.dot(
                    Sigma_hat_eta_j[jj], SzxT_j[jj])))
            for jj in range(J)]))
    Sigma_hat_inv_mu_hat = (
        pm.math.dot(Sigma_inv, mu) +
        1 / sigma2_y * Sxy - 1 / sigma2_y**2 * sum([(
                pm.math.dot(SzxT_j[jj].T, pm.math.dot(
                    Sigma_hat_eta_j[jj], Szy_j[jj])))
            for jj in range(J)]))
    mu_hat = pm.math.dot(
        pm.math.matrix_inverse(Sigma_hat_inv), Sigma_hat_inv_mu_hat)
    logp_ = -pm.math.log(pm.math.det(Sigma_hat_inv))
    logp_ += np.log(np.linalg.det(Sigma_inv))
    logp_ += -sum(n_j) * pm.math.log(2 * np.pi * sigma2_y)
    logp_ += J * pm.math.log(pm.math.det(Sigma_eta_inv))
    logp_ += -sum([
        pm.math.log(pm.math.det(Sigma_hat_eta_j_inv[jj])) for jj in range(J)])
    logp_ += pm.math.dot(mu_hat.T, Sigma_hat_inv_mu_hat)
    logp_ += -np.dot(mu.T, np.dot(Sigma_inv, mu))
    logp_ += -1 / sigma2_y * Syy
    logp_ += 1 / sigma2_y**2 * sum([(
            pm.math.dot(Szy_j[jj].T, pm.math.dot(
                Sigma_hat_eta_j[jj], Szy_j[jj])))
        for jj in range(J)])
    logp_ = 0.5 * logp_
    return logp_


def mu_hat_general_mm(
        sigma2_y, Sigma_eta_inv, beta_mu, beta_Sigma_inv,
        SzzT_j, SzxT_j, Szy_j, SxxT, Sxy, Syy, n_j, J):
    Sigma_hat_eta_j_inv = [
        Sigma_eta_inv + 1 / sigma2_y * SzzT_j[jj] for jj in range(J)]
    Sigma_hat_eta_j = [np.linalg.inv(S) for S in Sigma_hat_eta_j_inv]
    Sigma_hat_inv = beta_Sigma_inv + 1 / sigma2_y * SxxT
    Sigma_hat_inv += (
        -1 / sigma2_y**2 * sum([(
                np.dot(SzxT_j[jj].T, np.dot(Sigma_hat_eta_j[jj], SzxT_j[jj])))
            for jj in range(J)]))
    Sigma_hat_inv_mu_hat = (
        np.dot(beta_Sigma_inv, beta_mu) +
        1 / sigma2_y * Sxy - 1 / sigma2_y**2 * sum([(
                np.dot(SzxT_j[jj].T, np.dot(Sigma_hat_eta_j[jj], Szy_j[jj])))
            for jj in range(J)]))
    mu_hat = np.dot(np.linalg.inv(Sigma_hat_inv), Sigma_hat_inv_mu_hat)
    Sigma_hat_eta_j_inv_mu_hat_eta_j = (
        1 / sigma2_y * (Szy_j - np.dot(SzxT_j, mu_hat)))
    eta_hat = np.array([
        np.dot(
            np.linalg.inv(Sigma_hat_eta_j_inv[jj]),
            Sigma_hat_eta_j_inv_mu_hat_eta_j[jj])
        for jj in range(J)])
    Sigma_hat_eta_j_inv = np.array(Sigma_hat_eta_j_inv)
    output = dict(
        mu_tilde=mu_hat, Sigma_tilde_inv=Sigma_hat_inv,
        eta_hat=eta_hat, Sigma_hat_eta_inv=Sigma_hat_eta_j_inv)
    return output


def xy_sums(x, y):
    '''
    Generate fixed products and sums of data used in the integrated likelihood
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
    Generate fixed products and sums of data used in the integrated likelihood
    for the multilevel model
    '''
    J = np.unique(j).size
    n = x.shape[0]
    x_ij = [x[j == j_val, :] for j_val in np.unique(j)]
    y_ij = [y[j == j_val] for j_val in np.unique(j)]
    n_j = np.array([x.shape[0] for x in x_ij])
    # \sum_i x_{ij}
    Sx_j = np.array([
        sum([x_ij[jj][ii, :] for ii in range(n_j[jj])]) for jj in range(J)])
    # \sum_i y_{ij}
    Sy_j = np.array([np.sum(y_ij[jj], axis=0) for jj in range(J)])
    # (\sum_i x_{ij})(\sum_k x_{kj}^T)
    SxSxT_j = np.array([
        Sx_j[jj].reshape(-1, 1) @ Sx_j[jj].reshape(1, -1) for jj in range(J)])
    # (\sum_i x_{ij})(\sum_k y_{kj})
    SxSy_j = np.array([Sx_j[jj] * Sy_j[jj] for jj in range(J)])
    # (\sum_i y_{ij})(\sum_k y_{kj})
    SySy_j = np.array([Sy_j[jj]**2 for jj in range(J)])
    # \sum_{i,j} x_{ij} x_{ij}^T
    SxxT = np.array([
        x[ii, :].reshape(-1, 1) @ x[ii, :].reshape(1, -1)
        for ii in range(n)]).sum(axis=0)
    # \sum_{i,j} y_{ij} x_{ij}
    Sxy = (x * y.reshape(-1, 1)).sum(axis=0)
    # \sum_{i,j} y_{ij}^2
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


def xy_j_extra_sums(x, y, j):
    '''
    Generate fixed products and sums of data used in the integrated likelihood
    for the multilevel model
    '''
    J = np.unique(j).size
    x_ij = [x[j == j_val, :] for j_val in np.unique(j)]
    y_ij = [y[j == j_val] for j_val in np.unique(j)]
    n_j = np.array([x.shape[0] for x in x_ij])
    # \sum_i x_{ij} x_{ij}^T
    SxxT_j = np.array([
        np.array([
            x_ij[jj][ii, :].reshape(-1, 1) @ x_ij[jj][ii, :].reshape(1, -1)
            for ii in range(n_j[jj])]).sum(axis=0)
        for jj in range(J)])
    # \sum_i y_{ij} x_{ij}
    Sxy_j = np.array([
        (x_ij[jj] * y_ij[jj].reshape(-1, 1)).sum(axis=0) for jj in range(J)])
    # \sum_i y_{ij}^2
    Syy_j = np.array([(y_ij[jj]**2).sum(axis=0) for jj in range(J)])
    output = dict()
    output['SxxT_j'] = SxxT_j
    output['Sxy_j'] = Sxy_j
    output['Syy_j'] = Syy_j
    return output


def xy_jk_sums(x, y, j, k):
    '''
    Generate fixed products and sums of data used in the integrated likelihood
    for the hierarchical multilevel model (levels 0-2)
    '''
    J = np.unique(j).size
    K = np.unique(k).size
    j_k = [j[k == k_val] for k_val in np.unique(k)]
    n = x.shape[0]
    x_ijk = [
        [x[(k == np.unique(k)[kk]) & (j == np.unique(j)[jj]), :]
            for jj in range(J) if np.any(np.unique(j)[jj] == j_k[kk])]
        for kk in range(K)]
    y_ijk = [
        [y[(k == np.unique(k)[kk]) & (j == np.unique(j)[jj])]
            for jj in range(J) if np.any(np.unique(j)[jj] == j_k[kk])]
        for kk in range(K)]
    n_k = [np.unique(j).size for j in j_k]
    m_jk = [
        list(np.unique(j_k[k_ind], return_counts=True)[1])
        for k_ind in range(K)]
    m_k = [sum([m_jk[kk][jj] for jj in range(n_k[kk])]) for kk in range(K)]
    # \sum_i x_{ijk}
    Sx_jk = [[sum(x_ijk[kk][jj]) for jj in range(n_k[kk])] for kk in range(K)]
    # \sum_i y_{ijk}
    Sy_jk = [[sum(y_ijk[kk][jj]) for jj in range(n_k[kk])] for kk in range(K)]
    # (\sum_i x_{ijk})(\sum_q x_{qjk}^T)
    SxSxT_jk = [
        [Sx_jk[kk][jj].reshape(-1, 1) @ Sx_jk[kk][jj].reshape(1, -1)
            for jj in range(n_k[kk])]
        for kk in range(K)]
    # (\sum_i x_{ijk})(\sum_q y_{qjk})
    SxSy_jk = [
        [Sx_jk[kk][jj] * Sy_jk[kk][jj] for jj in range(n_k[kk])]
        for kk in range(K)]
    # (\sum_i y_{ijk})(\sum_q y_{qjk})
    SySy_jk = [[Sy_jk[kk][jj]**2 for jj in range(n_k[kk])] for kk in range(K)]
    # \sum_{ijk} x_{ijk} x_{ijk}^T
    SxxT = np.array([
        x[ii, :].reshape(-1, 1) @ x[ii, :].reshape(1, -1)
        for ii in range(n)]).sum(axis=0)
    # \sum_{ijk} y_{ijk} x_{ijk}
    Sxy = (x * y.reshape(-1, 1)).sum(axis=0)
    # \sum_{ijk} y_{ijk}^2
    Syy = (y**2).sum(axis=0)
    output = dict()
    output['SxSxT_jk'] = SxSxT_jk
    output['SxSy_jk'] = SxSy_jk
    output['SySy_jk'] = SySy_jk
    output['SxxT'] = SxxT
    output['Sxy'] = Sxy
    output['Syy'] = Syy
    output['Sy_jk'] = Sy_jk
    output['Sx_jk'] = Sx_jk
    output['n_k'] = n_k
    output['m_jk'] = m_jk
    output['m_k'] = m_k
    output['K'] = K
    # output['x_ijk'] = x_ijk
    # output['y_ijk'] = y_ijk
    return output


def xyz_j_sums(x, y, z, j):
    '''
    Generate fixed products and sums of data used in the integrated likelihood
    for the general multilevel model
    '''
    J = np.unique(j).size
    n = x.shape[0]
    x_ij = [x[j == j_val, :] for j_val in np.unique(j)]
    z_ij = [z[j == j_val, :] for j_val in np.unique(j)]
    y_ij = [y[j == j_val] for j_val in np.unique(j)]
    n_j = np.array([x.shape[0] for x in x_ij])
    # \sum_i z_{ij} z_{ij}^T
    SzzT_j = np.array([
        sum([z_ij[jj][ii, :].reshape(-1, 1) @ z_ij[jj][ii, :].reshape(1, -1)
            for ii in range(n_j[jj])]) for jj in range(J)])
    # \sum_i z_{ij} x_{ij}^T
    SzxT_j = np.array([
        sum([z_ij[jj][ii, :].reshape(-1, 1) @ x_ij[jj][ii, :].reshape(1, -1)
            for ii in range(n_j[jj])]) for jj in range(J)])
    # \sum_i z_{ij} y_{ij}
    Szy_j = np.array([
        sum([z_ij[jj][ii, :] * y_ij[jj][ii]
            for ii in range(n_j[jj])]) for jj in range(J)])
    # \sum_{i,j} x_{ij} x_{ij}^T
    SxxT = np.array([
        x[ii, :].reshape(-1, 1) @ x[ii, :].reshape(1, -1)
        for ii in range(n)]).sum(axis=0)
    # \sum_{i,j} y_{ij} x_{ij}
    Sxy = (x * y.reshape(-1, 1)).sum(axis=0)
    # \sum_{i,j} y_{ij}^2
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


def get_lm_integrated(
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


def get_mm_integrated(
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


def get_hierarchical_mm_integrated(
        pm_model, m_data, model_str='',
        sigma2_y_alpha=3, sigma2_y_beta=1,
        sigma2_eta_alpha=3, sigma2_eta_beta=1,
        sigma2_zeta_alpha=3, sigma2_zeta_beta=1,
        beta_mu=None, beta_Sigma_inv=None):
    '''
    Define the PyMC3 potential for a simple multilevel linear model (with
    group-varying intercept only), for MCMC sampling
    '''
    xy_dict = xy_jk_sums(**m_data)
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
        sigma2_zeta = pm.InverseGamma(
            'sigma2_zeta' + model_str,
            alpha=sigma2_zeta_alpha, beta=sigma2_zeta_beta)
    logp_input_dict = dict(
        sigma2_y=sigma2_y, sigma2_eta=sigma2_eta, sigma2_zeta=sigma2_zeta,
        mu=beta_mu, Sigma_inv=beta_Sigma_inv, **xy_dict)
    return logp_input_dict


def get_general_mm_integrated(
        pm_model, m_data, model_str='',
        sigma2_y_alpha=3, sigma2_y_beta=1,
        sigma2_eta_alpha=3, sigma2_eta_beta=1,
        beta_mu=None, beta_Sigma_inv=None,
        include_rho=False, rho_fixed=None, rho_eta_mu=0, rho_eta_var=1):
    '''
    Define the PyMC3 potential for a general multilevel linear model (with
    multiple group-varying covariates), for MCMC sampling
    '''
    xy_dict = xyz_j_sums(**m_data)
    d = m_data['x'].shape[1]
    m = m_data['z'].shape[1]
    rho_eta_shape = int(m*(m - 1) / 2)
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
            if rho_fixed is None:
                rho_eta = pm.TruncatedNormal(
                    'rho_eta' + model_str, mu=rho_eta_mu, tau=1/rho_eta_var,
                    lower=-1, upper=1, shape=rho_eta_shape)
            else:
                rho_eta = rho_fixed
            corr_matrix = (1 - rho_eta[0]) * np.eye(m)
            # triu_index has 0 on the diagonals, so adding rho_eta[triu_index]
            # adds rho_eta[0] to the diagonal terms. The above undos this!
            corr_matrix += rho_eta[triu_index]
            Sigma_eta_inv = pm.Deterministic(
                'Sigma_eta_inv' + model_str,
                pm.math.matrix_inverse(pm.math.matrix_dot(
                    pm.math.sqrt(sigma2_eta) * np.eye(m), corr_matrix,
                    pm.math.sqrt(sigma2_eta) * np.eye(m))))
        else:
            Sigma_eta_inv = pm.Deterministic(
                'Sigma_eta_inv' + model_str, 1 / sigma2_eta * np.eye(m))
    logp_input_dict = dict(
        sigma2_y=sigma2_y, Sigma_eta_inv=Sigma_eta_inv,
        mu=beta_mu, Sigma_inv=beta_Sigma_inv, **xy_dict)
    return logp_input_dict


def get_lm_full(
        pm_model, m_data, model_str='',
        sigma2_alpha=3, sigma2_beta=1, beta_mu=0, beta_Sigma_inv='id'):
    '''Define the PyMC3 potential for a linear model, for MCMC sampling'''
    d = m_data['x'].shape[1]
    if isinstance(beta_Sigma_inv, str):
        if beta_Sigma_inv == 'id':
            beta_Sigma_inv = np.eye(d)
    with pm_model:
        beta = pm.MvNormal('beta' + model_str, mu=beta_mu, tau=beta_Sigma_inv)
        sigma2 = pm.InverseGamma(
            'sigma2' + model_str, alpha=sigma2_alpha, beta=sigma2_beta)
        y = pm.math.dot(m_data['x'], beta)
    return y, sigma2


def get_lm_ninvg_full(
        pm_model, m_data, model_str='',
        sigma2_alpha=3, sigma2_beta=1,
        beta_mu=0, beta_Sigma_inv='id', gamma=1):
    '''Define the PyMC3 potential for a linear model, for MCMC sampling'''
    d = m_data['x'].shape[1]
    if isinstance(beta_Sigma_inv, str):
        if beta_Sigma_inv == 'id':
            beta_Sigma_inv = np.eye(d)
    with pm_model:
        sigma2 = pm.InverseGamma(
            'sigma2' + model_str, alpha=sigma2_alpha, beta=sigma2_beta)
        beta = pm.MvNormal(
            'beta' + model_str,
            mu=beta_mu, tau=1/(sigma2*gamma)*beta_Sigma_inv)
        y = pm.math.dot(m_data['x'], beta)
    return y, sigma2


def get_mm_full(
        pm_model, m_data, model_str='',
        sigma2_y_alpha=3, sigma2_y_beta=1,
        sigma2_eta_alpha=3, sigma2_eta_beta=1,
        beta_mu=0, beta_Sigma_inv='id'):
    '''Define the PyMC3 potential for a linear model, for MCMC sampling'''
    d = m_data['x'].shape[1]
    if isinstance(beta_Sigma_inv, str):
        if beta_Sigma_inv == 'id':
            beta_Sigma_inv = np.eye(d)
    unique_j = np.unique(m_data['j'])
    J = unique_j.size
    j = pd.Series(m_data['j']).replace(dict(zip(unique_j, range(J)))).values
    with pm_model:
        beta = pm.MvNormal('beta' + model_str, mu=beta_mu, tau=beta_Sigma_inv)
        sigma2_y = pm.InverseGamma(
            'sigma2_y' + model_str, alpha=sigma2_y_alpha, beta=sigma2_y_beta)
        sigma2_eta = pm.InverseGamma(
            'sigma2_eta' + model_str,
            alpha=sigma2_eta_alpha, beta=sigma2_eta_beta)
        eta = pm.Normal('eta' + model_str, mu=0, tau=1/sigma2_eta, shape=J)
        y = pm.math.dot(m_data['x'], beta) + eta[j]
    return y, sigma2_y


def get_hierarchical_mm_full(
        pm_model, m_data, model_str='',
        sigma2_y_alpha=3, sigma2_y_beta=1,
        sigma2_eta_alpha=3, sigma2_eta_beta=1,
        sigma2_zeta_alpha=3, sigma2_zeta_beta=1,
        beta_mu=0, beta_Sigma_inv='id'):
    '''Define the PyMC3 potential for a linear model, for MCMC sampling'''
    d = m_data['x'].shape[1]
    if isinstance(beta_Sigma_inv, str):
        if beta_Sigma_inv == 'id':
            beta_Sigma_inv = np.eye(d)
    unique_j = np.unique(m_data['j'])
    J = unique_j.size
    j = pd.Series(m_data['j']).replace(dict(zip(unique_j, range(J)))).values
    unique_k = np.unique(m_data['k'])
    K = unique_k.size
    k = pd.Series(m_data['k']).replace(dict(zip(unique_k, range(K)))).values
    with pm_model:
        beta = pm.MvNormal('beta' + model_str, mu=beta_mu, tau=beta_Sigma_inv)
        sigma2_y = pm.InverseGamma(
            'sigma2_y' + model_str, alpha=sigma2_y_alpha, beta=sigma2_y_beta)
        sigma2_eta = pm.InverseGamma(
            'sigma2_eta' + model_str,
            alpha=sigma2_eta_alpha, beta=sigma2_eta_beta)
        sigma2_zeta = pm.InverseGamma(
            'sigma2_zeta' + model_str,
            alpha=sigma2_zeta_alpha, beta=sigma2_zeta_beta)
        eta = pm.Normal('eta' + model_str, mu=0, tau=1/sigma2_eta, shape=J)
        zeta = pm.Normal('zeta' + model_str, mu=0, tau=1/sigma2_zeta, shape=K)
        y = pm.math.dot(m_data['x'], beta) + eta[j] + zeta[k]
    return y, sigma2_y


def get_general_mm_full(
        pm_model, m_data, model_str='',
        sigma2_y_alpha=3, sigma2_y_beta=1,
        sigma2_eta_alpha=3, sigma2_eta_beta=1,
        beta_mu=0, beta_Sigma_inv='id',
        include_rho=False, rho_fixed=None, rho_eta_mu=0, rho_eta_var=1):
    '''
    Define the PyMC3 potential for a general multilevel linear model (with
    multiple group-varying covariates), for MCMC sampling
    '''
    d = m_data['x'].shape[1]
    if isinstance(beta_Sigma_inv, str):
        if beta_Sigma_inv == 'id':
            beta_Sigma_inv = np.eye(d)
    m = m_data['z'].shape[1]
    unique_j = np.unique(m_data['j'])
    J = unique_j.size
    j = pd.Series(m_data['j']).replace(dict(zip(unique_j, range(J)))).values
    rho_eta_shape = int(m*(m - 1) / 2)
    if include_rho:
        triu_index = np.zeros((m, m), dtype=int)
        triu_index[np.triu_indices(m, 1)] = np.arange(rho_eta_shape)
        triu_index[np.triu_indices(m, 1)[::-1]] = np.arange(rho_eta_shape)
    with pm_model:
        beta = pm.MvNormal('beta' + model_str, mu=beta_mu, tau=beta_Sigma_inv)
        sigma2_y = pm.InverseGamma(
            'sigma2_y' + model_str, alpha=sigma2_y_alpha, beta=sigma2_y_beta)
        sigma2_eta = pm.InverseGamma(
            'sigma2_eta' + model_str,
            alpha=sigma2_eta_alpha, beta=sigma2_eta_beta, shape=m)
        if include_rho:
            # Include an off-diagonal term in the random effect covariance
            if rho_fixed is None:
                rho_eta = pm.TruncatedNormal(
                    'rho_eta' + model_str, mu=rho_eta_mu, tau=1/rho_eta_var,
                    lower=-1, upper=1, shape=rho_eta_shape)
            else:
                rho_eta = rho_fixed
            corr_matrix = (1 - rho_eta[0]) * np.eye(m)
            # triu_index has 0 on the diagonals, so adding rho_eta[triu_index]
            # adds rho_eta[0] to the diagonal terms. The above undos this!
            corr_matrix += rho_eta[triu_index]
            Sigma_eta_inv = pm.math.matrix_inverse(pm.math.matrix_dot(
                    pm.math.sqrt(sigma2_eta) * np.eye(m), corr_matrix,
                    pm.math.sqrt(sigma2_eta) * np.eye(m)))
        else:
            Sigma_eta_inv = (1 / sigma2_eta) * np.eye(m)
        eta = pm.MvNormal(
            'eta' + model_str, mu=0, tau=Sigma_eta_inv, shape=(J, m))
        y = pm.math.dot(m_data['x'], beta)
        y += pm.math.sum(m_data['z'] * eta[j, :], axis=1)
    return y, sigma2_y
