# Bayesian model selection for multilevel models using integrated likelihoods

This repository contains Python scripts for estimation of the model evidence for 
multilevel models within a Bayesian framework, using the integrated likelihood for 
improved Markov Chain Monte Carlo (MCMC) sampling. 

## Description

We provide a framework to evaluate the model evidence using sequential Monte Carlo 
(SMC) with the integrated likelihood and with the full likelihood 
(`model_setup_with_marginals.py`), as well as computation of the Akaike information 
criterion (AIC), a frequentist model selection tool. We illustrate this in practice 
with simulated datasets (`evaluate_models_simulation.py`)  and on examples of linear
and multilevel models for the *Minnesota radon contamination* dataset 
(`evaluate_models_radon.py`). Both examples have a hierarchical structure, and the 
latter is often used in multilevel modelling tutorials, e.g. 
[this PyMC3 primer](https://docs.pymc.io/en/v3/pymc-examples/examples/case_studies/multilevel_modeling.html). 
An accompanying manuscript describing the underlying methodology is available as a 
preprint (Edinburgh et al. 2022). The repository also contains figures and tables 
within this manuscript, as well as all data files.

## Usage

To run this code, you can clone this repository. You will need [Git](https://git-scm.com) 
and [conda](https://docs.conda.io) (or equivalent for package, dependency and environment 
management) installed on your computer. From your command line:

```bash
# Clone this repository
$ git clone https://github.com/tedinburgh/model-evidence-with-integrated-likelihood

# Go into the repository
$ cd model-evidence-with-integrated-likelihood

# Install dependencies and create virtual environment
$ conda create --name integrated_likelihood_pymc_env --file requirements.txt
$ conda activate integrated_likelihood_pymc_env
```

To generate all the results in this repository, run: 

```bash
# Run script to generate results for all models in the simulated datasets
$ mkdir simulated_data
$ python evaluate_models_simulated.py
# Run script to generate results for all models on Minnesota radon contamination dataset
$ mkdir radon_data
$ python evaluate_models_radon.py
```
You can provide arguments to this function, notably `--model_list`. This should be a comma-separated 
string, containing a subset of '0, 1, 2, 3' for the simulated data and'0, 1, 2, 3, 4, 5' for the 
*Minnesota radon dataset*. Full details of arguments are provided in the code. 
You may find that model 5 runs a bit slowly for moderate size datasets, as it involves complex linear 
algebra in the defined integrated likelihood. It may help to use a 
[screen](https://www.gnu.org/software/screen/manual/screen.html) session when running
the script, if you include this model and are working remotely.

Furthermore, we note that results may not to be identically replicable, even when initiated in PyMC
with identical random seeds. This is likely an artefact of a sufficiently complex model set-up 
involving MCMC sampling (where one parameter value in sampling explicitly relies on the values of other 
sampled parameters, which propagates even the smallest numerical error). However, you should get very 
similar results to those presented in `smc_likelihoods.csv`.

To close the virtual environment after usage:

```bash
conda deactivate
```

You can adapt the examples to your own data, 
importing the script `model_setup_with_marginals.py` similarly, and formatting the dataset 
as a dictionary in the same manner. The simplest example script is:
```bash
import numpy as np
import pymc3 as pm
import model_setup_with_marginals as msm
# Change the ellipses to your data
lm_data = dict(x=..., y=...)
mm_data = dict(x=..., y=..., j=...)
general_mm_data = dict(x=..., y=..., z=..., j=...)
# Change as appropriate
sample_inputs = dict(
    n_draws=2000,
    random_seed=8924,
    n_chains=8,
    save_output=True,
    output_file_path='./')
# Similarly replacing 'lm' with 'mm', 'general_mm' below
trace_marginal_lm, trace_full_lm = sample_lm_model(
  lm_data, m=0, **sample_inputs)
print(trace_marginal_lm.report.log_marginal_likelihood)
print(trace_full_lm.report.log_marginal_likelihood)
```

## Contributing

We encourage other researchers (or anyone interested in this code from a methodology
or application perspective) to submit an 
[issue](https://github.com/tedinburgh/model-evidence-with-marginal/issues) on this
repository if they would like to report or fix any bugs or to contribute improvements.
