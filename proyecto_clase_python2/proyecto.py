#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:13:50 2019

@author: roger
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import mwinai as mw
import numpy as np
import corner
import emcee
#import matplotlib.mlab as mlab

#from scipy import stats
# pymc3 for Bayesian Inference, pymc built on t
#import pymc3 as pm
#import theano.tensor as tt
#import scipy
#from scipy import optimize 

#%%

df = pd.read_csv("all_data_good.csv")

del df["Unnamed: 0"]
df["log_age"] = np.log10(df["Age(yr)"])
df["log_met"] = np.log10(df["Metal(Zo)"])
columnas = df.columns


sub_df = df[[columnas[0], columnas[1], columnas[2], columnas[3], columnas[4],
            columnas[5], columnas[6], columnas[8],columnas[10], columnas[11], columnas[13], columnas[14]]].copy()


data = pd.read_csv('NGC7793W_obs_wth_halpha')


#%%
#Vector con las edades
Y_train = sub_df[['F275W', 'F336W', 'F438W', 'F555W', 'F814W', 'F547M', 'F657N']].to_numpy()

X_train = sub_df[['Mass (Mo)', 'Mup (Mo)', 'log_age', 'log_met']].to_numpy()

#%%
def generate_ANN(ANN_name, X_train, Y_train,saved = False):
    RM = mw.manage_RM(RM_filename = ANN_name)
    if (RM.N_in == 0) or (saved == False):
        activation = 'tanh'
        RM = mw.manage_RM(RM_type = 'SK_ANN_Dis', X_train=X_train, y_train=Y_train,
                   scaling=True, clear_session=True, split_ratio = 0.4, verbose = True)
        RM.init_RM(hidden_layer_sizes = (50, 50), tol = 1e-6, max_iter = 2000, activation = activation,
                  solver = "adam")
        RM.train_RM()
        RM.predict(scoring = True)
        RM.save_RM(ANN_name, save_train = True, save_test = True)
    return RM


RM = generate_ANN('prueba2', X_train, Y_train, saved = False)

#%%
f, axes = plt.subplots(3, 3, figsize=(10,10))
RM.predict()
x_valid = RM.y_test
y_valid = RM.pred - RM.y_test
for i, ax in enumerate(axes.ravel()):
    ax.scatter(x_valid[:,i], y_valid[:,i], alpha=0.01)
    


#%%
def model(parameters):
    RM.set_test(np.array(parameters).reshape(-1,1).T)
    RM.predict()
    model_out = RM.pred[0]
    return model_out
#%%
#El modelo recibe un vector theta(Mass, Mup, log_age, log_met)
dist_modulu = 27.68

target = data[['F275W_1', 'F336W_1', 'F438W_1', 'F555W_1', 'F814W_1', 'rw_F547M', 'rw_F657N', 'F275W_err_1', 'F336W_err', 
               'F438W_err', 'F555W_err_1', 'F814W_err_1', 'rw_F547M_err', 'rw_F657N_err', 'bst_mass', 'bst_age']].sample(n = 1)
target_phot = target[['F275W_1', 'F336W_1', 'F438W_1', 'F555W_1', 'F814W_1', 'rw_F547M', 'rw_F657N']].to_numpy()
target_phot = target_phot - 27.68
target_phot = np.reshape(target_phot, (7,))

target_error = target[['F275W_err_1', 'F336W_err', 
               'F438W_err', 'F555W_err_1', 'F814W_err_1', 'rw_F547M_err', 'rw_F657N_err']].to_numpy()
target_error = np.reshape(target_error, (7,))

#%%
def lnlike(parameters):
    LnLike = -1 * np.sum(((model(parameters) - target_phot)/target_error)**2)

    return LnLike

def prior(parameters):
    Mass, Mup, age, met = parameters
    if (1e2 < Mass < 7e3) and (99. < Mup < 301.) and (5.0 < age < 8.0) and (-4 < met < 0.):
        return 0.0
    return -np.inf



def lnprob(parameters):
    lp = prior(parameters)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(parameters)


def main(p0, walkers, niter, ndim, lnprob):
    sampler = emcee.EnsembleSampler(walkers, ndim, lnprob)
    
    print("Calentamiento..")
    p0, _, _ = sampler.run_mcmc(p0, 100)
    sampler.reset()
    
    print("Corriendo MCMC")
    pos, porb, state = sampler.run_mcmc(p0, niter)
    
    return sampler, pos, porb, state


#%%
walkers = 500
niter = 1000
n_sample = 1
initial = sub_df[['Mass (Mo)', 'Mup (Mo)', 'log_age', 'log_met']].sample(n = 1).to_numpy()
initial = np.reshape(initial, (4,))
#initial =  [194.9, 100, 6.6020, -1.85]
ndim = len(initial)
p0 = [initial + 1e-1 * np.random.randn(ndim) for i in np.arange(walkers)]   
p01 = list( (X_train.max(0) - X_train.min(0)) * np.random.rand(walkers, ndim))

#%%
RM.verbose = False
sampler, pos, porb, state = main(p0, walkers, niter, ndim, lnprob)

#%%
samples = sampler.flatchain
print(samples[np.argmax(sampler.flatlnprobability)])
labels = ['Mass (Mo)', 'Mup (Mo)', 'log_age', 'log_met']
fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True)




