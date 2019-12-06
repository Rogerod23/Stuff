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





data = pd.read_csv('NGC7793W_obs_wth_halpha')


#%%
def generate_ANN(ANN_name, X_train, Y_train, activation, layers, solver,saved = False):
    RM = mw.manage_RM(RM_filename = ANN_name)
    if (RM.N_in == 0) or (saved == False):

        RM = mw.manage_RM(RM_type = 'SK_ANN', X_train=X_train, y_train=Y_train,
                   scaling=True, clear_session=True, split_ratio = 0.1, verbose = True)
        RM.init_RM(hidden_layer_sizes = layers, max_iter = 2000, tol = 1e-7, activation = activation,
                  solver = solver)
        RM.train_RM()
        RM.predict(scoring = True)
        RM.save_RM(ANN_name, save_train = True, save_test = True)
    else:
        print('Already saved ANN, reading..')
        RM.predict(scoring = True)
        print('DONE!')
    return RM


def model(parameters):
    RM.set_test(np.array(parameters).reshape(-1,1).T)
    RM.predict()
    model_out = RM.pred[0]
    return model_out


def lnlike(parameters):
    LnLike = -0.5 * np.sum(((model(parameters) - target_phot)/target_error)**2)

    return LnLike

def prior(parameters):
    Mass, age, met = parameters
    if (3. <= Mass <= 5.) and (6.0 <= age <= 6.899820502427096) and ( -3.0 <= met <= -1.853871964321762):
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
    print("Listo!")
    
    return sampler, pos, porb, state
#%%
#Vector con las edades
    

df = pd.read_csv("all_data_good.csv")

del df["Unnamed: 0"]
df["log_age"] = np.log10(df["Age(yr)"])
df["log_met"] = np.log10(df["Metal(Zo)"])


sub = df.loc[df['Metal(Zo)'] == 0.013999999999999999 ].copy()
sub['age_bin'] = 0
sub.loc[sub['Age(yr)'] ==1e6, 'age_bin'] = 1
sub.loc[sub['Age(yr)'] ==3.0e6, 'age_bin'] = 2
sub.loc[sub['Age(yr)'] ==3.98e6, 'age_bin'] = 3
sub.loc[sub['Age(yr)'] ==7.94e6, 'age_bin'] = 4
sub['log_mass'] = np.log10(sub['Mass (Mo)'])

Y_train_det = sub[['F275W', 'F336W', 'F438W', 'F555W', 'F814W', 'F547M', 'F657N']].to_numpy()

X_train_det = sub[['log_mass', 'age_bin', 'log_met']].to_numpy()
    

#%%

RM = generate_ANN('det_mass_ANN', X_train_det, Y_train_det,activation ='tanh', layers = (100, 100, 50, 50, 30, 30, 20, 10, 5), solver= 'lbfgs' , saved = False)

#%%
f, axes = plt.subplots(3, 2, figsize=(10,10))

RM_qlty = RM.pred - RM.y_test

for i, ax in enumerate(axes.ravel()):
    x = RM_qlty[:,i]
    ax.hist(x, bins = np.linspace(-1, 1, 100))
    



#%%
#El modelo recibe un vector theta(Mass, Mup, log_age, log_met)
dist_modulu = 27.68


a = data.loc[data['bst_mass'] >=  1000]
targets = a.loc[(a['bst_age'] >=  1e6) & (a['bst_age'] <=  8e6)]

targets['log_age'] = np.log10(targets['bst_age']) 

obs_1 = targets.loc[targets['ID_1'] == 513].copy()
obs_2 = targets.loc[targets['ID_1'] == 1252].copy()
target_phot = obs_1[['F275W_1', 'F336W_1', 'F438W_1', 'F555W_1', 'F814W_1']].to_numpy()
target_phot = target_phot - 27.68
target_phot = np.reshape(target_phot, (5,))

target_error = obs_1[['F275W_err_1', 'F336W_err', 
               'F438W_err', 'F555W_err_1', 'F814W_err_1']].to_numpy()
target_error = np.reshape(target_error, (5,))


#%%
obs_2 = targets.loc[targets['ID_1'] == 1252].copy()
target_phot = obs_2[['F275W_1', 'F336W_1', 'F438W_1', 'F555W_1', 'F814W_1']].to_numpy()
target_phot = target_phot - 27.68
target_phot = np.reshape(target_phot, (5,))

target_error = obs_2[['F275W_err_1', 'F336W_err', 
               'F438W_err', 'F555W_err_1', 'F814W_err_1']].to_numpy()
target_error = np.reshape(target_error, (5,))





#%%
#MCMC para una region HII del tipo 3.
dist_modulu = 27.68


a = data.loc[data['bst_mass'] >=  1000]
targets = a.loc[(a['bst_age'] >=  1e6) & (a['bst_age'] <=  8e6)]

targets['log_age'] = np.log10(targets['bst_age']) 

obs_1 = targets.loc[targets['ID_1'] == 513].copy()
target_phot = obs_1[['F275W_1', 'F336W_1', 'F438W_1', 'F555W_1', 'F814W_1', 'rw_F547M', 'rw_F657N']].to_numpy()
target_phot = target_phot - 27.68
target_phot = np.reshape(target_phot, (7,))

target_error = obs_1[['F275W_err_1', 'F336W_err', 
               'F438W_err', 'F555W_err_1', 'F814W_err_1','rw_F547M_err', 'rw_F657N_err']].to_numpy()
target_error = np.reshape(target_error, (7,))    


walkers = 1000
niter = 200
n_sample = 1
initial = sub_df[['log_mass', 'log_age', 'log_met']]
initial = initial.loc[51715, ['log_mass', 'log_age', 'log_met']].to_numpy()
initial = np.reshape(initial, (3,))
#initial =  [194.9, 100, 6.6020, -1.85]
ndim = len(initial)
p0 = [initial + 1e-2 * np.random.randn(ndim) for i in np.arange(walkers)]   
p01 = list(X_train.min(0) + ((X_train.max(0) - X_train.min(0)) * np.random.rand(walkers, ndim)))


RM.verbose = False
sampler, pos, porb, state = main(p01, walkers, niter, ndim, lnprob)
#%%
samples = sampler.flatchain

print(10**(samples[np.argmax(sampler.flatlnprobability)]))
labels = ['Mass (Mo)', 'log_age', 'log_met']
samp = pd.DataFrame(samples, columns = labels)
print(obs_1[['bst_mass', 'bst_age']])
fig = corner.corner(samp,show_titles=True,labels=labels,plot_datapoints=True, bins = 10, smoth = 3)


#%%
#MCMC para una region HII compacta (tipo 1)
a = data.loc[data['bst_mass'] >=  1000]
targets = a.loc[(a['bst_age'] >=  1e6) & (a['bst_age'] <=  8e6)]

targets['log_age'] = np.log10(targets['bst_age']) 

obs_2 = targets.loc[targets['ID_1'] == 1252].copy()
target_phot = obs_2[['F275W_1', 'F336W_1', 'F438W_1', 'F555W_1', 'F814W_1']].to_numpy()
target_phot = target_phot - 27.68
target_phot = np.reshape(target_phot, (5,))

target_error = obs_2[['F275W_err_1', 'F336W_err', 
               'F438W_err', 'F555W_err_1', 'F814W_err_1']].to_numpy()
target_error = np.reshape(target_error, (5,))


walkers = 500
niter = 2000
n_sample = 1
initial = sub_df[['Mass (Mo)', 'log_age', 'log_met']]
initial = initial.loc[47737, ['Mass (Mo)', 'log_age', 'log_met']].to_numpy()
initial = np.reshape(initial, (3,))
#initial =  [194.9, 100, 6.6020, -1.85]
ndim = len(initial)
p0 = [initial + 1e-7 * np.random.randn(ndim) for i in np.arange(walkers)]   
p01 = list(X_train.min(0) + (X_train.max(0) - X_train.min(0)) * np.random.rand(walkers, ndim))

RM.verbose = False
sampler, pos, porb, state = main(p01, walkers, niter, ndim, lnprob)

#%%
samples = sampler.flatchain
print(samples[np.argmax(sampler.flatlnprobability)])
labels = ['Mass (Mo)', 'log_age', 'log_met']
fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True)



