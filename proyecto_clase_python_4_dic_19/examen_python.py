# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:54:36 2019

@author: Rogelio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mwinai as mw
import corner

#%%
#Data and functions asociated with a class == atributes and methods

#Class == blueprint for creating instances

class proyect(object):
    """
    Clase que lee un archivo con modelos de sintesis de poblacion en formato CSV y contiene la
    funcionalidad de entrenar una red neuronal y realizar estimaciones por medio de algoritmos geneticos.
    """
    def __init__(self, data, X_parameters, Y_parameters):
        try:
            self.data = pd.read_csv(data)
        except FileNotFoundError:
            print("El archivo no se ha encontrado, o el path es incorrecto.")
            
        
        del self.data['Unnamed: 0']
        self.x_params = X_parameters
        self.y_params = Y_parameters
        self.data["log_age"] = np.log10(self.data["Age(yr)"])
        self.X_train = self.data[X_parameters].to_numpy()
        self.Y_train = self.data[Y_parameters].to_numpy()
        
    def entrenador(self, ANN_name,activation = 'tanh',solver = "lbfgs",saved = False):
        """
        Metodo para entrenar una red neuronal tomando los parametros fisicos que se deben de encontrar en el fichero
        csv con los modelos de síntesis de población (Masa, edad y metalicidad), arroja como output una red
        entrenada capaz de predecir fotometrías en 7 bandas distintas. 
        
        ANN_name = nombre de la red (string)
        
        activation = nombre de la función de activación con la cual se entrenará la red por default se entrena con tanh
        
        solver = solucionador de la red, por defecto emplea lbfgs (bueno para problemas regresionales).
        
        saved = boleano que indica si la red fue guardada con anterioridad, si fue de esta manera el metodo leerá la red y la muestra
                con la que se entrenó la red y hará predicción en base a esta muestra. default = False.
        """
        RM = mw.manage_RM(RM_filename = ANN_name)
        
        if saved != True or RM == 0:
            
            print("No hay una red pre-existente, inicializano entrenamiento...")
            X_train =self.X_train
            Y_train = self.Y_train
            
            
            RM = mw.manage_RM(RM_type = 'SK_ANN', X_train=X_train, y_train=Y_train,
                              scaling=True, clear_session=True, split_ratio = 0.2)
            
        
            RM.init_RM(hidden_layer_sizes = (50,50, 50), tol = 1e-6, max_iter = 2000, activation = activation,
                       solver = solver)
            print("Entrenando red {}...".format(ANN_name))
            RM.train_RM()
            print("TODO BIEN!")
            RM.predict()
            print("Guardando red y muestra de entrenamiento...")
            RM.save_RM(ANN_name, save_train = True, save_test = True)
            print("Listo! Red entrenada con {} parametros".format(len(X_train)))
        
        else:
            print("Red existente, cargando red y datos de entrenamiento guardados!")
            RM.predict(scoring = True)
            print("Prediciendo...")
            print('Listo!')
        return RM
        

    def population(self, Pop_number, X_train):
        """Funcion generadora de poblacion para algoritmo genético.
        
        Entrada: Numero de individuos en una poblacion
        
        Salida: Poblacion al azar de inviduos alrededor de el rango de valores
                del espacio de parametros con el que se entreno la red neuronal
                de la funcion entrenador
        """
        param_dim = X_train.shape
        pop =  X_train.min(0) + ((X_train.max(0) - X_train.min(0)) * np.random.rand(Pop_number, param_dim[-1]))
        
        
       
        return pop
    
    def model(self, ANN,population):
        ANN.set_test(population)
        ANN.predict()
        mod = ANN.pred
        return mod
    
    def fitness(self,ANN,population, goal, X_train, distance):
        """Calcula distancia entre el target (observaciones) y el modelo
        
        Entrada: ANN: red neuronal para emplearse en la funcion model.
        population (array) = array en el espacio de parámetros con los N-individuos de una poblacion
        goal = medicion o target que se busca aproximar con el modelo
        X_train = Muestra de parámetros con los uqe se entrenó la red nueronal.
        
        Salida: fit : distancia entre estimacion y target
        mod = estimacion f(parametros) producida por la red neuronal.
        
        """
        mod = self.model(ANN, population)
        X = X_train.shape[1]
        mask = ((population >= X_train.min(0)) & (population <= X_train.max(0))).sum(1) != X
        
        if distance == 'lnlike':
            fit = -np.sum((mod[:] - goal)**2,1)
            fit[mask] = -np.inf
        elif distance == 'euclidian':
            fit = -np.sqrt(np.sum((mod[:] - goal)**2,1))
            fit[mask] = -np.inf
        return fit, mod
    
    def mutation(self, selection, deltas, N_worst):
        mut = []
        for j in range(0,len(selection)):
            mut.append(selection[j] + (deltas[j] *  np.random.randn(N_worst)))
        return mut

    def genetic_alg(self,pop_num, N_best, N_worst, gen_num, goal, TOL,  ANN = None, distance = 'lnlike'):
        
        if ANN != None:
            print("Especificaste una ANN se empleará esa.")
            red = ANN
            training = red.X_train_unscaled
        else:
            print('No has especificado una red, se creará una de manera rápida.')
            print('Empleando funcion de activacion = tanh, y solver = adam.')
            red = self.entrenador(ANN_name = 'quick_test',activation = 'tanh',solver = "adam",saved = False)
            training = red.X_train_unscaled
        #Begin first population
        print("Comenzando con primera poblacion de {}".format(pop_num))
        pop1 = self.population(pop_num, training)
        
        #Calculate fitness and models
        fitness_g1, params_g1= self.fitness(red, pop1, goal, training, distance)
        print(fitness_g1)
        mask = fitness_g1 < TOL
        #Drop all the models that doesnt match the tolerance criteria...
        fitness_g1 = fitness_g1[~mask]
        pop1 = pop1[~mask]
        deltas = []
        pop_list = []
        #Taking little deltas for further mutations.
        for i in range(0, len(pop1.T)):
            deltas.append(pop1[:, i].max() - pop1[:, i].min())
            pop_list.append(pop1[:, i])
        
        pop_list.append(fitness_g1)
        all_array = np.array(pop_list).T
        #Sorting parameters an selection N_best parameters
        columnIndex = 3
        sortedArr = all_array[all_array[:,columnIndex].argsort()]
        sort = []
        print("Comenzando a mutar {} generaciones".format(gen_num))
        for i in range(0, len(pop1.T)):
            sort.append(sortedArr[:N_best, i])
            
        for gen in range(gen_num):
            print("Generacion {}".format(gen))
            selection = np.array(sort).T
            
            for i in range(0, len(selection)):
                #Creating new data around the distribution of values of N_best params.
                mut = self.mutation(selection[i, :], deltas= deltas, N_worst = N_worst) 
                new_pop = np.array(mut).T

                new_pop_fit, params_new = self.fitness(red, new_pop, goal, training, distance)
                mask = new_pop_fit < TOL
                new_pop_fit = new_pop_fit[~mask]
                new_pop = new_pop[~mask]
                new_stack = np.hstack((new_pop, np.atleast_2d(new_pop_fit).T))
                all_array = np.append(all_array, new_stack, axis = 0)
        print("Listo! Escupiendo resultado, have fun!...")
        res = pd.DataFrame()
        counter = 0
        for element in self.x_params:
            res['{}'.format(element)] = all_array[:, counter]
            counter += 1
            
        res['distance'] = all_array[:, -1]
        #res.rename(columns={0: 'Mass', 1: 'Age', 2: 'Met', 3:'Distance'}, inplace=True)
        params_all_array = all_array[:, 0:len(self.x_params)]
        modelos = self.model(red, params_all_array)
        counter = 0
        for elemnt in self.y_params:
            
            res['{}'.format(elemnt)] = modelos[:, counter]
            counter += 1
        self.res = res
        return res
    
    def corner_plot(self, bins, param_list):
        sample = self.res[param_list]
        mean = sample.mean()
        median = sample.median()
        ndim = sample.shape[-1]
        labels = param_list
        figure =  corner.corner(res[param_list], bins=20, smooth=3,plot_datapoints=True, quantiles=(0.16,0.5, 0.84), labels = labels, show_titles = True)
        axes = np.array(figure.axes).reshape(ndim, ndim)
        
        # Loop over the diagonal
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(mean[i], color="g")
            ax.axvline(median[i], color="r")

        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(mean[xi], color="g")
                ax.axvline(median[xi], color="r")
                ax.axhline(mean[yi], color="g")
                ax.axhline(median[yi], color="r")
                ax.plot(mean[xi], mean[yi], "sg")
                ax.plot(median[xi], median[yi], "sr")
    
    
            
     

#%%  
X_parameters = ["Mass (Mo)", "log_age", "Metal(Zo)"]
Y_parameters = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W', 'F547M', 'F657N']
A = proyect('all_data_good.csv',X_parameters = X_parameters,Y_parameters = Y_parameters)
A.data['log_met'] = np.log10(A.data['Metal(Zo)'])
A.data['log_mass'] = np.log10(A.data['Mass (Mo)'])
A.x_params = ['log_mass', 'log_age', 'log_met']
X_train = A.data[A.x_params].to_numpy()
A.X_train = X_train

red = A.entrenador('train_with_logmet',activation = 'tanh',solver = "adam", saved = False)


#%%
#red2 = A.entrenador('quick_test', saved = Fa)
data = pd.read_csv("NGC7793W_obs_wth_halpha")
dist_modulu = 27.68


a = data.loc[data['bst_mass'] >=  1000]
targets = a.loc[(a['bst_age'] >=  1e6) & (a['bst_age'] <=  8e6)]

targets['log_age'] = np.log10(targets['bst_age']) 

obs_1 = targets.loc[targets['ID_1'] == 513].copy()
obs_2 = targets.loc[targets['ID_1'] == 1252].copy()
target_phot = obs_1[['F275W_1', 'F336W_1', 'F438W_1', 'F555W_1', 'F814W_1', 'rw_F547M', 'rw_F657N']].to_numpy()
target_phot2 = obs_2[['F275W_1', 'F336W_1', 'F438W_1', 'F555W_1', 'F814W_1']].to_numpy()
target_phot = target_phot - 27.68
target_phot2 = target_phot2 - 27.68
res = A.genetic_alg(pop_num = 100000, ANN = red, N_best = 10, N_worst = 1000, gen_num = 100,goal =  target_phot, TOL = -5, distance = 'euclidian')
A.corner_plot(bins = 20, param_list=['log_mass', 'log_age', 'log_met'])
