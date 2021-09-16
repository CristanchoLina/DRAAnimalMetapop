#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import random
from datetime import datetime

# In[2]:


np.set_printoptions(suppress=True)


# In[3]:

import vaccination_functions as epidec

import os
task_id = str(os.getenv("SLURM_ARRAY_TASK_ID"))

start_time = datetime.now()


# In[4]:



#Read fixed parameters: #Endemic state here!!!!!!

N0s = np.loadtxt('DRAAnimalMetapop/fixed_parameters/N0s.txt')
L = len(N0s)
delta, nb_steps = np.loadtxt('DRAAnimalMetapop/fixed_parameters/setting.txt')
nb_steps = int(nb_steps*2)
demo_params = np.loadtxt('DRAAnimalMetapop/fixed_parameters/demo_params.txt')
theta_edges = np.loadtxt('DRAAnimalMetapop/fixed_parameters/theta_edges.txt')

#Decision times(fct of nb of steps and duration decision)
social_duration = 180/delta
simul_list = np.array(range(0, nb_steps))
social_decision_times = simul_list[np.mod(simul_list, social_duration) == 0.0]
social_decision_times = social_decision_times[1:]
social_decision_times = social_decision_times[6:]

# In[5]:


fixed_parameters_list = [L, N0s, delta, nb_steps, demo_params, theta_edges]



# In[8]:


def model_full(variable_parameters_list, fixed_parameters_list = fixed_parameters_list):
    
    eff_reduceinfec = 0
    
    # Assign fixed parameters
    L, N0s, delta, nb_steps, demo_params, theta_edges = fixed_parameters_list
    
    # Assign variable parameters to be used from given list
    scoring_function, budget_percentage = variable_parameters_list
    budget_percentage = float(budget_percentage)
    
    # CENTRALIZED PARAMETERS #budget defined later
    social_duration = 180 # any step for decisions anyway there is no budget
    
    # Epidemic parameters
    prop_inf_nodes = 0.1 # proportion of initially infected nodes 
    prop_inf_in_node = 0.15 # proportion of animals infected in the infected nodes 
    recovery_time = 90 #1/gamma
    beta_gamma = 2 #beta/gamma

    # Useful fixed parameters
    gamma = 1/recovery_time
    beta = beta_gamma*gamma
    eff_reduceinfec = 0
    epid_params = np.array([[beta, gamma]]*L)
    
   	#Fixed useful arrays for the epidemic-decision path
   
   	#thetas et probas de sortie (fct de matrix theta_edges)
    probs_edges = epidec.proba_edges(L, theta_edges, delta) # k, sum_theta_k, p_out_k (vector)
   
   	#thetas
    mus, taus = demo_params[:,0], demo_params[:,1]
    thetas = probs_edges[:,1].astype(float)
   
   	#list of prob. of exit
    probs_exports = list(probs_edges[:,2])
   
   	# list of childs of each herd (for imports later)
    neighbors_list = theta_edges[:,1].astype(int)
   
   	# list of parents of each herd (for returns later)
    parents_list = theta_edges[:, 0].astype(int)
   
   	#theta edges in format for neighbor ewpw strategy
    theta_edges_compact = []
    for k in range(0,L):
   	    neighbors_k = []
   	    for w in theta_edges[theta_edges[:,0]== k]:
               neighbors_k.append(int(w[1]))
   	    for w in theta_edges[theta_edges[:,1]== k]:
               neighbors_k.append(int(w[0]))
   	    theta_edges_compact.append([k, neighbors_k ])
    theta_edges_compact = np.array(theta_edges_compact, dtype=object)
   
   
   	#Fixed probabilities
   
   	# prob. of birth
    p_B = 1.0 - np.exp(-mus * delta)
   
   	#prob. for R 
    R_rates = taus + thetas
    p_RD=  (1.0 - np.exp(- R_rates * delta))*taus/ R_rates 
    p_Rout =  (1.0 - np.exp(-R_rates * delta))*thetas/R_rates 
   
   	#prob. for I
    I_rates = gamma + R_rates
    p_IR =  (1.0 - np.exp(-I_rates * delta))*gamma/I_rates
    p_ID =  (1.0 - np.exp(-I_rates * delta))*taus/I_rates
    p_Iout = (1.0 - np.exp(-I_rates * delta))*thetas/I_rates
   
   	#Stock prob. vectors
    p_B = np.array([p_B, 1.0-p_B]).T 
    p_I = np.array([p_IR, p_ID, p_Iout, 1.0-(p_IR + p_ID + p_Iout)]).T 
    p_R = np.array([p_RD, p_Rout, 1.0-(p_RD + p_Rout)]).T
   
   	#fixed_epid_probas 
    fixed_epid_probas = [p_B, p_I, p_R]


    
    # Initial state creation: RANDOMLY CHOOSING THE INITIAL INFECTED NODES
    # (function of fixed and variable parameters: initial Ns, prop_inf_nodes, prop_inf_in_node)
    perm = random.sample(range(L), L)
    num_infected_noeuds = int(prop_inf_nodes*L)

    initial_states_inf = []
    initial_states_noninf = [] 
    for k in perm[:num_infected_noeuds]:
        initial_states_inf.append([k, epidec.create_initial_state(k, N0s, prop_inf_in_node)])
    for k in perm[num_infected_noeuds:]:
        initial_states_noninf.append([k, epidec.create_initial_state(k, N0s, 0)])
    initial_states = initial_states_inf + initial_states_noninf
    initial_states = sorted(initial_states, key=lambda index: index[0])
    initial_states = np.stack(np.array(initial_states)[:, 1])

    herds_sizes = np.sum(np.delete(initial_states, (1, 4, 6), 1), axis = 1) 
    budget = np.sum(herds_sizes)*budget_percentage 

    # Epidemic evolution
    social_decision_times_i, social_decisions_i,\
    all_states_i, ventes_byfarm_i, achats_byfarm_i, scores_i = epidec.path(gamma, taus[0], initial_states,\
                                                                                    demo_params, epid_params,\
                                                                                    fixed_epid_probas,\
                                                                                    neighbors_list,\
                                                                                    parents_list,\
                                                                                    probs_exports,\ social_duration, budget, eff_reduceinfec, eff_protect, thetas, delta, nb_steps,\
	                                                                            theta_edges_compact, theta_edges, social_scoring = scoring_function)
    
    # Agregated trajectory SIR 
    sir = np.zeros((nb_steps, L, 5))
    sir[:,:,0] = all_states_i[:,:,0] + all_states_i[:,:,3] #S
    sir[:,:,1] = all_states_i[:,:,1] + all_states_i[:,:,4] #S to I
    sir[:,:,2] = all_states_i[:,:,2] + all_states_i[:,:,5] #I
    sir[:,:,3] = all_states_i[:,:,6]  #I to R
    sir[:,:,4] = all_states_i[:,:,7]  # R
   
    #Taille des fermes a travers le temps
    N = np.sum(sir[:, :, [0,2,4]], axis = 2)
   
    # OUTPUTS CONCERNANT LES INFECTES A LA FIN DE LA TRAJECTOIRE 
    I = sir[:,:,2]
    I_moitie = I[int(nb_steps)-1]
    
    # 1. Output prop of infected herds at time T:
    prop_infected_farms_end = np.sum(I_moitie!=0)/L

    # Output prop vaccinated herds at each day
    real_times = np.nonzero(np.sum(social_decisions_i, axis = 1))
    prop_vaccinated = np.sum(social_decisions_i,axis = 1)[real_times,]/L


    #Wasted doses
    R = sir[:,:,4]
    wasted_doses = []
    for i in social_decision_times_i:
        vaccinated_i = social_decisions_i[i, :].astype(int)
        I_vaccinated_i = np.sum(I[i, np.where(vaccinated_i)[0]]).astype(int)
        R_vaccinated_i = np.sum(R[i, np.where(vaccinated_i)[0]]).astype(int)
        wasted_doses.append((I_vaccinated_i + R_vaccinated_i)/budget)

    return prop_infected_farms_end, prop_vaccinated, wasted_doses 

# In[9]:


def eval_function(combinations):
    prop_infected_farms_end, prop_vaccinated, wasted_doses  = model_full(combinations) 
    return prop_infected_farms_end, prop_vaccinated, wasted_doses 


combinations_task_id = np.genfromtxt('DRAAnimalMetapop/vaccination/percolation/score_buget_list/combinations_'+ task_id + '.txt', dtype='str') #task_id

# In[17]:


nb_runs = 50
start_time = datetime.now()
combinations_task_id_shape = 1


prop_infected_farms_end = np.zeros([nb_runs, combinations_task_id_shape])
prop_vaccinated = np.zeros([nb_runs, combinations_task_id_shape, len(social_decision_times)])
wasted_doses = np.zeros([nb_runs, combinations_task_id_shape, len(social_decision_times)])


# In[18]:

random.seed(1) # SEED
for i in range(0, nb_runs):
    prop_infected_farms_end[i],  prop_vaccinated[i], wasted_doses[i] = eval_function(combinations_task_id)
    
    
# In[19]:
np.savetxt('DRAAnimalMetapop/vaccination/percolation/percolation_vacc_endemic_data/' + 'prop_infected_farms_end_' + task_id + '.txt', prop_infected_farms_end, fmt='%1.2f')



#Save original shape

prop_vaccinated_original_shape = prop_vaccinated.shape

# Write the array to disk
with open('DRAAnimalMetapop/vaccination/percolation/percolation_vacc_endemic_data/' +'prop_vaccinated_' + task_id + '.txt', 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    outfile.write('DRAAnimalMetapop/vaccination/percolation/# Array shape: {0}\n'.format(prop_vaccinated_original_shape))

    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for data_slice in prop_vaccinated:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        np.savetxt(outfile, data_slice)



#Save original shape

wasted_doses_original_shape = wasted_doses.shape

# Write the array to disk
with open('DRAAnimalMetapop/vaccination/percolation/percolation_vacc_endemic_data/' +'wasted_doses_' + task_id + '.txt', 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    outfile.write('DRAAnimalMetapop/vaccination/percolation/# Array shape: {0}\n'.format(wasted_doses_original_shape))

    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for data_slice in wasted_doses:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        np.savetxt(outfile, data_slice)

        # Writing out a break to indicate different slices...
        outfile.write('DRAAnimalMetapop/vaccination/percolation/# New step \n')



        # Writing out a break to indicate different slices...
        outfile.write('DRAAnimalMetapop/vaccination/percolation/# New step \n')


