#!/usr/bin/env python
# coding: utf-8

# Import modules
import numpy as np
import random
import numba
import igraph
import pandas as pd

#### Epidemic functions definition ####


def proba_edges(L, theta_edges, delta):
    '''
    Function that converts theta_ij into probabilities
    Uses theta_edges, delta (fixed parameters)
    '''
    proba_edges = []
    for k in range(0,L):
        theta_k = theta_edges[theta_edges[:,0] == k, 2]
        sum_theta_k = np.sum(theta_k)
        if sum_theta_k > 0:
            p_out_k = ((1.0 - np.exp(-sum_theta_k * delta))*theta_k/sum_theta_k)[:-1]
            p_out_k = np.append(p_out_k, 1 - np.sum(p_out_k))
        else:
            p_out_k = np.array([0.0])
        proba_edges.append([k, sum_theta_k, p_out_k])
        
    return np.array(proba_edges, dtype=object)

def create_initial_state(k, N0s, prop_infected_in_node):
    '''Function for creating the initial state of a herd with the given initial size and prop of infected animals'''
    N0 = N0s[k]
    I0 = int(N0*prop_infected_in_node)
    S0 = N0 - I0
    initial_state_k = [ S0,                                   # Snv
                        0,                                    # Snv to Inv
                        I0,                                   # Inv
                        0,                                    # Sv
                        0,                                    # Sv to Iv
                        0,                                    # Iv
                        0,                                    # I to R
                        0 ]                                   # R
    return np.array(initial_state_k)


@numba.jit(nopython=True)
def vec_multinomial(L, prob_matrix, sizes, res):
    '''
    Optimized function for drawing L multinomial samples, each with different probabilities
    for SIRstep_vectorized() function
    '''
    for i in range(L):
        res[i] = np.random.multinomial(n = sizes[i], pvals = prob_matrix[i])
    return res 


def SIRstep_vectorized(L, current_states, capacities, demo_params, epid_params, fixed_epid_probas, thetas, 
                       eff_reduceinfec, eff_protect, simul, delta):
    '''Fonction of a SIR step given the current states of all herds: '''
    Snv, Inv = current_states[:, 0],  current_states[:, 2]
    Sv, Iv = current_states[:, 3],  current_states[:, 5]
    R =  current_states[:, 7]

    N = Snv + Inv + Sv + Iv + R

    betas_Inv = epid_params[:,0]
    taus = demo_params[:, 1]
    betas_Iv = betas_Inv * (1-eff_reduceinfec) 
    
    # Fixed epidemic probabilities
    p_B, p_I, p_R = fixed_epid_probas
    
    # Probabilities that change:
    
    # Probas for SNV
    transmi_rate = betas_Inv*(Inv) + betas_Iv*(Iv)
    lambds = np.divide(transmi_rate ,N, out=np.zeros_like(transmi_rate), where=N!=0)

    Snv_rates = lambds + taus + thetas
    p_SnvInv = (1.0 - np.exp(-Snv_rates* delta))*lambds/Snv_rates 
    p_SnvD = (1.0 - np.exp(-Snv_rates * delta))*taus/Snv_rates
    p_Snvout = (1.0 - np.exp(-Snv_rates * delta))*thetas/Snv_rates

    # Probas for SV
    lambds_v = (1-eff_protect) * lambds
    lambds_v [np.isnan(lambds_v)] = 0.

    Sv_rates = lambds_v + taus + thetas
    p_SvIv = (1.0 - np.exp(-Sv_rates * delta))*lambds_v/Sv_rates
    p_SvD = (1.0 - np.exp(-Sv_rates * delta))*taus/Sv_rates
    p_Svout = (1.0 - np.exp(-Sv_rates * delta))*thetas/Sv_rates 

    #Add the probabilities
    p_Snv = np.array([p_SnvInv, p_SnvD, p_Snvout, 1.0-(p_SnvInv + p_SnvD + p_Snvout)]).T 
    p_Sv = np.array([p_SvIv, p_SvD, p_Svout, 1.0-(p_SvIv + p_SvD + p_Svout)]).T 
    
    # Draw from multinomials for each compartment:
    B_sample = vec_multinomial(L, prob_matrix = p_B, sizes = N.astype(int), res = np.zeros(shape=(L,2)))
    Snv_sample = vec_multinomial(L, prob_matrix = p_Snv, sizes = Snv.astype(int), res = np.zeros(shape=(L,4)))
    Sv_sample = vec_multinomial(L, prob_matrix = p_Sv, sizes = Sv.astype(int), res = np.zeros(shape=(L,4)))
    Inv_sample = vec_multinomial(L, prob_matrix = p_I, sizes = Inv.astype(int), res = np.zeros(shape=(L,4)))
    Iv_sample = vec_multinomial(L, prob_matrix = p_I, sizes = Iv.astype(int),res = np.zeros(shape=(L,4)))
    R_sample = vec_multinomial(L, prob_matrix = p_R, sizes = R.astype(int), res = np.zeros(shape=(L,3)))
    
    #Add samples and update counts in compartments:
    d_SnvI, d_SvI, d_InvR, d_IvR = Snv_sample[:, 0], Sv_sample[:,0], Inv_sample[:,0], Iv_sample[:,0]
    births =  B_sample[:,0] 
    conditioned_births = (capacities - N > 0) * np.minimum(births, capacities - N)  # Actual births are limited by herd capacity 
    Snv = Snv_sample[:,3] + conditioned_births
    Sv = Sv_sample[:,3]
    Inv = Inv_sample[:,3] + d_SnvI
    Iv = Iv_sample[:,3] + d_SvI
    R = R_sample[:,2] + d_InvR + d_IvR
    Snv_out, Inv_out, Sv_out, Iv_out = Snv_sample[:,2], Inv_sample[:,2], Sv_sample[:,2], Iv_sample[:,2]
    R_out = R_sample[:,1]
    
    # Return list of two arrays: current state, and exports by compartment.
    return np.array([Snv, d_SnvI, Inv, Sv, d_SvI, Iv, d_InvR + d_IvR, R]).T.astype(int), np.array([Snv_out, Inv_out, Sv_out, Iv_out, R_out]).T.astype(int)


@numba.jit(nopython=True)
def vec_exports_i(out_k, p_out_k):
    '''Optimized step fonction for assigning exports '''
    nb_neighb = len(p_out_k)
    res_k = np.zeros((5,nb_neighb))
    for i in range(5):
        res_k[i] = np.random.multinomial(out_k[i], p_out_k)
    return res_k.T


def vec_exports(L, thetas, probs_exports, outs):
    '''Optimized full fonction for assigning exports '''
    res = []
    for k in range(L):
        theta_k, p_out_k = thetas[k], probs_exports[k]
        if theta_k != 0:
            res_k = vec_exports_i(outs[k], p_out_k)
            res.append(res_k)
    return res



#SCORE FUNCTIONS


#Topological

@numba.jit(nopython=True)
def in_rate(current_states, theta_edges, L, *args):
    a = []
    for i in range(L):
        a.append(np.sum(theta_edges[theta_edges[:,0] == i , 2]))
    in_rate = np.array(a)
    return in_rate
@numba.jit(nopython=True)
def out_rate(current_states,theta_edges, L, *args):
    a = []
    for i in range(L):
        a.append(np.sum(theta_edges[theta_edges[:,1] == i , 2]))
    out_rate = np.array(a)
    return out_rate

def pagerank(current_states,theta_edges,L, *args):
    lista = theta_edges[:,:2].astype(int).tolist()
    g = igraph.Graph(n=L, edges=lista, directed=True, edge_attrs={'weight': theta_edges[:,2]})
    pagerank_scores = g.pagerank(directed = True, weights=g.es["weight"])
    return np.array(pagerank_scores)

def betweeness(current_states,theta_edges,L, *args):
    lista = theta_edges[:,:2].astype(int).tolist()
    g = igraph.Graph(n=L, edges=lista, directed=True, edge_attrs={'weight': theta_edges[:,2]})
    betweenness = g.betweenness(directed=True, weights=g.es["weight"])
    return np.array(betweenness)

def closeness(current_states,theta_edges,L, *args):
    lista = theta_edges[:,:2].astype(int).tolist()
    g = igraph.Graph(n=L, edges=lista, directed=True, edge_attrs={'weight': theta_edges[:,2]})
    closeness = g.closeness(weights=g.es["weight"])
    return np.array(closeness)

#Demographic

def herds_sizes(current_states, theta_edges, L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1) 
    return herds_sizes

def ventes_periode(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, *args):
    return cum_ventes_periode

def achats_periode(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, *args):
    return cum_achats_periode

def ventes_cumulees(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, cum_ventes, *args):
    return cum_ventes

def achats_cumulees(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, cum_ventes, cum_achats, *args):
    return cum_achats



#Epidemiological

def susceptible_proportion(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  
    susceptible_animals = current_states[:,0] + current_states[:,3]  
    return susceptible_animals/herds_sizes

def infected_proportion(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  
    infected_animals = current_states[:,2] + current_states[:,5]  
    return infected_animals/herds_sizes

def recovered(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  
    recovered_animals = current_states[:,7] 
    return recovered_animals/herds_sizes

def delta_recovered(current_states, theta_edges,L,last_states, *args):
    herds_sizes_now = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  
    recovered_animals_now = current_states[:,7] 
    herds_sizes_last = np.sum(np.delete(last_states, (1, 4, 6), 1), axis = 1)  
    recovered_animals_last = last_states[:,7] 
    return (recovered_animals_now/herds_sizes_now) - (recovered_animals_last/herds_sizes_last)

def delta_infected(current_states, theta_edges,L,last_states, *args):
    herds_sizes_now = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  
    infected_animals_now = current_states[:,2] + current_states[:,5] 
    herds_sizes_last = np.sum(np.delete(last_states, (1, 4, 6), 1), axis = 1)  
    infected_animals_last = last_states[:,2] + last_states[:,5] 
    delta_I = (infected_animals_now/herds_sizes_now) - (infected_animals_last/herds_sizes_last)
    return delta_I


#Greedy

def potentiel(current_states, theta_edges,L, *args):
    '''Greedy infanimals for vaccination'''
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  
    susceptible_animals = current_states[:,0] + current_states[:,3]  
    infected_animals = current_states[:,2] + current_states[:,5] 
    return susceptible_animals*infected_animals/herds_sizes


def opti_infherds(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, cum_ventes, cum_achats, gamma, tau, *args):
    '''Greedy infherds for vaccination'''
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  
    susceptible_animals = current_states[:,0] + current_states[:,3]  
    infected_animals = current_states[:,2] + current_states[:,5] 
    infected_indicator_one = infected_animals == 1 
    infected_indicator = infected_animals >0
    healthy_indicator = infected_animals==0
    a = []
    b = []
    for j in range(L):
        edges_jparent= theta_edges[theta_edges[:,0] == j]
        children_j = edges_jparent[:, 1] 
        thetaji =  edges_jparent[:, 2]
        susceptible_animals_childrenj = susceptible_animals[children_j.astype(int)]
        herds_sizes_childrenj = herds_sizes[children_j.astype(int)]
        healthy_indicator_childrenj = infected_animals[children_j.astype(int)] == 0
        a.append(np.sum(thetaji))
        infected_childrenj = infected_animals[children_j.astype(int)]
        b.append(np.sum(thetaji*healthy_indicator_childrenj))
    A = np.array(a)
    B = np.array(b)
    return ((gamma+tau+A)*infected_indicator_one + B)*infected_animals*susceptible_animals/herds_sizes  


def opti_infherds_seuil(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, cum_ventes, cum_achats, gamma, tau, *args):
    '''Greedy infherds soft for vaccination'''
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  
    susceptible_animals = current_states[:,0] + current_states[:,3]  
    infected_animals = current_states[:,2] + current_states[:,5] 
    infected_indicator_one = (infected_animals > 0) * (infected_animals < 20)
    infected_indicator = infected_animals >0
    healthy_indicator = infected_animals==0
    a = []
    b = []
    for j in range(L):
        edges_jparent= theta_edges[theta_edges[:,0] == j]
        children_j = edges_jparent[:, 1] 
        thetaji =  edges_jparent[:, 2]
        susceptible_animals_childrenj = susceptible_animals[children_j.astype(int)]
        herds_sizes_childrenj = herds_sizes[children_j.astype(int)]
        healthy_indicator_childrenj = infected_animals[children_j.astype(int)] == 0
        a.append(np.sum(thetaji))
        infected_childrenj = infected_animals[children_j.astype(int)]
        b.append(np.sum(thetaji*healthy_indicator_childrenj))
    A = np.array(a)
    B = np.array(b)
    return ((gamma+tau+A)*infected_indicator_one + B)*infected_animals*susceptible_animals/herds_sizes  



# Random score

def random_scoring(current_states, theta_edges,L, *args):
    random_scores = np.random.randint(L, size=L)
    return random_scores

#Others (unused)
@numba.jit(nopython=True)
def in_degree(current_states, theta_edges, L, *args):
    a = []
    for i in range(L):
        a.append(np.sum(theta_edges[:,1] == i))
    in_deg = np.array(a)
    return in_deg
@numba.jit(nopython=True)
def out_degree(current_states, theta_edges, L, *args):
    b = []
    for i in range(L):
        b.append(np.sum(theta_edges[:,0] == i))
    out_deg = np.array(b)
    return out_deg

def eigenvector(current_states,theta_edges,L, *args):
    lista = theta_edges[:,:2].astype(int).tolist()
    g = igraph.Graph(n=L, edges=lista, directed=True, edge_attrs={'weight': theta_edges[:,2]})
    eigenvector = g.eigenvector_centrality(directed=True, weights=g.es["weight"])
    return np.array(eigenvector)

def infectedindicator(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  
    susceptible_animals = current_states[:,0] + current_states[:,3]  
    infected_animals = current_states[:,2] + current_states[:,5] 
    infected_indicator = infected_animals>0 
    return infected_indicator

def opti_infherds_plusone(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, cum_ventes, cum_achats, gamma, tau, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  
    susceptible_animals = current_states[:,0] + current_states[:,3]  
    infected_animals = current_states[:,2] + current_states[:,5] 
    infected_indicator_one = infected_animals == 1 
    infected_indicator = infected_animals >0
    healthy_indicator = infected_animals==0
    a = []
    b = []
    for j in range(L):
        edges_jparent= theta_edges[theta_edges[:,0] == j]
        children_j = edges_jparent[:, 1] 
        thetaji =  edges_jparent[:, 2]
        susceptible_animals_childrenj = susceptible_animals[children_j.astype(int)]
        herds_sizes_childrenj = herds_sizes[children_j.astype(int)]
        healthy_indicator_childrenj = infected_animals[children_j.astype(int)] == 0
        a.append(np.sum(thetaji))
        infected_childrenj = infected_animals[children_j.astype(int)]
        b.append(np.sum(thetaji*healthy_indicator_childrenj))
    A = np.array(a)
    B = np.array(b)
    return ( 1 + (gamma+tau+A)*infected_indicator_one + B)*infected_animals*susceptible_animals/herds_sizes 


def opti_infherds_modified(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, cum_ventes, cum_achats, gamma, tau, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  
    susceptible_animals = current_states[:,0] + current_states[:,3]  
    infected_animals = current_states[:,2] + current_states[:,5] 
    infected_indicator_one = infected_animals == 1 
    infected_indicator = infected_animals >0
    healthy_indicator = infected_animals==0
    a = []
    b = []
    for j in range(L):
        edges_jparent= theta_edges[theta_edges[:,0] == j]
        children_j = edges_jparent[:, 1] 
        thetaji =  edges_jparent[:, 2]
        susceptible_animals_childrenj = susceptible_animals[children_j.astype(int)]
        herds_sizes_childrenj = herds_sizes[children_j.astype(int)]
        healthy_indicator_childrenj = infected_animals[children_j.astype(int)] == 0
        
        a.append(np.sum(thetaji))
        
        infected_childrenj = infected_animals[children_j.astype(int)]
        inverse_infected_childrenj = np.divide(np.ones(len(children_j)),infected_childrenj,
                                               out=np.zeros_like(np.ones(len(children_j))),
                                               where=infected_childrenj!=0)
        b.append(np.sum(thetaji*inverse_infected_childrenj ))
    A = np.array(a)
    B = np.array(b)
    inverse_infected = np.divide(np.ones(L),infected_animals,
                                 out=np.zeros_like(np.ones(L)),
                                 where=infected_animals!=0)
    return ( (gamma+tau+A)*inverse_infected + B)*infected_animals*susceptible_animals/herds_sizes

def LRIE(current_states,theta_edges,L, *args):  
    infected_animals = current_states[:,2] + current_states[:,5]  
    a = []
    b = []
    for j in range(L):
        edges_jchild = theta_edges[theta_edges[:,1] == j]
        parents_j = edges_jchild[:, 0]
        thetaij = edges_jchild[:, 2]
        indicator_I_parents_j = infected_animals[parents_j.astype(int)] > 0
        a.append(np.sum(thetaij*indicator_I_parents_j))

        edges_jparent= theta_edges[theta_edges[:,0] == j]
        children_j = edges_jparent[:, 1] 
        thetaji =  edges_jparent[:, 2]
        healthy_indicator_childrenj = infected_animals[children_j.astype(int)] == 0
        b.append(np.sum(thetaji*healthy_indicator_childrenj))
    LIN = np.array(a)
    MSN = np.array(b)
    return  MSN - LIN

def weighted_LRIE(current_states,theta_edges,L, *args):
    infected_animals = current_states[:,2] + current_states[:,5]
    susceptible_animals = current_states[:,0] + current_states[:,3]
    a = []
    b = []
    for j in range(L):
        edges_jchild = theta_edges[theta_edges[:,1] == j]
        parents_j = edges_jchild[:, 0]
        thetaij = edges_jchild[:, 2]
        I_parents_j = infected_animals[parents_j.astype(int)]
        a.append(np.sum(thetaij*I_parents_j))

        edges_jparent= theta_edges[theta_edges[:,0] == j]
        children_j = edges_jparent[:, 1] 
        thetaji =  edges_jparent[:, 2]
        susceptible_animals_childrenj = susceptible_animals[children_j.astype(int)]
        healthy_indicator_childrenj = infected_animals[children_j.astype(int)] == 0
        b.append(np.sum(thetaji*healthy_indicator_childrenj*susceptible_animals_childrenj))
    LIN = np.array(a)
    MSN = np.array(b)
    return MSN - LIN

def delta_newrec(current_states, theta_edges,L,last_states, cum_newinf_lastnow,cum_newrec_lastnow, *args):
    return cum_newrec_lastnow


def delta_newinf(current_states, theta_edges,L,last_states, cum_newinf_lastnow, *args):
    return cum_newinf_lastnow
	

######


# Resource Allocation function, as a function of the computed scores
def resource_allocation_function(computed_scores, budget, L, dynamic_score, current_states, social_scoring):
    b = np.random.random(computed_scores.size) #random for shuffle
    ordered_indices = np.flipud(np.lexsort((b,computed_scores)))
    susceptible_animals = current_states[:,0] + current_states[:,3]
    infected_animals = current_states[:,2] + current_states[:,5]
    
    ordered_susceptible_animals = susceptible_animals[ordered_indices]
    ordered_indices = ordered_indices[ordered_susceptible_animals > 0]
    
    vacc_social = np.zeros(L)
    recovered_animals = current_states[:,7]
    sizes = susceptible_animals+infected_animals+recovered_animals
    cumsum_sizes = np.cumsum(sizes[ordered_indices])
    conditioned = cumsum_sizes <= budget
    chosen_herds = ordered_indices[conditioned]
    vacc_social[chosen_herds] = 1
    return vacc_social
   
######

def path(gamma, tau, initial_states, demo_params, epid_params, fixed_epid_probas,
         neighbors_list, parents_list, probs_exports, social_duration, budget, eff_reduceinfec, eff_protect,
         thetas, delta, nb_steps, theta_edges_compact, theta_edges,
         social_scoring = 'herds_sizes', computed_out_going_chain = None, 
         computed_in_going_chain = None):
    
    '''Fonction for epidemic-decision path for all herds'''
    
    #Initialization
    
    L = len(initial_states)
    all_states = np.zeros((nb_steps, L, 8), dtype=int) 
    all_states[0] = np.copy(initial_states) #Snv, SnvI, Inv , Sv, SvI, Iv , IR, R
    ventes_byfarm = np.zeros((nb_steps, L))
    achats_byfarm = np.zeros((nb_steps, L))
    capacities = np.sum(initial_states, axis = 1)*1.5
    
    # Simulation times
    simul_list = np.array(range(0, nb_steps))
    
    # Social planner decision times
    social_decision_times = simul_list[np.mod(simul_list*delta, social_duration) == 0.0]
    social_decision_times = social_decision_times[1:]
    social_decision_times = social_decision_times[6:]
    social_decisions = np.zeros((nb_steps, L), dtype=int)
        
    #Choose social scoring
    if social_scoring == 'herds_sizes':    
        social_scoring_function = herds_sizes 
        dynamic_score = 'no'
    elif social_scoring == 'susceptible_proportion':    
        social_scoring_function = susceptible_proportion
        dynamic_score = 'yes'
    elif social_scoring == 'infectedindicator':    
        social_scoring_function = infectedindicator
        dynamic_score = 'yes'
    elif social_scoring == 'infected_proportion':    
        social_scoring_function = infected_proportion
        dynamic_score = 'yes'
    elif social_scoring == 'recovered':    
        social_scoring_function = recovered
        dynamic_score = 'yes'
    elif social_scoring == 'LRIE':    
        social_scoring_function = LRIE
        dynamic_score = 'yes'
    elif social_scoring == 'weighted_LRIE':    
        social_scoring_function = weighted_LRIE
        dynamic_score = 'yes'
    elif social_scoring == 'random_scoring':    
        social_scoring_function = random_scoring
        dynamic_score = 'yes'
    elif social_scoring == 'potentiel':
        social_scoring_function = potentiel
        dynamic_score = 'yes'
    elif social_scoring == 'opti_infherds':
        social_scoring_function = opti_infherds
        dynamic_score = 'yes'
    elif social_scoring == 'opti_infherds_seuil':
        social_scoring_function = opti_infherds_seuil
        dynamic_score = 'yes'
    elif social_scoring == 'opti_infherds_plusone':
        social_scoring_function = opti_infherds_plusone
        dynamic_score = 'yes'
    elif social_scoring == 'opti_infherds_modified':
        social_scoring_function = opti_infherds_modified
        dynamic_score = 'yes'
    elif social_scoring == 'in_degree':
        social_scoring_function = in_degree
        dynamic_score = 'no'
    elif social_scoring == 'out_degree':
        social_scoring_function = out_degree
        dynamic_score = 'no'
    elif social_scoring == 'in_rate':
        social_scoring_function = in_rate
        dynamic_score = 'no'
    elif social_scoring == 'out_rate':
        social_scoring_function = out_rate
        dynamic_score = 'no'
    elif social_scoring == 'delta_recovered':
        social_scoring_function = delta_recovered
        dynamic_score = 'yes'
    elif social_scoring == 'delta_infected':
        social_scoring_function = delta_infected
        dynamic_score = 'yes'
    elif social_scoring == 'delta_newinf':
        social_scoring_function = delta_newinf
        dynamic_score = 'yes'
    elif social_scoring == 'delta_newrec':
        social_scoring_function = delta_newrec
        dynamic_score = 'yes'
    elif social_scoring == 'pagerank':
        social_scoring_function = pagerank
        dynamic_score = 'no'
    elif social_scoring == 'betweeness':
        social_scoring_function = betweeness
        dynamic_score = 'no'
    elif social_scoring == 'eigenvector':
        social_scoring_function = eigenvector
        dynamic_score = 'no'
    elif social_scoring == 'closeness':
        social_scoring_function = closeness
        dynamic_score = 'no' 
    elif social_scoring == 'ventes_periode':
        social_scoring_function = ventes_periode
        dynamic_score = 'yes'  
    elif social_scoring == 'achats_periode':
        social_scoring_function = achats_periode
        dynamic_score = 'yes'  
    elif social_scoring == 'ventes_cumulees':
        social_scoring_function = ventes_cumulees
        dynamic_score = 'yes'  
    elif social_scoring == 'achats_cumulees':
        social_scoring_function = achats_cumulees
        dynamic_score = 'yes'  
        
    if dynamic_score == 'no':
        computed_scores = social_scoring_function(initial_states, theta_edges,L, 
                                                  computed_out_going_chain, computed_in_going_chain)
    
    #Evolution du path
    
    total_actual_movements = []
    all_scores = []

    for simul in range(1, nb_steps):
        
        current_states = np.copy(all_states[simul-1])
        sizes[simul-1] = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)
            
        # Decision if simul is a social planner decision moment
        if simul in social_decision_times:
            
            #states of previous decision
            if simul != social_decision_times[0]:
                time_social_prev_decision = social_decision_times[np.where(social_decision_times == simul)[0][0] - 1]
            else:
                time_social_prev_decision = 0
            last_states = all_states[time_social_prev_decision]
            cum_newinf_lastnow = np.sum(np.sum(all_states[time_social_prev_decision: simul, :, [1, 4]], axis= 2), axis = 0)
            cum_newrec_lastnow = np.sum(all_states[time_social_prev_decision: simul, :, 6], axis = 0)
            
            cum_ventes =  np.sum(ventes_byfarm[:simul, :], axis = 0)
            cum_achats =  np.sum(achats_byfarm[:simul, :], axis = 0)
            
            cum_ventes_periode =  np.sum(ventes_byfarm[time_social_prev_decision:simul, :], axis = 0)
            cum_achats_periode =  np.sum(achats_byfarm[time_social_prev_decision:simul, :], axis = 0)
           
            # Social planner computes scores according to scoring function
            if dynamic_score == 'yes':
                computed_scores = social_scoring_function(current_states, theta_edges, L, last_states, cum_newinf_lastnow,
                                                          cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode,
                                                          cum_ventes, cum_achats, gamma, tau)
            
            all_scores.append(computed_scores)
            
            
            #Social planner makes decision
            social_decisions[simul] = resource_allocation_function(computed_scores, budget, L, dynamic_score, current_states, social_scoring)
            
        ###################################################################
        #Change states
        prev_N = np.sum(current_states, axis = 1)
        current_states, outs = SIRstep_vectorized(L, current_states, capacities,
                                                  demo_params, epid_params, fixed_epid_probas, thetas,
                                                  eff_reduceinfec, eff_protect, simul, delta)
        ventes_byfarm[simul] = np.sum(outs, axis = 1)
        ###################################################################  
        #Assign exports
        exports = np.concatenate(vec_exports(L, thetas, probs_exports, outs))
        ####################################################################
    
        #Assign exports as imports
        
        open_neighbors_indicator = ((capacities- prev_N)[neighbors_list] > 0)
        
        imports =[]
        returns = []
        true_exports = np.copy(exports)
        for c in range(0, 5):
            souhait = ((capacities- prev_N)[neighbors_list])
            weights = open_neighbors_indicator* list(map(min, exports[:,c], souhait))
            unsold = exports[:,c] - weights
            true_exports[:,c] = np.copy(weights)
            imports.append(np.bincount(neighbors_list, weights=weights))
            returns.append(np.bincount(parents_list, weights=unsold))
        
        imports = np.array(imports).T 
        modif_imports = np.insert(imports, [1,3,4], 0, axis=1)
        
        returns = np.array(returns).T
        modif_returns = np.insert(returns , [1,3,4], 0, axis=1)
            
        all_states[simul] = current_states + modif_imports + modif_returns
                           
        achats_byfarm[simul] = np.sum(modif_imports, axis = 1)
        ventes_byfarm[simul] = ventes_byfarm[simul] - np.sum(modif_returns, axis = 1)
        
    return social_decision_times, social_decisions, all_states, ventes_byfarm, achats_byfarm, all_scores

