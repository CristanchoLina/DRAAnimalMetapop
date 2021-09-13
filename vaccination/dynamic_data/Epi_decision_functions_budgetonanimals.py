#!/usr/bin/env python
# coding: utf-8

# Import modules
import numpy as np
import random
import numba
import igraph
import pandas as pd

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


#### Epidemic functions definition ####

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

# Functions for defining the farmers decision farmers_mechanism

def nothing(simul, L, *args):
    '''Farmers never vaccinate'''
    return np.zeros(L)

def always(simul, L, *args):
    '''All Farmers vaccinate'''
    return np.ones(L)

def draw(weights):
    '''Fonction for drawing an option according to weights'''
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0
    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1

def log_exp_Sum(log_weights):
    '''Log of the sum of the exponential Log weights'''
    exp_logw = []
    for log_w in log_weights:
        exp_logw.append(np.exp(log_w))
    return np.log(sum(exp_logw))

def expw(simul, L, mean_rewards, counts, vaccinators, non_vaccinators, relat_reward, farmers_decision_times,
         log_weights, kappas, *args):
    '''Expw farmers_mechanism'''
    #update log weights
    log_weights[vaccinators, 1] += relat_reward[vaccinators] * kappas[vaccinators]
    log_weights[non_vaccinators, 0] += relat_reward[non_vaccinators] * kappas[non_vaccinators]
 
    farmers_decisions = np.zeros(L)
    for k in range(0,L):
        log_exp_Sum_k = np.logaddexp(log_weights[k][0], log_weights[k][1])  #float(log_exp_Sum(log_weights[k]))
        probabilityDistribution_k = tuple((np.exp(w - log_exp_Sum_k)) for w in log_weights[k])
        farmers_decisions[k] = draw(probabilityDistribution_k)
    return farmers_decisions

# NOTE: prev decision is farmers or social planner????? VERIFIER 

def neighb_expw(simul, L, mean_rewards, counts, vaccinators, non_vaccinators, relat_reward, farmers_decision_times,
         log_weights, kappas, rhos, farmers_prev_decision, theta_edges_compact, *args):
    
    '''Neigh_expw farmers_mechanism'''
    
    for k in range(0,L):
        # List of neighbors of k:
        neighbors_k = theta_edges_compact[theta_edges_compact[:, 0] == k, 1][0]
        if neighbors_k != []:
            #Choose a random neighbor
            neighbor = np.random.choice(neighbors_k)
            neigh_prev_decision = farmers_prev_decision[neighbor]
            neigh_reward = relat_reward[neighbor]
            log_weights[k][neigh_prev_decision] += neigh_reward* rhos[k]
    
    #update log weights
    log_weights[vaccinators, 1] += relat_reward[vaccinators] * kappas[vaccinators]
    log_weights[non_vaccinators, 0] += relat_reward[non_vaccinators] * kappas[non_vaccinators]
    
    farmers_decisions = np.zeros(L)
    for k in range(0,L):
        log_exp_Sum_k = np.logaddexp(log_weights[k][0], log_weights[k][1]) #float(log_exp_Sum(log_weights[k]))
        probabilityDistribution_k = tuple((np.exp(w - log_exp_Sum_k)) for w in log_weights[k])
        farmers_decisions[k] = draw(probabilityDistribution_k)
    return farmers_decisions


@numba.jit(nopython=True)
def vaccinate(current_states, vacc):
    '''Optimized function for applying the decision: SNV pass to SV, or SV pass to SNV'''
    vaccinators = np.where(vacc == 1)[0]
    non_vaccinators = np.where(vacc != 1)[0]
    
    states_vaccinators = np.copy(current_states[vaccinators,:])
    current_states[vaccinators,3] = states_vaccinators[:,0] + states_vaccinators[:,3]
    current_states[vaccinators,0] = np.zeros(len(vaccinators))
    
    states_non_vaccinators = np.copy(current_states[non_vaccinators,:])
    current_states[non_vaccinators,0] = states_non_vaccinators[:,0] + states_non_vaccinators[:,3]
    current_states[non_vaccinators,3] = np.zeros(len(non_vaccinators))
        
    return current_states

#SCORE FUNCTIONS QUI NE DEPEND PAS DE LETAT EPI

# Score: size of the herd 
def herds_sizes(current_states, theta_edges, L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    return herds_sizes
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
@numba.jit(nopython=True)
def rapport_out_degree_in_degree(current_states, theta_edges, L, *args):
    a = []
    b = []
    for i in range(L):
        a.append(np.sum(theta_edges[:,1] == i))
        b.append(np.sum(theta_edges[:,0] == i))
    in_deg = np.array(a)
    out_deg = np.array(b)
    return out_deg/in_deg
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
@numba.jit(nopython=True)
def rapport_out_rate_in_rate(current_states,theta_edges,L, *args):
    a = []
    b = []
    for i in range(L):
        a.append(np.sum(theta_edges[theta_edges[:,0] == i , 2]))
        b.append(np.sum(theta_edges[theta_edges[:,1] == i , 2]))
    in_rate = np.array(a)
    out_rate = np.array(b)
    return out_rate/in_rate

@numba.jit(nopython=True)
def inverse_rapport_out_rate_in_rate(current_states,theta_edges,L, *args):
    a = []
    b = []
    for i in range(L):
        a.append(np.sum(theta_edges[theta_edges[:,0] == i , 2]))
        b.append(np.sum(theta_edges[theta_edges[:,1] == i , 2]))
    in_rate = np.array(a)
    out_rate = np.array(b)
    return in_rate/out_rate

def pagerank(current_states,theta_edges,L, *args):
    #g = igraph.Graph.TupleList(theta_edges, directed = True,weights=True)
    lista = theta_edges[:,:2].astype(int).tolist()
    g = igraph.Graph(n=L, edges=lista, directed=True, edge_attrs={'weight': theta_edges[:,2]})
    pagerank_scores = g.pagerank(directed = True, weights=g.es["weight"])
    return np.array(pagerank_scores)

def betweeness(current_states,theta_edges,L, *args):
    #g = igraph.Graph.TupleList(theta_edges, directed = True,weights=True)
    lista = theta_edges[:,:2].astype(int).tolist()
    g = igraph.Graph(n=L, edges=lista, directed=True, edge_attrs={'weight': theta_edges[:,2]})
    betweenness = g.betweenness(directed=True, weights=g.es["weight"])
    return np.array(betweenness)

def eigenvector(current_states,theta_edges,L, *args):
    #g = igraph.Graph.TupleList(theta_edges, directed = True,weights=True)
    lista = theta_edges[:,:2].astype(int).tolist()
    g = igraph.Graph(n=L, edges=lista, directed=True, edge_attrs={'weight': theta_edges[:,2]})
    eigenvector = g.eigenvector_centrality(directed=True, weights=g.es["weight"])
    return np.array(eigenvector)

def closeness(current_states,theta_edges,L, *args):
    #g = igraph.Graph.TupleList(theta_edges, directed = True,weights=True)
    lista = theta_edges[:,:2].astype(int).tolist()
    g = igraph.Graph(n=L, edges=lista, directed=True, edge_attrs={'weight': theta_edges[:,2]})
    closeness = g.closeness(weights=g.es["weight"])
    return np.array(closeness)

def out_going_chain(current_states,theta_edges,L, out_going_chain, *args):
    return out_going_chain

def in_going_chain(current_states,theta_edges,L, out_going_chain, in_going_chain, *args):
    return in_going_chain


# SCORE FUNCTIONS AS A FUNCTION OF CURRENT STATES

def random_scoring(current_states, theta_edges,L, *args):
    random_scores = np.random.randint(L, size=L)
    return random_scores

def susceptible_proportion(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    susceptible_animals = current_states[:,0] + current_states[:,3]  
    return susceptible_animals/herds_sizes

def infected_proportion(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    infected_animals = current_states[:,2] + current_states[:,5]  
    return infected_animals/herds_sizes

def potentiel(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    susceptible_animals = current_states[:,0] + current_states[:,3]  
    infected_animals = current_states[:,2] + current_states[:,5] 
    return susceptible_animals*infected_animals/herds_sizes


def infectedindicator(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    susceptible_animals = current_states[:,0] + current_states[:,3]  
    infected_animals = current_states[:,2] + current_states[:,5] 
    infected_indicator = infected_animals>0 #==1
    return infected_indicator#MSN - LIN #+# susceptible_animals*infected_indicator_one/herds_sizes +


def infectedindicator_experiences(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    susceptible_animals = current_states[:,0] + current_states[:,3]  
    infected_animals = current_states[:,2] + current_states[:,5] 
    infected_indicator_one = infected_animals>0 #==1
    healthy_indicator = infected_animals==0
    a = []
    b = []
    for j in range(L):
        edges_jchild = theta_edges[theta_edges[:,1] == j]
        parents_j = edges_jchild[:, 0]
        thetaij = edges_jchild[:, 2]
        infected_animals_parents_j = infected_animals[parents_j.astype(int)]
        indicator_I_parents_j = infected_animals[parents_j.astype(int)]  > 0
        a.append(np.sum(thetaij*indicator_I_parents_j)*healthy_indicator[j]) # ET PEUT ETRE ICI

        edges_jparent= theta_edges[theta_edges[:,0] == j]
        children_j = edges_jparent[:, 1] 
        thetaji =  edges_jparent[:, 2]
        #I_j = infected_animals[j]
        susceptible_animals_childrenj = susceptible_animals[children_j.astype(int)]
        herds_sizes_childrenj = herds_sizes[children_j.astype(int)]
        healthy_indicator_childrenj = infected_animals[children_j.astype(int)] == 0
        b.append(np.sum(thetaji*healthy_indicator_childrenj)*infected_indicator_one[j]) ##### ICI EST LA DIFF AVEC LRIE
    LIN = np.array(a)
    MSN = np.array(b)
    return infected_indicator_one#MSN - LIN #+# susceptible_animals*infected_indicator_one/herds_sizes +

def opti_infherds(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, cum_ventes, cum_achats, gamma, tau, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
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
        #I_j = infected_animals[j]
        susceptible_animals_childrenj = susceptible_animals[children_j.astype(int)]
        herds_sizes_childrenj = herds_sizes[children_j.astype(int)]
        healthy_indicator_childrenj = infected_animals[children_j.astype(int)] == 0
        a.append(np.sum(thetaji))
        infected_childrenj = infected_animals[children_j.astype(int)]
        b.append(np.sum(thetaji*healthy_indicator_childrenj)) ##### ICI EST LA DIFF AVEC LRIE #*healthy_indicator_childrenj
    A = np.array(a)
    B = np.array(b)
    return ((gamma+tau+A)*infected_indicator_one + B)*infected_animals*susceptible_animals/herds_sizes # infected_animals*susceptible_animals/herds_sizes*(1+ 


def opti_infherds_seuil(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, cum_ventes, cum_achats, gamma, tau, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
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
        #I_j = infected_animals[j]
        susceptible_animals_childrenj = susceptible_animals[children_j.astype(int)]
        herds_sizes_childrenj = herds_sizes[children_j.astype(int)]
        healthy_indicator_childrenj = infected_animals[children_j.astype(int)] == 0
        a.append(np.sum(thetaji))
        infected_childrenj = infected_animals[children_j.astype(int)]
        b.append(np.sum(thetaji*healthy_indicator_childrenj)) ##### ICI EST LA DIFF AVEC LRIE #*healthy_indicator_childrenj
    A = np.array(a)
    B = np.array(b)
    return ((gamma+tau+A)*infected_indicator_one + B)*infected_animals*susceptible_animals/herds_sizes # infected_animals*susceptible_animals/herds_sizes*(1+ 


def opti_infherds_plusone(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, cum_ventes, cum_achats, gamma, tau, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
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
        #I_j = infected_animals[j]
        susceptible_animals_childrenj = susceptible_animals[children_j.astype(int)]
        herds_sizes_childrenj = herds_sizes[children_j.astype(int)]
        healthy_indicator_childrenj = infected_animals[children_j.astype(int)] == 0
        a.append(np.sum(thetaji))
        infected_childrenj = infected_animals[children_j.astype(int)]
        b.append(np.sum(thetaji*healthy_indicator_childrenj)) ##### ICI EST LA DIFF AVEC LRIE #*healthy_indicator_childrenj
    A = np.array(a)
    B = np.array(b)
    return ( 1 + (gamma+tau+A)*infected_indicator_one + B)*infected_animals*susceptible_animals/herds_sizes # infected_animals*susceptible_animals/herds_sizes*(1+ 


def opti_infherds_modified(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, cum_ventes, cum_achats, gamma, tau, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
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
        #I_j = infected_animals[j]
        susceptible_animals_childrenj = susceptible_animals[children_j.astype(int)]
        herds_sizes_childrenj = herds_sizes[children_j.astype(int)]
        healthy_indicator_childrenj = infected_animals[children_j.astype(int)] == 0
        
        a.append(np.sum(thetaji))
        
        infected_childrenj = infected_animals[children_j.astype(int)]
        inverse_infected_childrenj = np.divide(np.ones(len(children_j)),infected_childrenj,
                                               out=np.zeros_like(np.ones(len(children_j))),
                                               where=infected_childrenj!=0)
        b.append(np.sum(thetaji*inverse_infected_childrenj )) ##### ICI EST LA DIFF AVEC LRIE #*healthy_indicator_childrenj
    A = np.array(a)
    B = np.array(b)
    inverse_infected = np.divide(np.ones(L),infected_animals,
                                 out=np.zeros_like(np.ones(L)),
                                 where=infected_animals!=0)
    return ( (gamma+tau+A)*inverse_infected + B)*infected_animals*susceptible_animals/herds_sizes # infected_animals*susceptible_animals/herds_sizes*(1+ 

def potentiel_linearcomb_opt(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals = current_states[:,7] 
    infected_animals = current_states[:,2] + current_states[:,5]
    susceptible_animals = current_states[:,0] + current_states[:,3]
    a = []
    b = []
    for j in range(L):
        edges_jchild = theta_edges[theta_edges[:,1] == j]
        parents_j = edges_jchild[:, 0]
        thetaij = edges_jchild[:, 2]
        S_j = susceptible_animals[j]
        N_j = herds_sizes[j]
        I_parents_j = infected_animals[parents_j.astype(int)]
        a.append(np.sum(thetaij*I_parents_j*S_j/N_j))

        edges_jparent= theta_edges[theta_edges[:,0] == j]
        children_j = edges_jparent[:, 1] 
        thetaji =  edges_jparent[:, 2]
        I_j = infected_animals[j]
        susceptible_animals_childrenj = susceptible_animals[children_j.astype(int)]
        herds_sizes_childrenj = herds_sizes[children_j.astype(int)]
        b.append(np.sum(thetaji*I_j*susceptible_animals_childrenj/herds_sizes_childrenj))
    LIN = np.array(a)
    MSN = np.array(b)
    return (infected_animals*susceptible_animals/herds_sizes) + MSN - LIN

def recovered(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals = current_states[:,7] 
    return recovered_animals/herds_sizes

def recovered_inrate(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals = current_states[:,7] 
    a = []
    b = []
    for i in range(L):
        a.append(np.sum(theta_edges[theta_edges[:,0] == i , 2]))
        b.append(np.sum(theta_edges[theta_edges[:,1] == i , 2]))
    in_rate = np.array(a)
    out_rate = np.array(b)
    return (recovered_animals/herds_sizes) * (in_rate)

def recovered_outrate(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals = current_states[:,7] 
    a = []
    b = []
    for i in range(L):
        a.append(np.sum(theta_edges[theta_edges[:,0] == i , 2]))
        b.append(np.sum(theta_edges[theta_edges[:,1] == i , 2]))
    in_rate = np.array(a)
    out_rate = np.array(b)
    return (recovered_animals/herds_sizes) * (out_rate)


def delta_recovered_outrate(current_states, theta_edges,L,last_states, *args):
    herds_sizes_now = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals_now = current_states[:,7] 
    herds_sizes_last = np.sum(np.delete(last_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals_last = last_states[:,7] 
    a = []
    b = []
    for i in range(L):
        a.append(np.sum(theta_edges[theta_edges[:,0] == i , 2]))
        b.append(np.sum(theta_edges[theta_edges[:,1] == i , 2]))
    in_rate = np.array(a)
    out_rate = np.array(b)
    return ((recovered_animals_now/herds_sizes_now) - (recovered_animals_last/herds_sizes_last)) * out_rate


def recovered_inrate_outrate(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals = current_states[:,7] 
    a = []
    b = []
    for i in range(L):
        a.append(np.sum(theta_edges[theta_edges[:,0] == i , 2]))
        b.append(np.sum(theta_edges[theta_edges[:,1] == i , 2]))
    in_rate = np.array(a)
    out_rate = np.array(b)
    return (recovered_animals/herds_sizes) * (in_rate* out_rate) # *100) + (out_rate*100)


####CHANGER
def recovered_outrate_over_inrate(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals = current_states[:,7] 
    a = []
    b = []
    for i in range(L):
        a.append(np.sum(theta_edges[theta_edges[:,0] == i , 2]))
        b.append(np.sum(theta_edges[theta_edges[:,1] == i , 2]))
    in_rate = np.array(a)
    out_rate = np.array(b)
    return (recovered_animals/herds_sizes) * (out_rate/in_rate) # *100) + (out_rate*100)

def recovered_maxinrateoutrate(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals = current_states[:,7] 
    a = []
    b = []
    for i in range(L):
        a.append(np.sum(theta_edges[theta_edges[:,0] == i , 2]))
        b.append(np.sum(theta_edges[theta_edges[:,1] == i , 2]))
    in_rate = np.array(a)
    out_rate = np.array(b)
    return (recovered_animals/herds_sizes) * np.maximum(in_rate, out_rate)

def LRIE(current_states,theta_edges,L, *args):
    # herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    # susceptible_animals = current_states[:,0] + current_states[:,3]  # TO CHANGE, sizes are already computed
    infected_animals = current_states[:,2] + current_states[:,5]  # TO CHANGE, sizes are already computed
    # potentiel = infected_animals*susceptible_animals/herds_sizes
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
        #I_j = infected_animals[j]
        #susceptible_animals_childrenj = susceptible_animals[children_j.astype(int)]
        #herds_sizes_childrenj = herds_sizes[children_j.astype(int)]
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
        #I_j = infected_animals[j]
        susceptible_animals_childrenj = susceptible_animals[children_j.astype(int)]
        #herds_sizes_childrenj = herds_sizes[children_j.astype(int)]
        healthy_indicator_childrenj = infected_animals[children_j.astype(int)] == 0
        b.append(np.sum(thetaji*healthy_indicator_childrenj*susceptible_animals_childrenj))
    LIN = np.array(a)
    MSN = np.array(b)
    return MSN - LIN

def recovered_plus_LRIE(current_states,theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals = current_states[:,7] 
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
        #I_j = infected_animals[j]
        #susceptible_animals_childrenj = susceptible_animals[children_j.astype(int)]
        #herds_sizes_childrenj = herds_sizes[children_j.astype(int)]
        healthy_indicator_childrenj = infected_animals[children_j.astype(int)] == 0
        b.append(np.sum(thetaji*healthy_indicator_childrenj))
    LIN = np.array(a)
    MSN = np.array(b)
    return (recovered_animals/herds_sizes) + MSN - LIN

def recovered_plus_weighted_LRIE(current_states,theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals = current_states[:,7] 
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
        #I_j = infected_animals[j]
        susceptible_animals_childrenj = susceptible_animals[children_j.astype(int)]
        #herds_sizes_childrenj = herds_sizes[children_j.astype(int)]
        healthy_indicator_childrenj = infected_animals[children_j.astype(int)] == 0
        b.append(np.sum(thetaji*healthy_indicator_childrenj*susceptible_animals_childrenj))
    LIN = np.array(a)
    MSN = np.array(b)
    return (recovered_animals/herds_sizes) + MSN - LIN

def recovered_linearcomb(current_states, theta_edges,L, *args):
    herds_sizes = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals = current_states[:,7] 
    infected_animals = current_states[:,2] + current_states[:,5]
    susceptible_animals = current_states[:,0] + current_states[:,3]
    rec_prop = (recovered_animals/herds_sizes)
    a = []
    b = []
    for j in range(L):
        edges_jchild = theta_edges[theta_edges[:,1] == j]
        parents_j = edges_jchild[:, 0]
        thetaij = edges_jchild[:, 2]
        rec_prop_parents_j = rec_prop[parents_j.astype(int)]
        a.append(np.sum(thetaij*rec_prop_parents_j))

        edges_jparent= theta_edges[theta_edges[:,0] == j]
        children_j = edges_jparent[:, 1] 
        thetaji =  edges_jparent[:, 2]
        rec_prop_childrenj = rec_prop[children_j.astype(int)]
        b.append(np.sum(thetaji*rec_prop_childrenj))
    LIN = np.array(a)
    MSN = np.array(b)
    return rec_prop + MSN - LIN

def delta_recovered(current_states, theta_edges,L,last_states, *args):
    herds_sizes_now = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals_now = current_states[:,7] 
    herds_sizes_last = np.sum(np.delete(last_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals_last = last_states[:,7] 
    return (recovered_animals_now/herds_sizes_now) - (recovered_animals_last/herds_sizes_last)

def delta_recovered_linearcomb(current_states, theta_edges,L,last_states, *args):
    herds_sizes_now = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals_now = current_states[:,7] 
    herds_sizes_last = np.sum(np.delete(last_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    recovered_animals_last = last_states[:,7]
    delta_R  = (recovered_animals_now/herds_sizes_now) - (recovered_animals_last/herds_sizes_last)
    a = []
    b = []
    for j in range(L):
        edges_jchild = theta_edges[theta_edges[:,1] == j]
        parents_j = edges_jchild[:, 0]
        thetaij = edges_jchild[:, 2]
        delta_R_parents_j = delta_R[parents_j.astype(int)]
        a.append(np.sum(thetaij*delta_R_parents_j))

        edges_jparent= theta_edges[theta_edges[:,0] == j]
        children_j = edges_jparent[:, 1] 
        thetaji =  edges_jparent[:, 2]
        delta_R_children_j = delta_R[children_j.astype(int)]
        b.append(np.sum(thetaji*delta_R_children_j))
    delta_R_parents_j = np.array(a)
    delta_R_children_j = np.array(b)

    return  delta_R  + delta_R_parents_j - delta_R_children_j


def delta_infected(current_states, theta_edges,L,last_states, *args):
    herds_sizes_now = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    infected_animals_now = current_states[:,2] + current_states[:,5] 
    herds_sizes_last = np.sum(np.delete(last_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    infected_animals_last = last_states[:,2] + last_states[:,5] 
    delta_I = (infected_animals_now/herds_sizes_now) - (infected_animals_last/herds_sizes_last)
    return delta_I

def delta_infected_linearcomb(current_states, theta_edges,L,last_states, *args):
    herds_sizes_now = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    infected_animals_now = current_states[:,2] + current_states[:,5] 
    herds_sizes_last = np.sum(np.delete(last_states, (1, 4, 6), 1), axis = 1)  # TO CHANGE, sizes are already computed
    infected_animals_last = last_states[:,2] + last_states[:,5] 
    delta_I = (infected_animals_now/herds_sizes_now) - (infected_animals_last/herds_sizes_last)
    a = []
    b = []
    for j in range(L):
        edges_jchild = theta_edges[theta_edges[:,1] == j]
        parents_j = edges_jchild[:, 0]
        thetaij = edges_jchild[:, 2]
        delta_I_parents_j = delta_I[parents_j.astype(int)]
        a.append(np.sum(thetaij*delta_I_parents_j))

        edges_jparent= theta_edges[theta_edges[:,0] == j]
        children_j = edges_jparent[:, 1] 
        thetaji =  edges_jparent[:, 2]
        delta_I_children_j = delta_I[children_j.astype(int)]
        b.append(np.sum(thetaji*delta_I_children_j))
    delta_I_parents_j = np.array(a)
    delta_I_children_j = np.array(b)

    return  delta_I  + delta_I_parents_j - delta_I_children_j

def delta_newrec(current_states, theta_edges,L,last_states, cum_newinf_lastnow,cum_newrec_lastnow, *args):
    return cum_newrec_lastnow

def delta_newrec_linearcomb(current_states, theta_edges,L,last_states, cum_newinf_lastnow, cum_newrec_lastnow, *args):
    a = []
    b = []
    for j in range(L):
        edges_jchild = theta_edges[theta_edges[:,1] == j]
        parents_j = edges_jchild[:, 0]
        thetaij = edges_jchild[:, 2]
        cum_newrec_lastnow_parents_j = cum_newrec_lastnow[parents_j.astype(int)]
        a.append(np.sum(thetaij*cum_newrec_lastnow_parents_j))

        edges_jparent= theta_edges[theta_edges[:,0] == j]
        children_j = edges_jparent[:, 1] 
        thetaji =  edges_jparent[:, 2]
        cum_newrec_lastnow_children_j = cum_newrec_lastnow[children_j.astype(int)]
        b.append(np.sum(thetaji*cum_newrec_lastnow_children_j))
    cum_newrec_lastnow_parents_j = np.array(a)
    cum_newrec_lastnow_children_j = np.array(b)
    return  cum_newrec_lastnow - (cum_newrec_lastnow_parents_j - cum_newrec_lastnow_children_j)

def delta_newinf(current_states, theta_edges,L,last_states, cum_newinf_lastnow, *args):
    return cum_newinf_lastnow

def delta_newinf_linearcomb(current_states, theta_edges,L,last_states, cum_newinf_lastnow, *args):
    a = []
    b = []
    for j in range(L):
        edges_jchild = theta_edges[theta_edges[:,1] == j]
        parents_j = edges_jchild[:, 0]
        thetaij = edges_jchild[:, 2]
        cum_newinf_lastnow_parents_j = cum_newinf_lastnow[parents_j.astype(int)]
        a.append(np.sum(thetaij*cum_newinf_lastnow_parents_j))

        edges_jparent= theta_edges[theta_edges[:,0] == j]
        children_j = edges_jparent[:, 1] 
        thetaji =  edges_jparent[:, 2]
        cum_newinf_lastnow_children_j = cum_newinf_lastnow[children_j.astype(int)]
        b.append(np.sum(thetaji*cum_newinf_lastnow_children_j))
    cum_newinf_lastnow_parents_j = np.array(a)
    cum_newinf_lastnow_children_j = np.array(b)

    return  cum_newinf_lastnow  + cum_newinf_lastnow_parents_j - cum_newinf_lastnow_children_j


def ventes_periode(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, *args):
    return cum_ventes_periode

def achats_periode(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, *args):
    return cum_achats_periode

def ventes_cumulees(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, cum_ventes, *args):
    return cum_ventes

def achats_cumulees(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, cum_ventes, cum_achats, *args):
    return cum_achats

def ventesmoinsachats_periode(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, *args):
    return cum_ventes_periode - cum_achats_periode

def ventesplusachats_periode(current_states, theta_edges, L, last_states, cum_newinf_lastnow, cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode, *args):
    return (0.75*cum_ventes_periode) + (0.25*cum_achats_periode)

######


# Resource Allocation function, as a function of the computed scores
def resource_allocation_function(computed_scores, budget, L, dynamic_score, current_states, social_scoring):
    b = np.random.random(computed_scores.size) #random for shuffle
    #ordered_indices = np.flipud(np.argsort(computed_scores))
    ordered_indices = np.flipud(np.lexsort((b,computed_scores)))
    susceptible_animals = current_states[:,0] + current_states[:,3]
    infected_animals = current_states[:,2] + current_states[:,5]
    #if dynamic_score == 'yes':
    
    ordered_susceptible_animals = susceptible_animals[ordered_indices]
    ordered_indices = ordered_indices[ordered_susceptible_animals > 0]
    
    vacc_social = np.zeros(L)
    recovered_animals = current_states[:,7]
    sizes = susceptible_animals+infected_animals+recovered_animals
    cumsum_sizes = np.cumsum(sizes[ordered_indices])
    conditioned = cumsum_sizes <= budget
    chosen_herds = ordered_indices[conditioned] # implicitely chooses the min(budget, nb of herds with suscep.
    vacc_social[chosen_herds] = 1
    return vacc_social
   
######

def path(gamma, tau, initial_states, demo_params, epid_params, eco_params, fixed_epid_probas,
         neighbors_list, parents_list, probs_exports, farmer_duration, social_duration, budget, eff_reduceinfec, eff_protect,
         thetas, delta, nb_steps, nexpw_params, theta_edges_compact, theta_edges,
         farmers_mechanism = 'neighb_expw', social_scoring = 'herds_sizes', computed_out_going_chain = None, 
         computed_in_going_chain = None):
    
    '''Fonction for epidemic-decision path for all herds'''
    
    #Initialization
    
    L = len(initial_states)
    all_states = np.zeros((nb_steps, L, 8), dtype=int) 
    all_states[0] = np.copy(initial_states) #Snv, SnvI, Inv , Sv, SvI, Iv , IR, R
    ventes_byfarm = np.zeros((nb_steps, L))
    achats_byfarm = np.zeros((nb_steps, L))
    capacities = np.sum(initial_states, axis = 1)*1.5
    
    # Economic costs
    r, phi, cu_vacc, cf_vacc = eco_params
    c_inf = phi*r
    
    # Simulation times
    simul_list = np.array(range(0, nb_steps))
    
    # Farmers decision times(fct of nb of steps and duration decision)
    farmers_decision_times = simul_list[np.mod(simul_list*delta, farmer_duration) == 0.0]
    farmers_decision_times = farmers_decision_times[1:]
    farmers_decisions = np.zeros((nb_steps, L), dtype=int)
    
    # Social planner decision times
    social_decision_times = simul_list[np.mod(simul_list*delta, social_duration) == 0.0]
    social_decision_times = social_decision_times[1:]
    social_decisions = np.zeros((nb_steps, L), dtype=int)
    
    # Farmers useful arrays 
    ###############################################
    #For expw strategy 
    sizes = np.zeros((nb_steps, L))
    counts = np.array([[0., 0.]] * L )
    mean_rewards = np.array([[0., 0.]] * L) 
    relat_reward = np.zeros(L)
    vaccinators, non_vaccinators = [], []
   
    init_proba_vacc, kappa, rho = nexpw_params
    kappas = np.array([kappa] * L) 
    #Convert probas to weights
    if init_proba_vacc == 1.:
        w_nv = 0.
        w_v = 1.   
    else: 
        w_v_w_nv = init_proba_vacc/(1.-init_proba_vacc)
        w_nv = 1.
        w_v = w_nv*w_v_w_nv
        
    log_weights = np.array([[np.log(w_nv), np.log(w_v)]] * L) #prob de pas vacc, prob de vacc
    
    #For neighb_expw strategy 
    rhos = np.array([rho] * L )
    farmers_prev_decision = np.zeros(L, dtype=int)
    ##########################################"
    
    #Choose farmers_mechanism
    if farmers_mechanism == 'nothing':    
        farmers_decision_function = nothing
    elif farmers_mechanism == 'always':    
        farmers_decision_function = always
    elif farmers_mechanism == 'neighb_expw':    
        farmers_decision_function = neighb_expw
        
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
    elif social_scoring == 'recovered_inrate_outrate':    
        social_scoring_function = recovered_inrate_outrate
        dynamic_score = 'yes'
    elif social_scoring == 'recovered_outrate_over_inrate':    
        social_scoring_function = recovered_outrate_over_inrate
        dynamic_score = 'yes'
    elif social_scoring == 'recovered_inrate':    
        social_scoring_function = recovered_inrate
        dynamic_score = 'yes'
    elif social_scoring == 'recovered_outrate':    
        social_scoring_function = recovered_outrate
        dynamic_score = 'yes'
    elif social_scoring == 'recovered_maxinrateoutrate':    
        social_scoring_function = recovered_maxinrateoutrate
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
    elif social_scoring == 'rapport_out_rate_in_rate':
        social_scoring_function = rapport_out_rate_in_rate
        dynamic_score = 'no'
    elif social_scoring == 'inverse_rapport_out_rate_in_rate':
        social_scoring_function = inverse_rapport_out_rate_in_rate
        dynamic_score = 'no'
    elif social_scoring == 'recovered_plus_LRIE':
        social_scoring_function = recovered_plus_LRIE
        dynamic_score = 'yes'
    elif social_scoring == 'recovered_plus_weighted_LRIE':
        social_scoring_function = recovered_plus_weighted_LRIE
        dynamic_score = 'yes' 
    elif social_scoring == 'recovered_linearcomb':
        social_scoring_function = recovered_linearcomb
        dynamic_score = 'yes'
    elif social_scoring == 'potentiel_linearcomb_opt':
        social_scoring_function = potentiel_linearcomb_opt
        dynamic_score = 'yes'
    elif social_scoring == 'delta_recovered':
        social_scoring_function = delta_recovered
        dynamic_score = 'yes'
    elif social_scoring == 'delta_recovered_linearcomb':
        social_scoring_function = delta_recovered_linearcomb
        dynamic_score = 'yes'
    elif social_scoring == 'delta_recovered_outrate':
        social_scoring_function = delta_recovered_outrate
        dynamic_score = 'yes'
    elif social_scoring == 'delta_infected':
        social_scoring_function = delta_infected
        dynamic_score = 'yes'
    elif social_scoring == 'delta_infected_linearcomb':
        social_scoring_function = delta_infected_linearcomb
        dynamic_score = 'yes'
    elif social_scoring == 'delta_newinf':
        social_scoring_function = delta_newinf
        dynamic_score = 'yes'
    elif social_scoring == 'delta_newinf_linearcomb':
        social_scoring_function = delta_newinf_linearcomb
        dynamic_score = 'yes'
    elif social_scoring == 'delta_newrec':
        social_scoring_function = delta_newrec
        dynamic_score = 'yes'
    elif social_scoring == 'delta_newrec_linearcomb':
        social_scoring_function = delta_newrec_linearcomb
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
    elif social_scoring == 'out_going_chain':
        social_scoring_function = out_going_chain
        dynamic_score = 'no'    
    elif social_scoring == 'in_going_chain':
        social_scoring_function = in_going_chain
        dynamic_score = 'no'  
    elif social_scoring == 'ventes_periode':
        social_scoring_function = ventes_periode
        dynamic_score = 'yes'  
    elif social_scoring == 'achats_periode':
        social_scoring_function = achats_periode
        dynamic_score = 'yes'  
    elif social_scoring == 'ventesmoinsachats_periode':
        social_scoring_function = ventesmoinsachats_periode
        dynamic_score = 'yes'  
    elif social_scoring == 'ventesplusachats_periode':
        social_scoring_function = ventesplusachats_periode
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
    #dates = np.repeat(pd.date_range(start='1/1/2018', periods=365*3), 2)
    
    for simul in range(1, nb_steps):
        
        current_states = np.copy(all_states[simul-1])
        sizes[simul-1] = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)
        
        # Decision if simul is a farmer decision moment
        if simul in farmers_decision_times:
            
            #Compute reward of previous decision
            if simul != farmers_decision_times[0]:
                
                time_farmers_prev_decision = farmers_decision_times[np.where(farmers_decision_times == simul)[0][0] - 1]
                farmers_prev_decision = farmers_decisions[time_farmers_prev_decision]
                N_farmers_prev_decision = sizes[time_farmers_prev_decision]
                Nt_sum = np.sum(sizes[time_farmers_prev_decision:simul], axis = 0)
                nb_newinf = np.sum(np.sum(all_states[time_farmers_prev_decision:simul], axis = 0)[:,(1,4)], axis = 1)
    
                nb_newinf = np.maximum(nb_newinf, 0)
                reward_dec = -(cf_vacc + (cu_vacc*N_farmers_prev_decision))*farmers_prev_decision - (c_inf*nb_newinf)  
                relat_reward =  reward_dec/Nt_sum
                vaccinators = np.where(farmers_prev_decision == 1.)
                non_vaccinators = np.where(farmers_prev_decision == 0.)
                
            #Farmers make decision
            farmers_decisions[simul] = farmers_decision_function(simul, L, mean_rewards, counts, vaccinators, non_vaccinators, 
                                                 relat_reward, farmers_decision_times, log_weights, kappas, rhos,
                                                 farmers_prev_decision, theta_edges_compact)
            
            #Farmers decisions are applied here 
            current_states = vaccinate(current_states, farmers_decisions[simul])
        
        
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
            
            '''
            #states of previous decision
            if simul != social_decision_times[0]:
                time_social_prev_decision = social_decision_times[np.where(social_decision_times == simul)[0][0] - 1]
                last_states = all_states[time_social_prev_decision]
                cum_newinf_last = np.copy(cum_newinf_now)
                cum_newinf_now = np.sum(np.sum(all_states[time_social_prev_decision: simul, :, [1, 4]], axis= 2), axis = 0)
            else:
                time_social_prev_decision = 0
                last_states = initial_states
                cum_newinf_last = np.zeros(L)
                cum_newinf_now = np.sum(np.sum(all_states[0: simul, :, [1, 4]], axis= 2), axis = 0)
            '''
           
            # Social planner computes scores according to scoring function
            if dynamic_score == 'yes':
                computed_scores = social_scoring_function(current_states, theta_edges, L, last_states, cum_newinf_lastnow,
                                                          cum_newrec_lastnow, cum_ventes_periode, cum_achats_periode,
                                                          cum_ventes, cum_achats, gamma, tau)
            
            all_scores.append(computed_scores)
            
            
            #Social planner makes decision
            social_decisions[simul] = resource_allocation_function(computed_scores, budget, L, dynamic_score, current_states, social_scoring)
            
            #Farmers decisions are applied here 
            current_states = vaccinate(current_states, social_decisions[simul])
        
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
        #print(exports)
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
        
        '''
        sum_true_exports = np.sum(true_exports, axis = 1)
        movements = np.copy(theta_edges)
        movements[:,-1] = sum_true_exports  # add sum true exports column
        col_dates = np.repeat(dates[simul - 1], len(movements))
        movements = np.column_stack((movements,col_dates))
        actual_movements = movements[movements[:, 2] > 0, :] # only consider movements of at least 1 animal
        total_actual_movements.append(actual_movements)
        '''
        
        imports = np.array(imports).T 
        modif_imports = np.insert(imports, [1,3,4], 0, axis=1)
        
        returns = np.array(returns).T
        modif_returns = np.insert(returns , [1,3,4], 0, axis=1)
            
        all_states[simul] = current_states + modif_imports + modif_returns
                           
        achats_byfarm[simul] = np.sum(modif_imports, axis = 1)
        ventes_byfarm[simul] = ventes_byfarm[simul] - np.sum(modif_returns, axis = 1)

    #total_actual_movements = np.array(total_actual_movements)
        
    return farmers_decision_times, social_decision_times, farmers_decisions, social_decisions, all_states, ventes_byfarm, achats_byfarm, all_scores

