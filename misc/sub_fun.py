
import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from scipy.stats import norm
from scipy.stats import poisson
import networkx as nx


## Compute step size using natural gradient
def vi_naturalgrad(t, pram_grad, hess, mIndex, control):
    if mIndex == 1:
        hess = hess + np.outer(pram_grad,pram_grad)
        step_size = control['eta'] * 1/np.sqrt(np.diag(hess))
        return step_size, hess
    
    if mIndex == 2:
        t2 =hess = np.multiply(pram_grad,pram_grad)
        if t==0:
            hess = t2
        hess = control['alpha']*t2 + (1 - control['alpha'])*hess
        step_size = control['eta'] * np.power(t+1, -1/(2+ 1e-10)) * (1/(1 + np.sqrt(hess)))
        return step_size, hess
    

## function for converting factor to integer index:
def fac2index(Ifac):
    n , c = Ifac.shape
    uniqFac = []
    lenfac = []
    for i in range(c):
        dfac = Ifac[:,i]
        fac = np.unique(dfac)
        uniqFac.append(fac)
        lenfac.append(len(fac))
        for j in range(len(fac)):
            dfac[np.where(dfac == fac[j])[0]] = j
        Ifac[:,i] = dfac
    return Ifac, uniqFac, lenfac


# loglikelihood of the zeroinflated poisson 
def logzip(Y, np_pois, np_zero):
    # Initialise likeihood
    like = 0.0
    # Loop over data
    for i in range(Y.shape[1]):
        for j in range(Y.shape[0]):
            if not Y[j][i]:
                # Zero values
                like += np.log(np_zero[j][i] + (1.-np_zero[j][i])*np.exp(-np_pois[j][i]))
            else:
                # Non-zero values
                like += np.log(1-np_zero[j][i]) + poisson.logpmf(Y[j][i], np_pois[j][i])
    return like


# Return derivative and likelihood with respect to the natural parameter 
def grad_logzip_lp34(Y, np_pois, np_zero):
    # Initialise likeihood
    like = 0.0
    # Here we define derivative with respect to the natural paramegter 
    der = np.empty(Y.shape)             # Save derivative w.r.t the natural parameter 
    # Loop over data
    for i in range(Y.shape[1]):
        for j in range(Y.shape[0]):
            if not Y[j][i]:
                # Contribution due to zero outcome values
                tem1 = (1.- np_zero[j][i])*np.exp(-np_pois[j][i])
                like += np.log(np_zero[j][i] + tem1)            # likelihood 
                der[j][i] =  (np_zero[j][i]*(1.-np_zero[j][i])  - \
                   tem1*(np_pois[j][i] + np_zero[j][i]))/(tem1 + np_zero[j][i]) # derivative 
            else :
                # Contribution due to Non-zero outcome values
                der[j][i] = Y[j][i] - np_pois[j][i] - np_zero[j][i]  # derivative
                if np_zero[j][i] != 1. :
                    like += np.log(1.-np_zero[j][i]) - np_pois[j][i] + Y[j][i]*np.log(np_pois[j][i]) #- lgamma(Y[j][i]+1)    # likelihood 
    return der , like


# Return derivative and likelihood with respect to the natural parameter 
def grad_logzip_lp_old_latest(Y, np_pois, np_zero, nz_ind, n0_ind):
    # Initialise likeihood
    like = 0.0
    # Here we define derivative with respect to the natural paramegter 
    der = np.zeros(Y.shape)             # Save derivative w.r.t the natural parameter 
    # Loop over data
    for j, i in np.transpose(nz_ind):
        der[j][i] = Y[j][i] - np_pois[j][i] - np_zero[j][i]  # derivative
        if np_zero[j][i] != 1. :
            like += np.log(1.-np_zero[j][i]) - np_pois[j][i] + Y[j][i]*np.log(np_pois[j][i]) #- lgamma(Y[j][i]+1)    # likelihood 
    for j, i in np.transpose(n0_ind):
        tem1 = (1.- np_zero[j][i])*np.exp(-np_pois[j][i])
        like += np.log(np_zero[j][i] + tem1)            # likelihood 
        der[j][i] =  (np_zero[j][i]*(1.-np_zero[j][i])  - \
           tem1*(np_pois[j][i] + np_zero[j][i]))/(tem1 + np_zero[j][i]) # derivative 
    return der , like

# Return derivative and likelihood with respect to the natural parameter 
def grad_logzip_lp(Y, np_pois, np_zero, nz_ind, n0_ind, sc_tau, LP):
    # Initialise likeihood
    like = 0.0
    # Here we define derivative with respect to the natural paramegter 
    der = np.zeros(Y.shape)             # Save derivative w.r.t the natural parameter 
    der_tau = 0.0;
    # Loop over data
    for j, i in np.transpose(nz_ind):
        der_tau += (LP[j][i])*np_zero[j][i]
        der[j][i] = Y[j][i] - np_pois[j][i] + sc_tau*np_zero[j][i]  # derivative
        if np_zero[j][i] != 1. :
            like += np.log(1.-np_zero[j][i]) - np_pois[j][i] + Y[j][i]*np.log(np_pois[j][i]) #- lgamma(Y[j][i]+1)    # likelihood 
    for j, i in np.transpose(n0_ind):
        tem1 = (1.- np_zero[j][i])*np.exp(-np_pois[j][i])
        temx1 = np_zero[j][i] + tem1
        temx2 = (LP[j][i])*(np_zero[j][i])
        if temx1 > 0.  :
            like += np.log(temx1)            # likelihood 
        if temx1 != 0:
            der_tau += (temx2*tem1 - temx2*(1-np_zero[j][i]))/temx1
            der[j][i] =  (-1*sc_tau*np_zero[j][i]*(1.-np_zero[j][i])  + \
               tem1*(sc_tau*np_zero[j][i] - np_pois[j][i]))/(temx1) # derivative 
    return der , like, der_tau

#logzip(Y, np_pois, np_zero)
def grad_logzip_lp2(Y, np_pois, np_zero):
    # Here we define derivative with respect to the natural paramegter 
    # define output array 
    der = np.empty(Y.shape)
    for i in range(Y.shape[1]):
        for j in range(Y.shape[0]):
            if not Y[j][i]:
                # Contribution due to zero outcome values
                tem1 = (1- np_zero[j][i])*np.exp(-np_pois[j][i])
                # Zero values
                der[j][i] =  (np_zero[j][i]*(1-np_zero[j][i])  - \
                   tem1*(np_pois[j][i] + np_zero[j][i]))/(tem1 + np_zero[j][i]) # derivative 
            else :
                # Non-zero values
                der[j][i] = Y[j][i] - np_pois[j][i] - np_zero[j][i]   # derivative
    return der


## get derivative w.r.t. interaction variable
## Make exclusive in addition 
def d_interaction1(C_latent,nz_ind,L,Q,der_lp):
    out = np.zeros((L,Q))
    for i, j in np.transpose(nz_ind):
        tm = np.setdiff1d(nz_rowY[i][0],j)
        out[:,j] = out[:,j]+ np.sum(C_latent[:,tm],1)*der_lp[i,j]/(nz_len[i]-1)
    return out




## get linear predictor for interaction
def lp_interaction(C_inter,C_latent,nz_ind,N,Q,nz_rowY,nz_len):
    # saving C_interaction.T*C_latent variable 
    tm = np.matmul(C_inter.T, C_latent) 
    np.fill_diagonal(tm, 0)
    tm2 = np.zeros((N,Q))
    for i, j in np.transpose(nz_ind):
        # computing interaction term exclusively
        tm2[i,j] = np.sum(tm[j][nz_rowY[i]])/(nz_len[i]-1)
    return tm2





def post_pred_check(ppd_ss, param_mu, param_log_sigma,Y_te, index,N,P,Q,L,lenfac,X,Ifac,nz_ind,nz_rowY,nz_len, libS):
    latent_sample = np.random.normal(size = ppd_ss*len(param_mu)).reshape(len(param_mu),ppd_ss)  
    latent_sample = (latent_sample.T * np.exp(param_log_sigma) + param_mu).T
    ppd_ind_n0 = np.where(Y_te >= 0)
    pred_val  = np.zeros((len(ppd_ind_n0[1]),ppd_ss))  
    for ppd_ind in range(ppd_ss):
        z_0 = latent_sample[:,ppd_ind]
        C_tem = z_0[index[0]].reshape(P,Q)              # coef matrix sample [BioGeo chemical]
        C_latent = z_0[index[1]].reshape(L,Q)           # Latent variable species [Species latent variable]
        C_inter = z_0[index[2]].reshape(L,Q)            # Latent variable interaction [Interaction latent variable]
        
        I_coef1 = z_0[index[3]].reshape(lenfac[0],L)    # Latent variable Indicator - I
        I_coef2 = z_0[index[4]].reshape(lenfac[1],L)    # Latent variable Indicator - II
        sc_tau  = z_0[index[5]][0]                      # shape parameter in the model 
        sc_tau  = np.exp(sc_tau)
        ### get linear predictor 
        LP_covar = np.matmul(X, C_tem)                  # Linear predictor: coefficient
        
        ## Linear predictor due to indicator variable
        LP_indic = np.matmul(I_coef1[list(Ifac[:,0])], C_latent)
        LP_indic += np.matmul(I_coef2[list(Ifac[:,1])], C_latent)
        
        # get linear predictor for interaction
        LP_interact = lp_interaction(C_inter,C_latent,nz_ind,N,Q,nz_rowY,nz_len)
        
        LP = LP_covar + LP_indic  + LP_interact         # linear predictor 
        #np_pois = np.exp(LP)                            # NP for the poisson component 
        #np_zero = sigmoid(LP)                           # NP for the degenerate component 
        
        np_pois = np.exp(LP)                            # NP for the poisson component 
        np_pois = np_pois * libS[:, np.newaxis]
        np_zero = sigmoid(-1*sc_tau*LP) 
    
        lambda_zip = np_pois[ppd_ind_n0]
        pstr_zip = np_zero[ppd_ind_n0]
        pred_val[:,ppd_ind] = qzipois(lambda_zip, pstr_zip)
    return pred_val, Y_te[ppd_ind_n0]
    
    
    
    
# p is the 
def qzipois(lambda_zip, pstr_zip):
    lambda_zip = lambda_zip.astype(np.float64)
    pstr_zip = pstr_zip.astype(np.float64)
    # capping large value for the estimation purpose 
    lambda_zip[np.where(lambda_zip>1e18)] = 1e18
    if any(pstr_zip<0.) or any(pstr_zip>1.):
        print('Invalid zero inflated parameters.')
    randn = np.random.uniform(size=lambda_zip.shape[0])  
    gind = np.where(randn > pstr_zip)
    val = np.zeros(lambda_zip.shape)
    val[gind] = np.random.poisson(lam=lambda_zip[gind])
    return val

    
# p is the 
def qzipoist(lambda_zip, pstr_zip):
    lambda_zip = lambda_zip
    tm = len(lambda_zip)
    p = np.random.uniform(size=tm)
    ans = np.zeros(tm)
    deflat_limit = lambda_zip
    for i in range(lambda_zip.shape[0]):
        deflat_limit[i] = -1/(np.exp(lambda_zip[i])-1)
    #deflat_limit =  -1/(np.exp(lambda_zip)-1)
    ans[np.where(p <= pstr_zip)] = 0.
    pindex = np.where((pstr_zip < p) & (deflat_limit <= pstr_zip))
    ans[pindex] = poisson.ppf((p[pindex] - pstr_zip[pindex])/(1 - pstr_zip[pindex]),lambda_zip[pindex])
    ans[pstr_zip < deflat_limit] = nan
    ans[1 < pstr_zip] = nan
    ans[lambda_zip < 0] = nan
    ans[p < 0] = nan
    ans[1 < p] = nan
    return ans
    






def return_indices_of_a(a, b):
    out = []
    for i in range(len(a)):
        out.append(list(np.where(a[i] == b)[0]))
    return np.concatenate(out)




def modify_pos(G, ng, cindex):
    pos=nx.spring_layout(G) # positions for all nodes
    x_box = np.linspace(-5., 5.0, num=ng)
    y_box = np.linspace(-5., 5.0, num=ng)
    box_index = np.random.choice(range(9),size=len(cindex[1]), replace=False)
    box_mat = np.array(range((ng-1)*(ng-1))).reshape(((ng-1),(ng-1)))
    
    # if node is in second group, move it up
    for i in range(len(cindex[1])):
        indx = np.where(cindex[0] == i)
        (rin,cin)=np.where(box_mat==box_index[i])
        txcord = np.random.uniform(x_box[rin],x_box[rin+1],len(indx[0]))
        tycord = np.random.uniform(y_box[cin],y_box[cin+1],len(indx[0]))
        for j in range(len(txcord)):
            pos[indx[0][j]] = np.array([txcord[j],tycord[j]])
            
    return pos





def cap_mat(B, lv, uv):
    ind = np.where((B>lv) & (B<uv))
    B[ind] = 0
    return B


def get_geomean(a, err):
    import scipy.optimize as optmin
    def delfun(x):
        return -x
    a_gp  = (1+err)*np.exp(np.mean(np.log(a[a>0])))
    bounds = optmin.Bounds([err], [np.inf])
    
    
    def cons_f(x):
        return np.exp(np.mean(np.log(a+x)))
    
    nonlinear_constraint = optmin.NonlinearConstraint(cons_f,\
                                                      -np.inf, a_gp,\
                                                      jac='2-point',\
                                                      hess=optmin.SR1())
    
    res = optmin.minimize(delfun, 0.5, method='trust-constr',  jac="2-point",\
                          hess=optmin.SR1(),\
                          constraints=[nonlinear_constraint],\
                          options={'verbose': 0}, bounds=bounds)
    return res.x




def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    """

    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max


def hpd(x, alpha=0.05):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI).
    :Arguments:
        x : Numpy array
        An array containing MCMC samples
        alpha : float
        Desired probability of type I error (defaults to 0.05)
    """

    # Make a copy of trace
    x = x.copy()
    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:]+[0])
        dims = np.shape(tx)
        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1]+(2,))

        for index in make_indices(dims[:-1]):
            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])
            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)
        # Transpose back before returning
        return np.array(intervals)
    else:
        # Sort univariate node
        sx = np.sort(x)
        return np.array(calc_min_interval(sx, alpha))
    

def community_layout(g, partition, scale_c, scale_n):
    """
    Compute the layout for a modular graph.  scale_c  = 3.; scale_n = 1.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=scale_c)

    pos_nodes = _position_nodes(g, partition, scale=scale_n)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = tuple(pos_communities[node] + pos_nodes[node])

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=np.log(len(edges)))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos



