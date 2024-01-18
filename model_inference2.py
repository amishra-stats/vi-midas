#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:42:21 2021

@author: amishra
"""

%reset

## call python script to load data and column order
import os
os.chdir('/Users/amishra/Desktop/googledrive/vi/mem')
exec(open('data_file.py').read())
# load packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pltx
import seaborn as sns
import pystan
import pickle
import scipy.cluster.hierarchy as sch
import copy
import vb_stan as vbfun
import sub_fun as sf
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
import copy 
import random
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import kneighbors_graph
from fa2 import ForceAtlas2
from networkx.algorithms.community import greedy_modularity_communities
import networkx as nx
random.seed(123)
figfol = 'mem_plot/'

plt.rcParams.update(plt.rcParamsDefault)
params = {'legend.fontsize': 12,
          'font.weight': 'bold',
         'axes.labelsize': 12,
         'axes.titlesize':12,
         'axes.labelweight': 'bold',
         'xtick.labelsize':12,
          'axes.titleweight': 'bold',
         #'figure.figsize': (15,8),
          'figure.dpi': 200,
         'ytick.labelsize':12}
plt.rcParams.update(params)

%matplotlib inline
distinct_colp = ["maroon","brown","olive", "teal", "navy", "red", "orange",\
 "yellow", "lime", "green", "cyan", "blue", "purple", "magenta",\
 "grey", "pink", "darkorange", "beige", "slategray", "lavender", 'lightgreen',\
     "cornflowerblue","olivedrab",'darkolivegreen']
# https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib


'''
Load all the data from the file 
'''

# Call the output from the fitted model:
fname_best = '66_model_nb_cvtest.pkl'
folname = 'MMSens/'
[holdout_mask, llpd, n_test, l,m_seed,sp_mean,\
 sp_var, h_prop, uid, nsample_o, Yte_fit, cv_test] = pickle.load(open(folname + fname_best, "rb"))

# -------------------------------------------------
# save model output 
fname_ot = folname + str(uid) + '_' + 'model_nb.pkl' 
#with open(fname_o, 'wb') as f:
#    pickle.dump(NB_vb, f)
with open(fname_ot, 'rb') as f:
    results = pickle.load(f)
    
sample_fname = folname + str(uid) + '_' + 'sample_model_nb_cvtest.pkl' 
with open(sample_fname, 'rb') as f:
    [Yte_sample,Yte_cv] = pickle.load(f)

parma_mean  = dict(vbfun.vb_extract_mean(results))





'''
Get taxonomic classification of the species 
'''


tax_name = pd.read_csv('species_tax.csv')
tax_name = tax_name.rename(columns={'Unnamed: 0': 'OTU'})
tax_name = tax_name[1:]
tax_name.insert(0, 'Id', tax_name['OTU'].str[3:])
tax_name.columns.values[1] = 'Label'
tax_name.to_csv('node_otu.csv', index = False) 
temx = tax_name.iloc[:,:8]
temx = temx.replace(np.nan,'')
species_name = []
# Add taxonomy identifier to the each of the species name 
for i in range(temx.shape[0]):
    a = temx.iloc[i,:].values
    for j in range(a.shape[0]-1,-1,-1):
        if len(a[j]) > 0:
            species_name.append(temx.columns[j][0].lower()+'_'+ a[j])
            break;           
species_name = np.array(species_name)  
tax_name['Name'] = species_name
tax_name['ord_abu'] = np.linspace(10,1,tax_name.shape[0])[(-1*Y.mean(axis=0)).argsort()]
tax_name[['Id']] = tax_name[['Id']].values.astype(np.int64)
tax_name = tax_name.replace(np.nan,'Empty')
tem = pd.read_csv('species_tax_anot.amended.csv').iloc[:,[1,12]]
tax_name = tax_name.merge(tem,on = 'Label')
tax_name = tax_name.rename(columns={"Ecologically_relevant_classification_aggregated": "ECR"})

## Update the new annotation in the
i = 11    # index for ECR variable 
ind_var = tax_name.iloc[:,i].values
vals, counts = np.unique(ind_var, return_counts=True)
tem_ind1 = (-1*counts).argsort()
tem_val = vals[tem_ind1][range(np.min([np.sum(counts > 10),tem_ind1.shape[0]]))]
tmp = np.setdiff1d(np.unique(ind_var), tem_val)
tmp = sf.return_indices_of_a(tmp, ind_var)
ind_var[tmp] = "Other" 
tax_name.iloc[:,i] = ind_var




'''
Perform softmax regression with based on the latent characteristic of the species specific parameters
'''

## Prepare response matrix for softmax regression 
tax_name2 = copy.copy(tax_name) 
encode_dict = [[val,i] for i,val in enumerate(np.unique(tax_name2['ECR'].values)) ]
ord_enc = OrdinalEncoder()
ord_enc.fit(tax_name2['ECR'].values.reshape(-1,1))
tax_name2["ECR_code"] = ord_enc.fit_transform(tax_name2['ECR'].values.reshape(-1,1))
cat_val = tax_name2["ECR_code"].values.astype(np.int)


sp_feature = parma_mean['L_sp']   
#sp_feature = np.concatenate((parma_mean['L_sp'], parma_mean['C_geo']), axis = 1)
softReg = LogisticRegressionCV(random_state=10, multi_class = 'multinomial', max_iter = 2000)
clf = softReg.fit(sp_feature, cat_val)
cat_pred = clf.predict(sp_feature)
tax_name2["ECR_pred"]  = ord_enc.inverse_transform(cat_pred.reshape(-1,1))


# Show confusion matrix and present the 
confusion_matrix = pd.crosstab(tax_name2['ECR'], tax_name2["ECR_pred"],\
                               rownames=['Actual'], colnames=['Predicted'],\
                                   dropna = False, normalize = 'index')
fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(confusion_matrix, annot=True, ax = ax)



## Select species for similarity matrix display 
selected_species = [i for i in list(confusion_matrix.columns) if confusion_matrix.loc[i,i] > 0.35]
#selected_species.remove('Other')
selected_species_index = tax_name['ECR'].isin(selected_species).values
species_col_dict = dict(zip(selected_species,distinct_colp[:len(selected_species)]))
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
label = selected_species 
fig, ax  = plt.subplots(dpi = 100)
handles = [f("s", species_col_dict[i]) for i in selected_species[:18]]
plt.legend(handles, label, loc=4, framealpha=1, frameon=False, ncol=3)
plt.gca().set_axis_off()
plt.show()






'''
Similarity among speecies 
'''


def most_important(G,n_sel):
    ranking = nx.betweenness_centrality(G).items()
    ranking = sorted(list(ranking),key=lambda x: x[1], reverse=True)
    nodes_sel = [ranking[i][0] for i in range(n_sel)]
    Gt = g.subgraph(nodes_sel)
    return Gt

#tax_name3 = tax_name[tax_name['ECR'].isin(selected_species)]
tax_name3 = copy.copy(tax_name)
tax_name3['Id'] =  tax_name3['Label']
tax_name3 = tax_name3[selected_species_index]
#tax_name3.to_csv('Species_annotate_new.csv',index=False)

## Select species based on #k = 10# neares neighbour and see graph 
sp_feature = parma_mean['L_sp'][selected_species_index] 
#sp_feature = np.concatenate((parma_mean['L_sp'], parma_mean['C_geo']), axis = 1)[selected_species_index] 

dist = kneighbors_graph(sp_feature, 10, mode='distance', metric = "cosine",
                        include_self=False).toarray()
node_id = tax_name3['Label'].values
species_selected = pd.DataFrame()
for i in range(dist.shape[0]):
    a = np.where(dist[i] != 0.)[0]
    if a.shape[0] > 0:
        d = {'Id': node_id[i],'Source': node_id[i], 'Target': node_id[a],\
         'Type' : 'Undirected', 'Weightx': 1,\
         'weight': dist[i][a] }
        species_selected = species_selected.append(pd.DataFrame(data=d))
# sanity check
species_selected.to_csv('edge_cosinedist_splatent.csv', index = False)
species_selected = pd.merge(species_selected, tax_name3, on='Id')


tax_name3['color'] = [species_col_dict[n] for n in tax_name3['ECR'].values]
node_atr = {}
for i in range(tax_name3.shape[0]):
    node_atr[tax_name3['Id'].values[i] ] = tax_name3.iloc[i,1:].to_dict()
    

 

np.random.seed(1234)
out = species_selected
g = nx.from_pandas_edgelist(out, source='Source', target='Target',\
                             edge_attr = 'weight',\
                             create_using=nx.Graph()) 
nx.set_node_attributes(g, node_atr)
deg = g.degree()
to_keep = [n for (n, a) in deg if a > 0]
g_sub = g.subgraph(to_keep)

forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=True,  # Dissuade hubs
                        linLogMode=False,  # NOT IMPLEMENTED
                        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=1.2,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=2.0,
                        strongGravityMode=False,
                        gravity=1.0,

                        # Log
                        verbose=True)

random.seed(123)
positions = forceatlas2.forceatlas2_networkx_layout(g_sub, pos=None, iterations=2000)
layout_flip = positions# {node: (x,-y) for (node, (x,y)) in layout.items()}


n_size = [g_sub.nodes[club]['ord_abu'] for club in list(g_sub)]
edges = g_sub.edges()
widths_n = weights = 2*np.array([g_sub[u][v]['weight'] for u,v in edges])

node_labl = [g_sub.nodes[n]['ECR'] for n in list(g_sub) ]
colors2 = [species_col_dict[n]  for n in node_labl ]
Gt = most_important(g_sub,10) # trimming
colors3 = [species_col_dict[Gt.nodes[n]['ECR']]  for n in list(Gt) ]

fnamex = figfol + 'species_interaction_mat1.pdf'
fig, ax  = plt.subplots(dpi = 200)
nx.draw_networkx_edges(g_sub, layout_flip, width=0.4*widths_n, alpha=0.2, ax =ax)
nc = nx.draw_networkx_nodes(g_sub, 
                       layout_flip, 
                       nodelist=list(g_sub), 
                       node_size= 1*np.array(n_size), 
#                       node_color='blue',\
                       ax = ax,alpha = 0.9,
                       #label= node_labl,
                       node_color=colors2)
# nx.draw_networkx_nodes(Gt,layout_flip,node_shape="*",alpha=0.9,node_color=colors3, node_size = 70)
ax.axis('off')
ax.set_title("Species species interaction", fontsize = 10)
fig.tight_layout()
# fig.savefig(fnamex)
#pp.savefig(fig,dpi = 200, bbox_inches='tight', pad_inches=0)





'''
Detect communities and highlight 
'''


def group_summary(otu_sel,g_def,top_c,m_t, a):
    temp = [g_def.nodes[n]['ECR'] for n in list(otu_sel)]
    typ, ct = np.unique(temp, return_counts=True)
    sel = typ[ct.argsort()[-top_c:]]
    for n in list(otu_sel):
        g_def.nodes[n]['alp'] = 0.02
        g_def.nodes[n]['mtype'] = '.'
        g_def.nodes[n]['nlab'] = None
        if any(sel == g_def.nodes[n]['ECR']):
            g_def.nodes[n]['alp'] = 0.9 
            g_def.nodes[n]['mtype'] = m_t
            g_def.nodes[n]['nlab'] = a
            

comunity = list(greedy_modularity_communities(g_sub))

markers_type = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h',\
                'H', 'D', 'd', 'P', 'X', '+', '1', '2', '3', '4']

g_def = copy.copy(g_sub)
top_c = 3
for i in range(len(comunity)):
    print(len(comunity[i]))
    group_summary(comunity[i],g_def,top_c,markers_type[i],i)
    


nlab_l = {}; nlab_n = []; 
for n in list(g_def):
    if g_def.nodes[n]['nlab'] != None:
        nlab_l[n] = g_def.nodes[n]['nlab']
        
        
alp_v = np.array([g_def.nodes[n]['alp'] for n in list(g_def)])
mtype = np.array([g_def.nodes[n]['mtype'] for n in list(g_def)])
layout_flip = layout #{node: (x,-y) for (node, (x,y)) in layout.items()}
               
fig, ax  = plt.subplots(dpi = 200)
for m_t in np.unique(mtype):
    #m_t = mtype[0]
    sel_ind = np.where(mtype == m_t)
    nc = nx.draw_networkx_nodes(g_def, 
                           layout_flip, 
                           nodelist= [list(g_def)[i] for i in sel_ind[0]], 
                           node_size= 1*np.array(n_size)[sel_ind], 
                           ax = ax,alpha = alp_v[sel_ind],
                           node_shape = m_t,
                           #label= node_labl,
                           node_color= [colors2[i] for i in sel_ind[0]])
nx.draw_networkx_edges(g_def, layout_flip, width=0.4*widths_n, alpha=0.2, ax =ax)
nx.draw_networkx_labels(g_def, layout_flip, labels = nlab_l,font_size = 4,\
                        alpha = 0.9, ax= ax, verticalalignment = 'top', horizontalalignment = 'left')
#nx.draw_networkx_nodes(Gt,layout,node_shape="*",alpha=0.9,node_color=colors3, node_size = 70)
ax.axis('off')
ax.set_title("Species species interaction", fontsize = 10)
fig.tight_layout()


'''
Prepare species selected matrix for each cluster: 
Cluster representative in the model 
[cluster shape, color ecr, ecr name ]
1080 species are depicted here 
'''


f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
label = selected_species 
fig, ax  = plt.subplots(dpi = 200)
handles = [f("s", species_col_dict[i]) for i in selected_species[:18]]
plt.legend(handles, label, loc=4, framealpha=1, frameon=False, ncol = 1)
plt.gca().set_axis_off()
plt.show()

temp = [[g_def.nodes[n]['ECR'], g_def.nodes[n]['nlab']] for n in list(nlab_l.keys())]
temp = pd.DataFrame(temp).drop_duplicates()
cl_lab = dict({i:[] for i in np.unique(temp[0].values)})
for i in range(temp.shape[0]):
    cl_lab[temp.iloc[i,0]].append(temp.iloc[i,1])

label = [ key+': '+ str(cl_lab[key]) for key in cl_lab]
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
#label = selected_species 
fig, ax  = plt.subplots(dpi = 200, figsize = (10,1))
handles = [f("s", species_col_dict[i]) for i in selected_species[:18]]
ax.legend(handles, label, loc=4, framealpha=1, frameon=False, ncol = 4)
ax.set_axis_off()





'''
Positive and negative interaction based on cosine distance 
'''


cov_mat = np.matmul(parma_mean['L_sp'],parma_mean['L_i'].T)
cov_mat  = cov_mat.max() - cov_mat
cov_mat = (cov_mat + cov_mat.T)/2
np.fill_diagonal(cov_mat,0)

selected_species_index = tax_name['ECR'].isin(selected_species).values
dist_pos = copy.copy(cov_mat[selected_species_index][:,selected_species_index])
dist_neg = copy.copy(cov_mat[selected_species_index][:,selected_species_index])
for i in range(dist_pos.shape[0]):
    tem = dist_pos[i].argsort()
    dist_pos[i,tem[5:]] = 0.
    dist_neg[i,tem[:-5]] = 0.
    

dist = copy.copy(dist_neg)


#tax_name3 = tax_name[tax_name['ECR'].isin(selected_species)]
tax_name3 = copy.copy(tax_name)
tax_name3['Id'] =  tax_name3['Label']
tax_name3 = tax_name3[selected_species_index]
#tax_name3.to_csv('Species_annotate_new.csv',index=False)

## Select species based on #k = 10# neares neighbour and see graph 
node_id = tax_name3['Label'].values
species_selected = pd.DataFrame()
for i in range(dist.shape[0]):
    a = np.where(dist[i] != 0.)[0]
    if a.shape[0] > 0:
        d = {'Id': node_id[i],'Source': node_id[i], 'Target': node_id[a],\
         'Type' : 'Undirected', 'Weightx': 1,\
         'weight': dist[i][a] }
        species_selected = species_selected.append(pd.DataFrame(data=d))
# sanity check
#species_selected.to_csv('edge_cosinedist_posit.csv', index = False)
species_selected = pd.merge(species_selected, tax_name3, on='Id')



tax_name3['color'] = [species_col_dict[n] for n in tax_name3['ECR'].values]
node_atr = {}
for i in range(tax_name3.shape[0]):
    node_atr[tax_name3['Id'].values[i] ] = tax_name3.iloc[i,1:].to_dict()
    



np.random.seed(1234)
out = species_selected
g = nx.from_pandas_edgelist(out, source='Source', target='Target',\
                             edge_attr = 'weight',\
                             create_using=nx.Graph()) 
#layout = nx.circular_layout(g)
nx.set_node_attributes(g, node_atr)
deg = g.degree()
val, counts = np.unique([a for (n, a) in deg], return_counts=True)
to_keep = [n for (n, a) in deg if a > 0]
deg_keep = [n for n, a in deg if a > val[-10]]
Gt_d = g.subgraph(deg_keep)
g_sub = g.subgraph(to_keep)

#layout = nx.spring_layout(g_sub,iterations=100, scale =2, k=3*1/np.sqrt(len(g_sub.nodes())))

forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=True,  # Dissuade hubs
                        linLogMode=False,  # NOT IMPLEMENTED
                        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=1.2,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=2.0,
                        strongGravityMode=False,
                        gravity=1.0,

                        # Log
                        verbose=True)

random.seed(123)
positions1 = forceatlas2.forceatlas2_networkx_layout(g_sub, pos=None, iterations=2000)
# positions = nx.spring_layout(g_sub, pos=None, iterations=200, seed = 123, weight = None)
layout = positions1

#layout = nx.circular_layout(g)
club_size = [g_sub.degree(club) for club in list(g_sub)]
n_size = [g_sub.nodes[club]['ord_abu'] for club in list(g_sub)]
edges = g_sub.edges()
widths_n = weights = 2*np.array([g_sub[u][v]['weight'] for u,v in edges])
clubsx = [str(x) +'[' +str(y)+']' for x,y in zip(list(g_sub),club_size)]
#colors = [g_sub.nodes[n]['color'] for n in list(g_sub) ]
node_labl = [g_sub.nodes[n]['ECR'] for n in list(g_sub) ]
# mapping = dict(zip(sorted(set(node_labl)),count()))
# 
colors2 = [species_col_dict[n]  for n in node_labl ]


#Gt_central = most_important(g_sub,10) # trimming
Gt = Gt_d
colors3 = [species_col_dict[Gt.nodes[n]['ECR']]  for n in list(Gt) ]
#nx.draw(g)
#fnamex = 'species_interaction_mat1.pdf'
# pp = PdfPages(fnamex)
fig, ax  = plt.subplots(dpi = 200)
nx.draw_networkx_edges(g_sub, layout, width=0.1*widths_n, alpha=0.2, ax =ax)
nc = nx.draw_networkx_nodes(g_sub, 
                       layout, 
                       nodelist=list(g_sub), 
                       node_size= 0.5*np.array(n_size), # a LIST of sizes, based on g.degree
#                       node_color='blue',\
                       ax = ax,alpha = 0.9,
                       #label= node_labl,
                       node_color=colors2)

# draw the most important nodes with a different style
nx.draw_networkx_nodes(Gt,layout,node_shape="*",alpha=0.9,node_color=colors3, 
                       nodelist=list(Gt),
                       node_size = 70)
nx.draw_networkx_labels(Gt, layout, font_size=10, font_color='black',\
                        labels={list(Gt)[i]:i for i in range(len(list(Gt)))},\
                            horizontalalignment = 'center',verticalalignment = 'center')

# also the labels this time
#nx.draw_networkx_labels(Gt,layout,font_size=12,font_color='b')
#plt.colorbar(nc)
ax.axis('off')
ax.set_title("Species species interaction", fontsize = 10)
fig.tight_layout()
#pp.savefig(fig,dpi = 200, bbox_inches='tight', pad_inches=0)



labelx_deg = [Gt.nodes[n]['ECR'] +' [' +str(g_sub.degree(n))+ ']' for n in list(Gt)]

f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
fig, ax  = plt.subplots(dpi = 200)
handles = [f("s", species_col_dict[Gt.nodes[n]['ECR']])  for n in list(Gt) ]
plt.legend(handles, labelx_deg, loc=4, framealpha=1, frameon=False, ncol = 1)
plt.gca().set_axis_off()
plt.show()



## Interaction matrix 
A = nx.adjacency_matrix(g_sub, weight = None).toarray()
selind = np.isin(list(g_sub), list(Gt))
temp_df = pd.DataFrame(A[selind].transpose())
temp_df.columns = list(Gt)
temp_df['ECR'] = [g_sub.nodes[n]['ECR'] for n in list(g_sub)]
df_plot = temp_df.groupby('ECR').agg('sum')
df_plot.columns = [g_sub.nodes[n]['ECR'] for n in list(Gt)]





df_plotP = copy.copy(df_plot)
df_plotN = copy.copy(df_plot)


fig, ax = plt.subplots(nrows=2, ncols =1, figsize = (16,8), sharex = True)
sns.heatmap(df_plotP.transpose(), annot=True, ax = ax[0], fmt='',\
            annot_kws={"fontsize": 10}, cbar = None)
ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_title('Positive association', c = 'b')
sns.heatmap(df_plotN.transpose(), annot=True, ax = ax[1], fmt='',\
            annot_kws={"fontsize": 10},cbar = None)
ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].set_title('Negative association', c = 'b')




