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
import sub_fun as sf
tmp = sf.return_indices_of_a(tmp, ind_var)
ind_var[tmp] = "Other" 
tax_name.iloc[:,i] = ind_var



'''
Vertically divide the species into groups
# [65, 135,326, 339, 381, 590, 1026] [xticklabels = 50]
# Biome and geochemical compinents are most important component 
# causes most significant drop in the 
'''


v_index = [0, 320, 635 ,1000, 1290, Y.shape[1]]
v_index_minor = [v_index[i] + 0.5*(v_index[i+1]- v_index[i]) for i  in range(len(v_index)-1)]
h_index = [0, 108, Y.shape[0]]
h_index_minor = [h_index[i] + 0.5*(h_index[i+1]- h_index[i]) for i  in range(len(h_index)-1)]
tem = np.log(Y[r_ord][:,c_ord])
fig, ax = pltx.subplots(dpi = 100)
sns.heatmap(tem, cmap="jet", ax = ax, xticklabels = False, yticklabels = False)

# Formating the figure 
ax.vlines(v_index, colors = 'black',\
          linestyles = 'dashed', ymin = 0, ymax = 139,\
              linewidths = 1.8)
ax.hlines(h_index, colors = 'black',\
          linestyles = 'dashed', xmin = 0, xmax = 1378,\
              linewidths = 1.8)
ax.set_xticks(v_index)
ax.xaxis.set_major_formatter(ticker.NullFormatter())
ax.xaxis.set_minor_locator(ticker.FixedLocator(v_index_minor))
ax.xaxis.set_minor_formatter(ticker.FixedFormatter(range(1,len(v_index))))
ax.set_yticks(h_index)
ax.yaxis.set_major_formatter(ticker.NullFormatter())
ax.yaxis.set_minor_locator(ticker.FixedLocator(h_index_minor))
ax.yaxis.set_minor_formatter(ticker.FixedFormatter([' ','MES']))
plt.setp( ax.yaxis.get_minorticklabels(), rotation=90, va='center' )
ax.set_xlabel('Species groups')
ax.set_ylabel('Samples')
fig.tight_layout()





'''
BiogeoChemical components analysis 
'''


tem = np.matmul(X,parma_mean['C_geo'].transpose())
tem = tem[r_ord][:,c_ord]

fig, ax = plt.subplots(dpi = 100)
#sns.heatmap(tem, cmap="jet", ax = ax, xticklabels = False, yticklabels = False)
sns.heatmap(tem, cmap="jet", ax = ax, xticklabels = False, yticklabels = False)
# Formating the figure 
ax.vlines(v_index, colors = 'black',\
          linestyles = 'dashed', ymin = 0, ymax = 139,\
              linewidths = 1.8)
ax.hlines(h_index, colors = 'black',\
          linestyles = 'dashed', xmin = 0, xmax = 1378,\
              linewidths = 1.8)
ax.set_xticks(v_index)
ax.xaxis.set_major_formatter(ticker.NullFormatter())
ax.xaxis.set_minor_locator(ticker.FixedLocator(v_index_minor))
ax.xaxis.set_minor_formatter(ticker.FixedFormatter(range(1,len(v_index))))
ax.set_yticks(h_index)
ax.yaxis.set_major_formatter(ticker.NullFormatter())
ax.yaxis.set_minor_locator(ticker.FixedLocator(h_index_minor))
ax.yaxis.set_minor_formatter(ticker.FixedFormatter([' ','MES']))
plt.setp( ax.yaxis.get_minorticklabels(), rotation=90, va='center' )
ax.set_xlabel('Species groups')
ax.set_ylabel('Samples')
fig.tight_layout()



## Visualize coefficient matrix estimate 
tem = parma_mean['C_geo'].transpose()
label_var = ['Temperature', 'Salinity','Oxygen', 'Nitrates', 'NO2', 'PO4', 'NO2NO3', 'SI', 'grad_SST']

fig, ax = plt.subplots(dpi = 100)
#sns.heatmap(tem, cmap="jet", ax = ax, xticklabels = False, yticklabels = False)
sns.heatmap(tem, cmap="coolwarm", ax = ax, xticklabels = False, yticklabels = label_var)
ax.vlines(v_index, colors = 'black',\
          linestyles = 'dashed', ymin = 0, ymax = 139,\
              linewidths = 1.8)
    
ax.set_xlabel('Species groups')
ax.set_ylabel('Environmental factors')
ax.set_xticks(v_index)
ax.xaxis.set_major_formatter(ticker.NullFormatter())
ax.xaxis.set_minor_locator(ticker.FixedLocator(v_index_minor))
ax.xaxis.set_minor_formatter(ticker.FixedFormatter(range(1,len(v_index))))


avg_effect = np.zeros((len(label_var), len(v_index) -1))
for i in range(len(v_index)-1):
     avg_effect[:,i] = np.mean(parma_mean['C_geo'][v_index[i]:v_index[i+1]], axis =0) 
     
avg_effect = pd.DataFrame(avg_effect)
avg_effect.index = label_var
avg_effect.columns = range(1,6)


# '''
# Effect size of the genechemical covariates
# '''
# fig, ax = plt.subplots(dpi = 100)
# avg_effect.plot( y=range(1,6), kind="bar", ax =ax)
# plt.legend(ncol=len(avg_effect.columns), frameon = False, bbox_to_anchor=(1.0, 1.1))


'''
Pie chart and bar chart for the effect size for each groups
'''

tax_name2 = copy.copy(tax_name.iloc[c_ord,:])
tem = np.array([None]*tax_name2.shape[0])
for i in range(len(v_index)-1):
    tem[v_index[i]:v_index[i+1]] = i+1
tax_name2['Group'] = tem
tax_name3 = tax_name2[tem!=None]

selected_species = []
for i in range(1,6):
    temp = tax_name3['ECR'][tax_name3['Group'] == i ].values
    vals, counts = np.unique(temp, return_counts=True)
    selected_species += list(vals[counts.argsort()][-5:])
selected_species = list(set(selected_species + ['Other'] )) 
species_col_dict = dict(zip(selected_species,distinct_colp[:len(selected_species)]))

f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
label = selected_species 
fig, ax  = plt.subplots(dpi = 100)
handles = [f("s", species_col_dict[i]) for i in selected_species]
plt.legend(handles, label, loc=4, framealpha=1, frameon=False, ncol = 5)
plt.gca().set_axis_off()
plt.show()



fig, ax = plt.subplots(1,5, dpi = 200, figsize = (10,2))
for i in range(1,6):
    temp = tax_name3['ECR'][tax_name3['Group'] == i ].values
    vals, counts = np.unique(temp, return_counts=True)
    sel_var = vals[counts.argsort()][-5:]
    #temp[np.invert(np.isin(temp, sel_var))] = 'Others'
    temp = temp[np.isin(temp, sel_var)] 
    vals, counts = np.unique(temp, return_counts=True)
    col_pie = [species_col_dict[i] for i in vals]
    ax[i-1].pie(x=100*counts/sum(counts), autopct="%.1f%%", labels=None,\
                colors = col_pie, textprops={'fontsize': 4})
    ax[i-1].set_xlabel('G' + str(i) + ': ' + str(temp.shape[0]), fontsize=4)
    

fig, ax = plt.subplots(1,5, dpi = 200, sharey = True, figsize = (10,2))
for i in range(1,6):
    avg_effect.plot( y=i, kind="barh", ax =ax[i-1], legend = False, fontsize = 4)
    plt.xticks(rotation = 90)







    


# ## Geochemical covariates effects summary 
# tax_name2 = copy.copy(tax_name.iloc[c_ord,:])
# tem = np.array([None]*tax_name2.shape[0])
# for i in range(len(v_index)-1):
#     tem[v_index[i]:v_index[i+1]] = i+1
# tax_name2['Group'] = tem
# tax_name3 = tax_name2[tem!=None]

# # Show confusion matrix and present the 
# confusion_matrix = pd.crosstab(tax_name3["Group"], tax_name3['ECR'],\
#                                rownames=['Actual'], colnames=['Predicted'],\
#                                    dropna = False, normalize = 'index')
# fig, ax = plt.subplots(figsize=(15, 10))
# sns.heatmap(confusion_matrix, annot=True, ax = ax)
# selected_species = [i for i in list(confusion_matrix.columns) if confusion_matrix[i].sum() > 0.05]
# selected_species.remove('Other')
# tax_name3 = tax_name3[tax_name3['ECR'].isin(selected_species)]


# ## get pie chart for group 1:  top 5

# ## Average effect size 
# ax = sns.catplot(x="ECR", kind="count", hue = 'Group', data=tax_name3, 
#             height=5, aspect=2.2)
# ax.set_xticklabels(rotation=90, fontsize = 14)
# ax.set_yticklabels(fontsize = 14)
# plt.axhline(y = 8, color='r', ls = '--', lw = 0.99)

    



'''
Biome components analysis 
[65, 135,326, 339, 381, 590, 1026]
np.where(np.diff(pd.DataFrame(B[r_ord])[1].values) ==1)
'''

tem = np.matmul(B,np.matmul(parma_mean['A_b'],\
                            parma_mean['L_sp'].transpose()))
tem = tem[r_ord][:,c_ord]
fig, ax = pltx.subplots(dpi = 100)
sns.heatmap(tem, cmap="jet", ax = ax, xticklabels = False, yticklabels = False)
ax.vlines(v_index, colors = 'black',\
          linestyles = 'dashed', ymin = 0, ymax = 139,\
              linewidths = .8)
# Formating the figure 
ax.vlines(v_index, colors = 'black',\
          linestyles = 'dashed', ymin = 0, ymax = 139,\
              linewidths = 1.8)
ax.hlines(h_index, colors = 'black',\
          linestyles = 'dashed', xmin = 0, xmax = 1378,\
              linewidths = 1.8)
ax.set_xticks(v_index)
ax.xaxis.set_major_formatter(ticker.NullFormatter())
ax.xaxis.set_minor_locator(ticker.FixedLocator(v_index_minor))
ax.xaxis.set_minor_formatter(ticker.FixedFormatter(range(1,len(v_index))))
ax.set_yticks(h_index)
ax.yaxis.set_major_formatter(ticker.NullFormatter())
ax.yaxis.set_minor_locator(ticker.FixedLocator(h_index_minor))
ax.yaxis.set_minor_formatter(ticker.FixedFormatter([' ','MES']))
plt.setp( ax.yaxis.get_minorticklabels(), rotation=90, va='center' )
ax.set_xlabel('Species groups')
ax.set_ylabel('Samples')
fig.tight_layout()



# B biome indicator 

label_var = np.unique(I[:,0])
tem =  np.matmul(parma_mean['A_b'],parma_mean['L_sp'].transpose()).transpose()
    
avg_effect_b = np.zeros((len(label_var), len(v_index) -1))
for i in range(len(v_index)-1):
     avg_effect_b[:,i] = np.mean(tem[v_index[i]:v_index[i+1]], axis =0) 
     
avg_effect_b = pd.DataFrame(avg_effect_b)
avg_effect_b.index = label_var
avg_effect_b.columns = range(1,6)

fig, ax = plt.subplots(1,5, dpi = 200, sharey = True, figsize = (10,2))
for i in range(1,6):
    avg_effect_b.plot( y=i, kind="barh", ax =ax[i-1], legend = False, fontsize = 4)
    plt.xticks(rotation = 90)


# # SNS Bar plot for the composition of each group 
# import copy 
# tax_name2 = copy.copy(tax_name.iloc[c_ord,:])
# tem = np.array([None]*tax_name2.shape[0])
# tem[range(65)] = 1
# tem[range(135,326)] = 2
# tem[range(326, 339)] = 3
# tem[range(381, 590)] = 4
# tem[range(1026, 1378)] = 5
# tax_name2['Group'] = tem
# tax_name3 = tax_name2[tem!=None]



# # Show confusion matrix and present the 
# confusion_matrix = pd.crosstab(tax_name3["Group"], tax_name3['ECR'],\
#                                rownames=['Actual'], colnames=['Predicted'],\
#                                    dropna = False, normalize = 'index')
# fig, ax = plt.subplots(figsize=(15, 10))
# sns.heatmap(confusion_matrix, annot=True, ax = ax)
# selected_species = [i for i in list(confusion_matrix.columns) if confusion_matrix[i].sum() > 0.05]
# selected_species.remove('Other')
# tax_name3 = tax_name3[tax_name3['ECR'].isin(selected_species)]


# ax = sns.catplot(x="ECR", kind="count", hue = 'Group', data=tax_name3, 
#             height=5, aspect=2.2)
# ax.set_xticklabels(rotation=90, fontsize = 14)
# ax.set_yticklabels(fontsize = 14)
# plt.axhline(y = 8, color='r', ls = '--', lw = 0.99)




fig = plt.figure(constrained_layout=True, dpi = 200, figsize = (10,6))
G = plt.GridSpec(ncols=5, nrows=4, figure=fig)
for i in range(1,6):
    temp = tax_name3['ECR'][tax_name3['Group'] == i ].values
    vals, counts = np.unique(temp, return_counts=True)
    sel_var = vals[counts.argsort()][-5:]
    #temp[np.invert(np.isin(temp, sel_var))] = 'Others'
    temp = temp[np.isin(temp, sel_var)] 
    vals, counts = np.unique(temp, return_counts=True)
    col_pie = [species_col_dict[i] for i in vals]
    ax = fig.add_subplot(G[0, i-1])
    ax.pie(x=100*counts/sum(counts), autopct="%.1f%%", labels=None,\
                colors = col_pie, textprops={'fontsize': 4})
    ax.set_xlabel('G' + str(i) + ': ' + str(temp.shape[0]), fontsize=4)
    if i == 1:
        ay = fig.add_subplot(G[1, i-1])
        avg_effect.plot( y=i, kind="barh", ax =ay, legend = False, fontsize = 4)
        az = copy.copy(ay)
    else:
        ay = fig.add_subplot(G[1, i-1])
        avg_effect.plot( y=i, kind="barh", ax =ay, legend = False, fontsize = 4, sharey = az)
    
    if i == 1:
        ay = fig.add_subplot(G[2, i-1])
        avg_effect_b.plot( y=i, kind="barh", ax =ay, legend = False, fontsize = 4)
        ab = copy.copy(ay)
    else:
        ay = fig.add_subplot(G[2, i-1])
        avg_effect_b.plot( y=i, kind="barh", ax =ay, legend = False, fontsize = 4, sharey = ab)

fig, ax  = plt.subplots(dpi = 100, figsize = (10,1))  
f = lambda m,c: ax.plot([],[],marker=m, color=c, ls="none")[0]
label = selected_species 
handles = [f("s", species_col_dict[i]) for i in selected_species]
ax.legend(handles, label, loc=4, framealpha=1, frameon=False, ncol = 5)
ax.set_axis_off()
        
# ax = fig.add_subplot(G[3, :]) 
# f = lambda m,c: ax.plot([],[],marker=m, color=c, ls="none")[0]
# label = selected_species 
# handles = [f("s", species_col_dict[i]) for i in selected_species]
# ax.legend(handles, label, loc=4, framealpha=1, frameon=False, ncol = 5)
# ax.set_axis_off()

