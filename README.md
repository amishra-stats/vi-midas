# VI-MIDAS: a variational inference for microbiome survey data analysis
We present a probabilistic multivariate framework VI-MIDAS to jointly model the microbial abundance data as outcomes in terms of components related to host/environment-associated factors and species-species interactions.   The framework uniquely expresses the interactions among species in terms of the species-speciefic latent vectors of a fixed length. 

![VI-MIDAS](http://imgur.com/a/PCfhsOT.png)

Estimating the posterior distribution of the model parameters is challenfing because of intractable marginal distribution of the data. Using the framework of variational inference, we minimize the KL divergence, i.e., maximize evidence lower bound (ELBO) to obtain the variational posterior of the model parameters under the mean-field assumption. 

Data analysis using the framework of VI-MIDAS have the following steps: 

+ Specify generative model 
+ Hyperparameter Tuning  
+ Evaluate the contribution of each component via component excluded generative model 
+ Model sensitivity to initialization
+ Model Analysis 
+ Model inference 

---
All the python modules used in the analysis of data are provided in the **requirements.txt** file. 


---
## Generative model
We propose a generative model for the ocean microbiome survey data from the Tara Ocean Expedition. The model can be extended for the gut microbiome data as well. 

Tara ocean expedition has collected 139 samples from different geographical locations in the ocean across the globe to understand the role and functions of the global ocean microbiome. Sample associated factors/covariates include geochemical features and spatio-temporal features including province (location), depth (Biome) and time (year quarters) indicators. Hence, MEM for the ocean microbiome data has five components identified as a geochemical component, province component, biome component, time component, and finally species-species interaction components. Please see the manuscript for the details of the generative model. 
  
+ Stan script for the generative model: **stan_model/NB_microbe_ppc.stan**



[//]: # ( User needs to define a generative model for the microbial abundance data as outcome with an embedding component that account for the interactions among the microbial species. Here we propose a generative model for the ocean microbiome data from the Tara Ocean Expedition. The model can be extended for the gut microbiome data as well. Please see the manuscript for the generative model.)

##### Interaction component
Following the factor model approach, we characterize each of the microbial species in terms of the *shared* species-specific latent vectors _&beta;_ and other species-specific latent vectors _&rho;_. Our approach then uses the species-specific latent vectors to define the interaction component.  Please see the manuscript for the details of formulation and the Stan script for implementation.



---
## Hyperparameter tuning 
Here we evaluate the randomly selected sets of hyperparameters of the MEM model in terms of the **out of sample log-likelihood predictive density (LLPD)**. Supporting files for the step:

+ **hyperparameter_tuning_fit.py**
+ **hyperparameter_tuning_analysis.ipynb**


We compute the variational posterior on the training sample and evaluate the output on the test sample using the python script **hyperparameter_tuning_fit.py** for a given choice of the hyperparameters. Please follow the detailed instruction in the jupyter notebook file **hyperparameter_tuning_analysis.ipynb** to understand the procedure and analyze/compare the outputs. 





---
## Components contribution 

After selecting the hyperparameter, we aim to evaluate the contribution of each of the components in the generative model (MEM). The generative model for the ocean microbiome data for the microbial abundance data consists of **geochemical component**, **spatio-temporal components indicating geographical location, ocean depth, and time**, and **species-species interaction component**. To understand their contribution in the MEM, we drop each of the specific components in the generative model, estimate the variational posterior and then evaluate the model performance in terms of the **out of sample log-likelihood predictive density**. Supporting files for the step:

+ **component_contribution_fit.py**
+ **component_contribution_analysis.ipynb**

We have defined the component excluded stan model (see stan_model folder) in the following files:
 + **NB_microbe_ppc.stan** : Full model [Model = 0]
 + **NB_microbe_ppc-1.stan** : Province component dropped  [Model = 1]
 + **NB_microbe_ppc-2.stan** : Biome component dropped  [Model = 2]
 + **NB_microbe_ppc-3.stan** : Quarter/Time component dropped  [Model = 3]
 + **NB_microbe_ppc-G.stan** : Geochemical component dropped  [Model = 4]
 + **NB_microbe_ppc_nointer.stan** : Species-species interaction component dropped  [Model = 5]

We evaluate the component excluded generative model by computing the variational posterior on the training sample and evaluate the output on the test sample using the python script **component_contribution_fit.py** for the selected hyperparameters. Please follow the detailed instruction in the jupyter notebook file **component_contribution_analysis.ipynb** to understand the procedure and analyze/compare the model. 


---
## Model sensitivity to initialization
Maximizing the ELBO is a non-convex optimization problem. The parameters estimate are sensitive to the choice of their initial estimates. Hence, we further evaluate the chosen set of hyperparameters for 50 random initialization and then select the best model out of it. Supporting files for the step:

+ **model_sensitivity_fit.py**
+ **model_sensitivity_analysis.ipynb**

We evaluate the MEM with different initialization by computing the variational posterior on full data and evaluating the output in terms of **LLPD** using the python script **model_sensitivity_fit.py** for the selected hyperparameters. Please follow the detailed instruction in the jupyter notebook file **model_sensitivity_analysis.ipynb** to understand the procedure and select the model with the highest LLPD on full data. 


---
## Model output analysis   

Now we analyze the best model parameters estimate obtained after sensitivity analysis. Supporting files for the step: 

+ **model_analysis1.ipynb**: Data visual summary, numerical measure of model validity 
+ **model_analysis2.ipynb**: ELBO convergence, Observed vs predicted outcome using scatter plot and heatmaps,  Hitogram for comparing distributions
+ **model_analysis3.ipynb**: Heatmap showing contribution of the individual components in the MEM



---
## Model output inference 

Now, we make inferences based on the estimated parameters.  Supporting files for the step: 

+ **model_inference1.ipynb**: Summarizing the effect of geochemical covariates and biome component on the microbial species
+ **model_inference2.ipynb**: Plots showing the modules of similar species
+ **model_inference3.ipynb**: Plots showing the mutualistic and competitive interactions among species



## Queries
Please contact authors and creators for any queries related to using the analysis 


-   Aditya Mishra: [mailto](mailto:amishra@flatironinstitute.org)
-   Christian Mueller: [mailto](mailto:cmueller@flatironinstitute.org)
