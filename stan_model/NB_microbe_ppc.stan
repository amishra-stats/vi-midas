data {
// here we have implemented the model estimation with holdout index for training and testing  
    int n; // number of observation 
    int q; // number of OUT takein into account
    int p; // number of geo-chemical covariates 
    int l; // number of latent variable 
    int s; // number of seasonal indicators
    int b; // number of biome indicators 
    int m; // number of monnth indicators 
    real sp_mean; // regularize in mean parameter of the model 
    real sp_var; // regularize in variance parameter of the model 
       
    int<lower=0> Y[n,q];  // observed species count data
    row_vector[p] X[n];  // observed covariates
    row_vector[s] S[n];  // observed seasonal indicator model matrix 
    row_vector[b] B[n];  // observed biome indicator model matrix  
    row_vector[m] Q[n];  // observed biome indicator model matrix  
    matrix[n,q] Yi; // Indicator variable for species presence absence data
    row_vector[n] T;  // library size of the sample  
    row_vector[n] Bs;  // basket size of each sample  
    row_vector[q] holdout[n]; // holdout sample index for testing/training 
}
parameters{
    // mean parameters 
    row_vector[q] C0;  // intercept term in the model 
    vector[p] C_geo[q];    // coefficient matrix geo-chemical factors 
    matrix[q,l] L_sp;    // coefficient matrix latent variable for the species
    vector[l] L_i[q];  // latent variable for the interaction coefficient 
    matrix[s,l] A_s;  // latent variable for seasonal indicators  
    matrix[b,l] A_b;  // latent variable for biome indicators 
    matrix[m,l] A_m;  // latent variable for biome indicators 
    vector<lower = 0, upper = 2>[q] tau; // positive variable for the shape parameter
    vector<lower = 0>[q] phi; // dispersion parameter of the negative binomial regression
}
model {
  row_vector[q] mu;
  row_vector[l] temp;
  for(j in 1:q){
      C_geo[j] ~ double_exponential(0, sp_mean); //  make it sparse 
      L_i[j]   ~  double_exponential(0, sp_mean);  //  make it sparse 
      L_sp[j,]   ~  double_exponential(0, sp_mean);  // make it sparse 
  }
  to_vector(A_s)   ~  double_exponential(0, sp_mean);  // make it sparse 
  to_vector(A_b)   ~  double_exponential(0, sp_mean);  // sparse 
  to_vector(A_m)   ~  double_exponential(0, sp_mean);  // sparse 
  phi   ~  normal(0, sp_var);

  for(i in 1:n){
    temp = Yi[i, ] * L_sp;
    for(j in 1:q){
        if (holdout[i,j] == 0){
            // generates mean parameters 
            mu[j] = C0[j] + X[i]*C_geo[j] + S[i]*(A_s*(L_sp[j,]')) + B[i]*(A_b*(L_sp[j,]')) + Q[i]*(A_m*(L_sp[j,]'));
            if (Yi[i,j] != 0){ 
                // add species-species interaction 
                mu[j] = mu[j] + (temp - L_sp[j,])*L_i[j]/(Bs[i]-1.0);
            }
            // compute log-likelihood 
            target += neg_binomial_2_log_lpmf( Y[i,j] | log(T[i]) + tau[j]*mu[j],1/sqrt(phi[j]));
        }
    }
  }
}