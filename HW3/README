The third assignment analyses customer churn behavior using parametric survival 
models and estimates Customer Lifetime Value (CLV) based on predicted survival 
probabilities. The dataset represents telecom subscribers with demographic, 
service usage, and churn information.


Objectives
We must fit multiple Accelerated Failure Time (AFT) models using different distributions.
Compare the model performances using (AIC, BIC, log-likelihood) and select the best model.
Then we must interpret coefficients of the final survival model.
Estimate Customer Lifetime Value (CLV) from predicted survival curves.
Identify at-risk customer segments and propose targeted retention strategies.

Methods
Survival models were fitted using survreg and flexsurvreg. Best model selected based on AIC and 
interpretability was LogNormal AFT. CLV was computed using monthly survival probabilities 
and discounted monthly margin. Segmentation was explored across service categories, 
subscription bundles, demographics, and risk thresholds.

Key Findings
Service tier and bundling (voice + internet) have the strongest impact on churn 
and CLV. For the Basic plan, the customers show higher churn probability and lower CLV.
Plus, E-service, and Total service customers exhibit longer expected lifetime and 
thus have a higher value.
A targeted retention budget of about 15% of expected at-risk CLV loss is more cost-effective than 
blanket incentives that may lead to unnecessary expenditures.