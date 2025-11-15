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
High-tier plans such as Plus, E-service, and Total significantly increase customer 
lifetime, while voice and internet subscriptions shorten it (time ratios < 1).
Older and more settled customers stay longer, whereas Basic + Internet users show 
the lowest CLV and the highest churn risk (~35%). E-service customers without internet 
have the highest CLV and the lowest churn (~1–2%), and demographics like gender, 
education, and region have minimal impact. Most at-risk subscribers (S(12) ≤ 0.5) 
are in the Basic tier, with an expected churn loss of 244k–326k AMD; targeting only 
these customers lowers the required retention budget to around 29,565 AMD. Overall, 
high-tier non-internet customers are the most valuable, and retention efforts should 
prioritise Basic customers with bundled internet or voice due to their high churn and low CLV.