# HW2 - Multi-Armed Bandit Experiments


In this assignment we should implement and compare two algorithms used for A/B testing in a 4-armed bandit environment:

Epsilon-Greedy (with decaying exploration: ε = 1/t)

Thompson Sampling (Bayesian inference with known precision)

The goal is to analyse how each algorithm balances exploration and exploitation, and compare their performance over repeated trials. In this case, the number of trials is 20.000.

Each algorithm attempts to learn which arm provides the highest expected reward and maximise cumulative reward over time.

Outputs

The program automatically generates:

- CSV files in the Report/ folder
- Plots saved in the Images/ folder
- Logging of progress using loguru

Generated plots include:

Learning curve for each algorithm (both linear and logarithmic scale), cumulative reward comparison as well as cumulative regret comparison. Where, the regret measures how much reward was lost compared to always pulling the best arm.



Insights from the comparisons:
*Cumulative Reward Comparison*

The cumulative reward curves for Epsilon-Greedy and Thompson Sampling overlap almost perfectly. Both algorithms eventually exploit the optimal arm most of the time.
Sure this is great, but are both as efficient? The answer is no. If we analyse the cumulative regret comparison, we may notice that the Thompson Sampling is better early on because it smartly explores using Bayesian belief updates. Unlike, the Epsilon-Greedy that improves only once ε decays, so it reacts slower but eventually locks in strongly.
This explains why your regret curves cross over. And if we pay attention to the learning curves (the win rate convergence plots). Thompson Sampling rises smoothly, learning gradually. Whereas, Epsilon-Greedy initially fluctuates quite a lot because ε is large at the beginning (ε = 1/t). Eventually, both converge tightly around 4, showing successful identification of the best arm.



Bonus: A useful improvement would be to run each algorithm multiple times with different random seeds and compare the average cumulative regret. 
This would make the comparison more reliable by reducing the effect of randomness from a single run. 

Another meaningful extension would be to include the UCB1 (Upper Confidence Bound) algorithm as a third baseline. 
UCB is generally simpler to implement than Thompson Sampling because it does not require Bayesian updating — it just adds a confidence bonus to each arm’s estimated reward to guide exploration. 
However, while UCB is more straightforward, Thompson Sampling typically learns more efficiently and more quickly in noisy environments and adapts faster. 
Including UCB would therefore provide a clearer contrast between the three algorithms used.

