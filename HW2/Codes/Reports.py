####### Bandit Experimennt

import warnings
warnings.filterwarnings("ignore")
from Bandit import *

## Experiment Parameters
# Defining the number of trials and the true win rates for each bandit arm.


Bandit_Reward = [1, 2, 3, 4]
NumberOfTrials = 20000

## Running the Epsilon-Greedy

res = EpsilonGreedy(Bandit).experiment(Bandit_Reward, NumberOfTrials)


EpsilonGreedy(Bandit).report(NumberOfTrials, res, algorithm = "EpsilonGreedy")

Visualization().plot1(NumberOfTrials, res, 'Epsilon Greedy')


## Running the Thompson Sampling

result1 = ThompsonSampling(Bandit).experiment(Bandit_Reward, NumberOfTrials)

Visualization().plot1(NumberOfTrials, result1, algorithm = 'Thompson Sampling')
ThompsonSampling(Bandit).report(NumberOfTrials, result1, 'ThompsonSampling')

Visualization().plot2(res, result1)

## Comparing
comparison(NumberOfTrials, res, result1)