"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""

############################### Logging ###############################
# Logger levels: TRACE < DEBUG < INFO < WARNING < ERROR < CRITICAL
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import ttest_ind
import os


CODES_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(CODES_DIR, ".."))

REPORT_DIR = os.path.join(BASE_DIR, "Report")
IMAGES_DIR = os.path.join(BASE_DIR, "Images")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------------------
# Helper Functions
# ---------------------------
def log_trial_progress(trial, N, chosen_index, bandit, rewards):
    """
    Logs progress for each trial.
    """
    cumulative_reward = np.sum(rewards[:trial])
    logger.info(f"Trial {trial}/{N}: Chosen bandit {chosen_index}, Estimated win rate {bandit.p_estimate:.2f}, "
                f"Reward {rewards[trial-1]:.2f}, Cumulative reward {cumulative_reward:.2f}")
    


def _sanitize_filename(name: str) -> str:
    import re
    name = str(name)
    name = re.sub(r'[\\/:*?"<>|]+', '', name) 
    name = name.replace(' ', '_')
    return name

def save_report_data(algorithm, chosen_bandit, reward, bandits):
    """
    Saves experiment and final results in CSV files.
    Returns (experiment_csv_path, final_csv_path).
    """
    try:
        algo_safe = _sanitize_filename(algorithm)
        exp_path = os.path.join(REPORT_DIR, f"{algo_safe}_Result_Experiment.csv")
        final_path = os.path.join(REPORT_DIR, f"{algo_safe}_Result_Final.csv")

        chosen_bandit_list = np.asarray(chosen_bandit).tolist()
        reward_list = np.asarray(reward).tolist()

        data_df = pd.DataFrame({
            'Bandit': chosen_bandit_list,
            'Reward': reward_list,
            'Algorithm': algo_safe})
        
        data_df.to_csv(exp_path, index=False)
        logger.info(f"[{algo_safe}] Wrote per-trial CSV to: {exp_path}")

        data_df1 = pd.DataFrame({
            'Bandit': list(range(len(bandits))),
            'Reward': [float(b.p_estimate) for b in bandits],
            'Algorithm': algo_safe})
        
        data_df1.to_csv(final_path, index=False)
        logger.info(f"[{algo_safe}] Wrote final snapshot CSV to: {final_path}")

        return exp_path, final_path

    except Exception as e:
        logger.error(f"Failed to write CSVs for algorithm '{algorithm}': {e}")
        return None, None



def log_report_statistics(algorithm, bandits, reward, cumulative_regret, N,
                          cumulative_reward_average=None, cumulative_reward=None,
                          count_suboptimal=None):
    for i, b in enumerate(bandits):
        logger.info(f'Bandit {i}: True Win Rate {b.p} - Pulled {b.N} times - '
                    f'Estimated avg reward {round(b.p_estimate, 4)} - '
                    f'Estimated avg regret {round(b.r_estimate, 4)}')
    logger.info(f"Cumulative Reward : {np.sum(reward)}")
    if cumulative_reward_average is not None:
        logger.info(f"Cumulative Reward Average (last value): {cumulative_reward_average[-1]}")
    if cumulative_reward is not None:
        logger.info(f"Cumulative Reward (last value): {cumulative_reward[-1]}")
    logger.info(f"Cumulative Regret : {cumulative_regret[-1]}")
    if count_suboptimal is not None:
        logger.info(f"Percent suboptimal : {round((float(count_suboptimal) / N), 4)}")



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#

class Visualization:
    """
    Visualization class for plotting results of bandit experiments.
    """

    def plot1(self, N, results, algorithm='EpsilonGreedy'):
        """
        Plots performance (both linear and log scale) of the bandit algorithm.
        """
        cumulative_reward_average = results[0]
        bandits = results[3]
        
        # linear
        plt.figure()
        plt.plot(cumulative_reward_average, label='Cumulative Average Reward')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Linear Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        _out_linear = os.path.join(IMAGES_DIR, f"{algorithm}_winrate_linear.png")
        plt.savefig(_out_linear, bbox_inches="tight")
        logger.info(f"Saved {algorithm} linear win-rate plot: {_out_linear}")
        plt.close()

        # logarithmic
        plt.figure()
        plt.plot(cumulative_reward_average, label='Cumulative Average Reward')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Log Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.xscale("log")
        _out_log = os.path.join(IMAGES_DIR, f"{algorithm}_winrate_log.png")
        plt.savefig(_out_log, bbox_inches="tight")
        logger.info(f"Saved {algorithm} log win-rate plot: {_out_log}")
        plt.close()

    def plot2(self, results_eg, results_ts):
        """
        Compares Epsilon-Greedy and Thompson Sampling algorithms.
        """
        cumulative_rewards_eps = results_eg[1]
        cumulative_rewards_th = results_ts[1]
        cumulative_regret_eps = results_eg[2]
        cumulative_regret_th = results_ts[2]

        # cumulative rewards
        plt.figure()
        plt.plot(cumulative_rewards_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_rewards_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Reward Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        _out_cr_lin = os.path.join(IMAGES_DIR, "Comparison_cumulative_reward_linear.png")
        plt.savefig(_out_cr_lin, bbox_inches="tight")
        logger.info(f"Saved comparison cumulative reward (linear): {_out_cr_lin}")
        plt.close()

        # cumulative regerts
        plt.figure()
        plt.plot(cumulative_regret_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_regret_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Regret Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regret")
        _out_cg_lin = os.path.join(IMAGES_DIR, "Comparison_cumulative_regret_linear.png")
        plt.savefig(_out_cg_lin, bbox_inches="tight")
        logger.info(f"Saved comparison cumulative regret (linear): {_out_cg_lin}")
        plt.close()

        # cumulative rewar (logarithm)
        plt.figure()
        plt.plot(cumulative_rewards_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_rewards_th, label='Thompson Sampling')
        plt.legend()
        plt.yscale("log")
        plt.title("Cumulative Reward Comparison Log Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        _out_cr_log = os.path.join(IMAGES_DIR, "Comparison_cumulative_reward_log.png")
        plt.savefig(_out_cr_log, bbox_inches="tight")
        logger.info(f"Saved comparison cumulative reward (log): {_out_cr_log}")
        plt.close()

        # cumulative regret (logarithm)
        plt.figure()
        plt.plot(cumulative_regret_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_regret_th, label='Thompson Sampling')
        plt.legend()
        plt.yscale("log")
        plt.title("Cumulative Regret Comparison Log Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regret")
        _out_cg_log = os.path.join(IMAGES_DIR, "Comparison_cumulative_regret_log.png")
        plt.savefig(_out_cg_log, bbox_inches="tight")
        logger.info(f"Saved comparison cumulative regret (log): {_out_cg_log}")
        plt.close()


#--------------------------------------#
class BanditArm(Bandit):
    """
    A simple bandit arm.
    """
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0 
        self.r_estimate = 0
        logger.info(f"Initialized Bandit with true win rate: {self.p}")
    
    def __repr__(self):
        return f'An Arm with {self.p} Win Rate'
   
    def pull(self):
        return np.random.random() < self.p
    
    def update(self, x, iteration=None):
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N
        self.r_estimate = self.p - self.p_estimate
    
    def experiment(self):
        pass

    def report(self, N, results, algorithm="BanditArm"):
        cumulative_reward_average, cumulative_reward, cumulative_regret, bandits, chosen_bandit, reward = results 
        save_report_data(algorithm, chosen_bandit, reward, bandits)
        log_report_statistics(algorithm, bandits, reward, cumulative_regret, N)

#--------------------------------------#
class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy multi-armed bandit algorithm.
    """
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0 
        self.r_estimate = 0
        logger.info(f"Initialized EpsilonGreedy Bandit with true win rate: {self.p}")
    
    def __repr__(self):
        return f'An Arm with {self.p} Win Rate'

    def pull(self):
        # Reward is from the normal distribution with mu = self.p, sigma^2 = 1
        reward = np.random.randn() + self.p
        return reward

    def update(self, x, iteration=None):
        old_estimate = self.p_estimate
        self.N += 1
        self.p_estimate = (1 - 1.0/self.N) * self.p_estimate + (1.0/self.N) * x
        self.r_estimate = self.p - self.p_estimate

    def experiment(self, BANDIT_REWARDS, N, t=1):
        logger.info(f"Starting EpsilonGreedy experiment with {len(BANDIT_REWARDS)} bandits and {N} trials.")

        bandits = [EpsilonGreedy(p) for p in BANDIT_REWARDS]
        means = np.array(BANDIT_REWARDS)
        true_best = np.argmax(means)
        best_mu = float(np.max(means))
        count_suboptimal = 0
        EPS = 1 / t

        reward = np.empty(N)
        chosen_bandit = np.empty(N)

        for i in range(N):
            p_rand = np.random.random()
            if p_rand < EPS:
                j = np.random.choice(len(bandits))
            else:
                j = np.argmax([b.p_estimate for b in bandits])

            x = bandits[j].pull()
            bandits[j].update(x, iteration=i)

            if j != true_best:
                count_suboptimal += 1

            reward[i] = x
            chosen_bandit[i] = j
            
            t += 1
            EPS = 1 / t

            # Logging our progress per 2000 iteration
            if (i + 1) % 2000 == 0:
                log_trial_progress(i + 1, N, j, bandits[j], reward)

        cumulative_reward_average = np.cumsum(reward) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(reward)
        
        cumulative_regret = np.empty(N)
        for i in range(N):
            cumulative_regret[i] = (i + 1) * best_mu - cumulative_reward[i]

        return (cumulative_reward_average, cumulative_reward, cumulative_regret, bandits, chosen_bandit, reward, count_suboptimal)
    
    def report(self, N, results, algorithm="EpsilonGreedy"):
        (cumulative_reward_average, cumulative_reward, cumulative_regret, bandits,
         chosen_bandit, reward, count_suboptimal) = results 
        exp_path, final_path = save_report_data(algorithm, chosen_bandit, reward, bandits)
        logger.info(f"{algorithm} CSV paths: exp={exp_path}, final={final_path}")

        log_report_statistics(algorithm, bandits, reward, cumulative_regret, N, cumulative_reward_average, cumulative_reward, count_suboptimal)

#--------------------------------------#
class ThompsonSampling(Bandit):
    """
    Thompson Sampling multi-armed bandit algorithm.
    """
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0 
        self.r_estimate = 0
        self.lambda_ = 1  # Initial precision of the prior
        self.tau = 1  # Precision of the reward noise
        logger.info(f"Initialized ThompsonSampling Bandit with true win rate: {self.p}")

    def __repr__(self):
        return f'An Arm with {self.p} Win Rate'

    def pull(self):
        reward = np.random.randn() / np.sqrt(self.tau) + self.p
        return reward
    
    def sample(self):
        # Sample from the normal distribution with mu = p_estimate, sigma = 1/lambda
        sample_val = np.random.randn() / np.sqrt(self.lambda_) + self.p_estimate
        return sample_val

    def update(self, x, iteration=None):
        old_estimate = self.p_estimate
        self.p_estimate = (self.tau * x + self.lambda_ * self.p_estimate) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1
        self.r_estimate = self.p - self.p_estimate
    
    def plot(self, bandits, trial):
        x = np.linspace(-1, 1, 200)
        logger.info(f"Plotting bandit distributions after {trial} trials.")
        for b in bandits:
            y = norm.pdf(x, b.p_estimate, np.sqrt(1. / b.lambda_))
            plt.plot(x, y, label=f"Real mean: {b.p:.4f}, Plays: {b.N}")
        plt.title(f"Bandit distributions after {trial} trials")
        plt.legend()
        out_path = os.path.join(IMAGES_DIR, f"TS_dists_trial_{trial}.png")
        plt.savefig(out_path, bbox_inches="tight")
        logger.info(f"Saved TS distribution figure: {out_path}")
        plt.close()

    def experiment(self, BANDIT_REWARDS, N):
        logger.info(f"Starting ThompsonSampling experiment with {len(BANDIT_REWARDS)} bandits and {N} trials.")
        bandits = [ThompsonSampling(rate) for rate in BANDIT_REWARDS]
        best_mu = max([b.p for b in bandits])

        sample_points = [5, 20, 50, 100, 200, 500, 1000, 1999, 5000, 19999]
        reward = np.empty(N)
        chosen_bandit = np.empty(N)
        
        for i in range(N):
            j = np.argmax([b.sample() for b in bandits])
            
            if i in sample_points:
                self.plot(bandits, i)

            x = bandits[j].pull()
            bandits[j].update(x, iteration=i)

            reward[i] = x
            chosen_bandit[i] = j

            if (i + 1) % 2000 == 0:    #again tracking our logs per every 2000 iteration
                log_trial_progress(i + 1, N, j, bandits[j], reward)
    
        cumulative_reward_average = np.cumsum(reward) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(reward)
        cumulative_regret = np.empty(N)
        
        for i in range(N):
            cumulative_regret[i] = (i + 1) * best_mu - cumulative_reward[i]

        logger.info("ThompsonSampling experiment completed.")
        return (cumulative_reward_average, cumulative_reward, cumulative_regret, bandits, chosen_bandit, reward)

    def report(self, N, results, algorithm="ThompsonSampling"):
        (cumulative_reward_average, cumulative_reward, cumulative_regret, bandits,
         chosen_bandit, reward) = results
        exp_path, final_path = save_report_data(algorithm, chosen_bandit, reward, bandits)
        logger.info(f"{algorithm} CSV paths: exp={exp_path}, final={final_path}")

        log_report_statistics(algorithm, bandits, reward, cumulative_regret, N, cumulative_reward_average, cumulative_reward)

#--------------------------------------#


def comparison(num_trials, eps_results, thompson_results):
    """
    Compare performance of EpsilonGreedy and ThompsonSampling algorithms visually.
    
    :param num_trials: Total number of trials.
    :param eps_results: Tuple of experiment results from EpsilonGreedy.
    :param thompson_results: Tuple of experiment results from ThompsonSampling.
    """

    cum_avg_reward_eps = eps_results[0]
    cum_avg_reward_thompson = thompson_results[0]
    eps_bandit_list = eps_results[3]
    rewards_eps = eps_results[5]
    rewards_thompson = thompson_results[5]
    final_regret_eps = eps_results[2][-1]
    final_regret_thompson = thompson_results[2][-1]

    logger.info(f"Total Reward EpsilonGreedy: {np.sum(rewards_eps)}")
    logger.info(f"Total Reward ThompsonSampling: {np.sum(rewards_thompson)}")
    logger.info("")
    logger.info(f"Total Regret EpsilonGreedy: {final_regret_eps}")
    logger.info(f"Total Regret ThompsonSampling: {final_regret_thompson}")

    plt.figure(figsize=(12, 5))

    # linear plots
    plt.subplot(1, 2, 1)
    plt.plot(cum_avg_reward_eps, label='Cum. Avg. Reward (EpsilonGreedy)')
    plt.plot(cum_avg_reward_thompson, label='Cum. Avg. Reward (ThompsonSampling)')
    plt.plot(np.ones(num_trials) * max([bandit.p for bandit in eps_bandit_list]), label='Optimal Reward')
    plt.legend()
    plt.title("Win Rate Convergence - Linear Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")

    # logarithmic scales
    plt.subplot(1, 2, 2)
    plt.plot(cum_avg_reward_eps, label='Cum. Avg. Reward (EpsilonGreedy)')
    plt.plot(cum_avg_reward_thompson, label='Cum. Avg. Reward (ThompsonSampling)')
    plt.plot(np.ones(num_trials) * max([bandit.p for bandit in eps_bandit_list]), label='Optimal Reward')
    plt.legend()
    plt.title("Win Rate Convergence - Log Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")
    plt.xscale("log")
    
    plt.tight_layout()
    _out = os.path.join(IMAGES_DIR, "Comparison_winrate_linear_and_log.png")
    plt.savefig(_out, bbox_inches="tight")
    logger.info(f"Saved comparison win-rate (linear & log) figure: {_out}")
    plt.close()



#--------------------------------------#
if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    # running the experiment
    BANDIT_REWARDS = [1, 2, 3, 4]
    N = 20_000

    # Epsilon-Greedy
    eg_runner = EpsilonGreedy(0.0)  
    eg_results = eg_runner.experiment(BANDIT_REWARDS, N, t=1)
    eg_runner.report(N, eg_results, algorithm="EpsilonGreedy")

    # Thompson Sampling
    ts_runner = ThompsonSampling(0.0)
    ts_results = ts_runner.experiment(BANDIT_REWARDS, N)
    ts_runner.report(N, ts_results, algorithm="ThompsonSampling")

    viz = Visualization()
    viz.plot1(N, eg_results, algorithm='EpsilonGreedy')
    viz.plot1(N, ts_results, algorithm='ThompsonSampling')
    viz.plot2(eg_results, ts_results)

    # side by side comparison of the two algo.s
    comparison(N, eg_results, ts_results)

try:
    files_now = os.listdir(REPORT_DIR)
    logger.info(f"Report dir: {REPORT_DIR}")
    logger.info(f"Files present: {files_now}")
except Exception as e:
    logger.error(f"Could not list Report dir '{REPORT_DIR}': {e}")
