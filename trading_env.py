######################################################## FILE HEADER ########################################################
# This file defines the Trading Environment
# This file defines key aspects of the model, mainly the step function which contains the reward function logic

# Below are the imports required for this file
import numpy as np
import gym
from gym import spaces
import random
from gym.spaces import Box

# This class handles the Trading Environment and is the core of this file
class TradingEnv(gym.Env):

    # Initializing function
    ### raw_prices - raw prices of a stock
    ### norm_log_returns - normalized log returns for a stock
    ### model_preds - list of 5 length arrays of stock predictions for 10,30,60,90,150 timesteps (days) in the future
    ### window_size - window size required to start stock assessment and trading
    ### initial_balance - initial cash balance of the portfolio
    def __init__(self, raw_prices, norm_log_returns, model_preds,
                 window_size=60, initial_balance=100000):
        super(TradingEnv, self).__init__()

        # sanity length check for the data to be used
        assert len(raw_prices) == len(norm_log_returns) == len(model_preds), "Mismatch in input lengths"

        # assigning raw prices, normalized log returns, model predictions, window size, initial cash balance
        self.raw_prices = raw_prices
        self.norm_log_returns = norm_log_returns
        self.model_preds = model_preds  # Shape: (N, 5)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # defining the continuous action space for buying and selling
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # length of the engineered observations required by the model
        obs_len = window_size + 12 + 5

        # defining the observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        # an 'episode' is a given timestep of the stock
        # defines the price, portfolio net worth, action, stock inventory, and reward lists across the episodes / timesteps
        self.episode_prices = []
        self.episode_net_worth = []
        self.episode_actions = []
        self.episode_inventory = []
        self.episode_rewards = []

        # initializing the previous action as hold
        self.prev_action = 0

        # calls the reset function
        self.reset()


    # resets the trading environment to its initial state
    def reset(self):

        # sets the current step (same as window size as a minimum data of length 'window_size' is needed to make trading decisions)
        self.current_step = self.window_size

        # initializes the cash balance, stock inventory, portfolio net worth, previous net worth (set as the initial net worth to start)
        self.balance = self.initial_balance
        self.inventory = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance

        # lists for recent percentage returns and trade actions initialized
        self.recent_returns = []
        self.trades = []

        # clearing the previous action; price, portfolio net worth, action, stock inventory, and reward lists
        self.prev_action = 0
        self.episode_prices = []
        self.episode_net_worth = []
        self.episode_actions = []
        self.episode_inventory = []
        self.episode_rewards = []

        # episode price appended at current step
        self.episode_prices.append(self.raw_prices[self.current_step])

        # initial net worth appended
        self.episode_net_worth.append(self.net_worth)

        # initial default action hold (0) appended
        self.episode_actions.append(0)

        # initial inventory (0) appended
        self.episode_inventory.append(self.inventory)

        # initial reward (0.0) appended
        self.episode_rewards.append(0.0)

        # lists storing percentage returns, sharpe ratios, sortino ratios, and over trading penalties initialized for testing purposes
        self.pct_returns_15 = []
        self.pct_returns_40 = []
        self.sharpe_ratio = []
        self.overtrading_penalty = []

        # get observation function is returned
        return self._get_observation()


    # constructs the current observation for the model
    def _get_observation(self):

        # normalized log return window of size 'window_size'
        price_window = self.norm_log_returns[self.current_step - self.window_size: self.current_step]

        # price window of size 'window_size'
        raw_window = self.raw_prices[self.current_step - self.window_size: self.current_step]

        # features of the raw_window computed by _compute_features function
        features = self._compute_features(raw_window)

        # model predictions of the current episode / timestep
        model_preds = self.model_preds[self.current_step]  # Shape (5,)

        # compiled observations
        obs = np.concatenate([price_window, features, model_preds]).astype(np.float32)

        # return the observations
        return obs

    # function that imports and uses the compute_all_features function
    ### prices - raw price window
    def _compute_features(self, prices):

        # imports function compute_all_features
        from utils.features import compute_all_features

        # returns the computed features
        return compute_all_features(prices)  # 12-dimensional

    # function that handles the buy logic
    ### amount_shares - number of shares to buy (integer)
    def _buy(self, amount_shares):

        # current price at given episode / timestep
        current_price = self.raw_prices[self.current_step]

        # total cost needed to buy necessary number of shares
        cost = amount_shares * current_price

        # if the cost is exceeding the cash in hand, reset cost to maximum shares that can be bough with cash in hand
        if cost > self.balance:
            amount_shares = self.balance // current_price
            cost = amount_shares * current_price

        # deduct balance cash
        self.balance -= cost

        # increase stock inventory with corresponding number of stocks
        self.inventory += amount_shares

        # append buy information to trades list
        self.trades.append(('buy', self.current_step, amount_shares))

    # function that handles the buy logic
    ### amount_shares - number of shares to sell (integer)
    def _sell(self, amount_shares):

        # current price at given episode / timestep
        current_price = self.raw_prices[self.current_step]

        # if the number of stocks to sell is greater than the stocks owned, reset it to the number of stocks owned itself
        if amount_shares > self.inventory:
            amount_shares = self.inventory

        # cash that will come in after selling the necessary number of stocks
        revenue = amount_shares * current_price

        # increase the balance cash
        self.balance += revenue

        # decrease the number of stocks in inventory
        self.inventory -= amount_shares

        # append sell information to trades list
        self.trades.append(('sell', self.current_step, amount_shares))

    # main function of this file that handles the step and reward function logic
    ### action - continuous action taken by the trading model
    def step(self, action):

        # initialize the reward to 0
        reward = 0

        # action to be taken, clipped within the range [-1 , 1]
        action = float(np.clip(action[0], -1.0, 1.0))

        # current stock price
        current_price = self.raw_prices[self.current_step]

        # maximum shares that can be bought at the given price and cash at hand
        max_shares = int(self.balance / current_price)

        # storing the previous inventory
        self.prev_inventory = self.inventory

        # executing the actual trade

        # buy
        if action > 0:

            # fraction of how much to buy
            buy_fraction = action

            # calculating the number of shares to buy as per the buy_fraction
            shares_to_buy = int(max_shares * buy_fraction)

            # buying the required number of shares
            self._buy(shares_to_buy)

        # sell
        elif action < 0:

            # fraction of how much to sell (made negative as absolute stock number to sell is needed)
            sell_fraction = -action

            # calculating the number of shares to sell as per the sell_fraction
            shares_to_sell = int(self.inventory * sell_fraction)

            # selling the required number of shares
            self._sell(shares_to_sell)

        # if action is 0, no trading needed

        # increment current_step
        self.current_step += 1

        # check whether the episode / timestep is over
        done = self.current_step >= len(self.raw_prices) - 1

        # updating the previous net worth
        self.prev_net_worth = self.net_worth

        # calculating the new net worth based on cash in hand and the new stock price
        self.net_worth = self.balance + self.inventory * self.raw_prices[self.current_step]

        # STARTING HERE IS THE REWARD LOGIC

        # a minimum history of 15 is required
        if len(self.episode_net_worth) >= 15:

            # calculates the average net worth over the last 15 timesteps
            avg_prev_net_worth = np.mean(self.episode_net_worth[-15:])

            # calculate the percentage return from the average of the last 15 timesteps to now
            pct_return_15 = float((self.net_worth - avg_prev_net_worth) / (avg_prev_net_worth + 1e-8))

        # if insufficient length, set pct_return_15 to 0
        else:
            pct_return_15 = 0.0



        # a minimum history of 40 is required
        if len(self.episode_net_worth) >= 40:

            # calculates the average net worth over the last 40 timesteps
            avg_prev_net_worth = np.mean(self.episode_net_worth[-40:])

            # calculate the percentage return from the average of the last 40 timesteps to now
            pct_return_40 = float((self.net_worth - avg_prev_net_worth) / (avg_prev_net_worth + 1e-8))

        # if insufficient length, set pct_return_40 to 0
        else:
            pct_return_40 = 0.0


        # TODO: try appending the 40 length returns to the recent_returns instead of the 15 length returns, maybe it is more stable
        # the recent_returns list will be updated / appended with the percentage returns for 15 timestep history
        self.recent_returns.append(float(pct_return_40))

        # a minimum history of 20 is required to calculate a stable sharpe ratio
        if len(self.recent_returns) >= 20:

            # mean return over the past 20 timesteps
            mean_return = np.mean(self.recent_returns[-20:])

            # standard deviation of the returns over the past 20 timesteps
            std_return = np.std(self.recent_returns[-20:]) + 1e-8

            # sharpe ratio calculation (risk-free return is set to 0 for simplicity)
            sharpe_ratio = mean_return / std_return

        # if insufficient length, set sharpe_ratio to 0
        else:
            sharpe_ratio = 0.0

        # standard overtrading penalty of -0.1 to discourage over trading
        overtrading_penalty = -1.5 if abs(action) >= 1e-2 else 0.0

        # clamping the numbers to ensure nothing explodes ;)
        # note: certain numbers have been multiplied by constants, the constants have been chosen after testing and debugging to ensure
        # a split like the following:
        # pct_return_15, pct_return_40, sharpe_ratio stays around ~0.5
        pct_return_15 = max(min(pct_return_15 * 30, 2), -2)
        pct_return_40 = max(min(pct_return_40 * 25, 2), -2)
        sharpe_ratio = max(min(sharpe_ratio * 0.5, 2), -2)
        overtrading_penalty = max(min(overtrading_penalty, 2), -2)

        # final reward
        reward = (
                pct_return_15 +
                pct_return_40 +
                sharpe_ratio +
                overtrading_penalty
        )

        # percentage returns (for 15 history length), sharpe ratio, and overtrading penalty stored for testing purposes
        self.pct_returns_15.append(abs(float(pct_return_15)))
        self.pct_returns_40.append(abs(float(pct_return_40)))
        self.sharpe_ratio.append(abs(float(sharpe_ratio)))
        self.overtrading_penalty.append(abs(float(overtrading_penalty)))

        #print("----------")
        #print(sum(self.pct_returns_15) / len(self.pct_returns_15))
        #print(sum(self.pct_returns_40) / len(self.pct_returns_40))
        #print(sum(self.sharpe_ratio) / len(self.sharpe_ratio))
        #print(sum(self.overtrading_penalty) / len(self.overtrading_penalty))
        #print("----------")

        # update the previous action
        self.prev_action = action

        # store the logs until done
        if not done:
            self.episode_prices.append(float(current_price))
            self.episode_net_worth.append(float(self.net_worth))
            self.episode_actions.append(float(action))
            self.episode_inventory.append(float(self.inventory))
            self.episode_rewards.append(float(reward))

        # get the observations
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)

        # return the observations, reward value, done status, and the information dictionary (empty for now)
        return obs, reward, done, {}

    # function that renders the current environment state by displaying information such as current step, current stock price,
    # current cash balance, current inventory, and current portfolio net worth
    def render(self, mode='human'):

        print(f"Step: {self.current_step}, Price: {self.raw_prices[self.current_step]:.2f}, "
              f"Balance: {self.balance:.2f}, Inventory: {self.inventory}, "
              f"Net worth: {self.net_worth:.2f}")

    # function that sets the seed for reproducibility purposes
    ### seed - seed to set
    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    # function that compute and return useful trading metrics such as returns, sharpe ratio, and number of trades
    def get_metrics(self):
        returns = np.array(self.recent_returns)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) if len(returns) > 1 else 0.0
        return {
            "net_worth": self.net_worth,
            "num_trades": len(self.trades),
            "sharpe_ratio": sharpe_ratio
        }

    # function that logs state metrics such as prices, portfolio net worth, trade actions, stock inventory, and rewards
    # useful to analyse and help in model inprovements
    def get_episode_logs(self):

        print(f"Episode Logs - Lengths: prices={len(self.episode_prices)}, "
              f"actions={len(self.episode_actions)}, inventory={len(self.episode_inventory)}")

        return {
            "prices": self.episode_prices,
            "net_worth": self.episode_net_worth,
            "actions": self.episode_actions,
            "inventory": self.episode_inventory,
            "rewards": self.episode_rewards
        }

    # function that returns the current stock price
    def get_current_price(self):
        return self.raw_prices[self.current_step]

    # function that returns the current state metrics such as current stock price, portfolio net worth, stock inventory, and current step number
    def get_current_state_metrics(self):
        return {
            "price": self.raw_prices[self.current_step],
            "net_worth": self.net_worth,
            "inventory": self.inventory,
            "step": self.current_step
        }

    # function that returns the dimensionality of the continuous action space
    def get_action_space_size(self):
        return self.action_space.shape[0]