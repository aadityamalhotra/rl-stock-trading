######################################################## FILE HEADER ########################################################
# This file is the training script for a Reinforcement - Learning based trading model (specifically tech stocks)
# This file tests the trained model on unseen tech stocks and performance is compared to a buy and hold strategy

# Below are the imports required for this file
import os
import pickle
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.trading_env import TradingEnv
from utils.logging_tools import Logger
import matplotlib.pyplot as plt

# path of the model to test
MODEL_PATH = "models/ppo_agent/_seed_893_NEW_pct15-40-60"

# path of the testing data
DATA_PATH = "TECH_TESTING.pkl"

# minimum window size needed by the model to make trading decisions
WINDOW_SIZE = 60

# starting cash balance
INITIAL_BALANCE = 100_000

# folder to store the output plots
PLOT_DIR = "plots/single_stock_plots/NEWgyg"

# create a folder with path PLOT_DIR if it does not currently exist
os.makedirs(PLOT_DIR, exist_ok=True)

# Logging the testing process
logger = Logger(experiment_name="PPO_Evaluation")

# Logging the testing process
logger.log("Loading test stock data...")

# loading the test data into 'test_stock_data' in format = dict: ticker -> stock data
with open(DATA_PATH, 'rb') as f:
    test_stock_data = pickle.load(f)

# get a sample stock data from the total testing data to initialize the environment
dummy_data = next(iter(test_stock_data.values()))

# initializing the dummy trading environment
dummy_env = DummyVecEnv([lambda: TradingEnv(
    raw_prices=np.array(dummy_data['raw_prices']),
    norm_log_returns=np.array(dummy_data['norm_log_returns']),
    model_preds=np.array(dummy_data['model_preds']),
    window_size=WINDOW_SIZE,
    initial_balance=INITIAL_BALANCE
)])

# loading in the vecnorlamize stats that were saved during training to ensure consistent normalization
vec_norm = VecNormalize.load(os.path.join(MODEL_PATH, "vecnormalize.pkl"), dummy_env)

# set the vecnormalize to evaluation mode
vec_norm.training = False

# disable reward normalization to get actual rewards
vec_norm.norm_reward = False

# load the trained agent and the evaluation set vecnormalize environment
model = PPO.load(os.path.join(MODEL_PATH, "ppo_agent"), env=vec_norm)

# Logging the testing process
logger.log("Loaded PPO agent and normalization stats.")

# Logging the testing process
logger.log("Starting evaluation...")

# looping through the test stocks
for stock_name, stock_data in test_stock_data.items():

    # create a new TradingEnv instance for the current test stock
    trading_env = TradingEnv(
        raw_prices=np.array(stock_data['raw_prices']),
        norm_log_returns=np.array(stock_data['norm_log_returns']),
        model_preds=np.array(stock_data['model_preds']),
        window_size=WINDOW_SIZE,
        initial_balance=INITIAL_BALANCE
    )

    # function that returns pre-initialized trading environment for testing
    def make_env():
        return trading_env

    # function that flattens numpy types to normal python floats for easier use
    def flatten_log(log):
        return [float(x) if isinstance(x, (np.ndarray, np.generic)) else x for x in log]

    # creates a vectorized environment wrapper using the make_env() function
    vec_env = DummyVecEnv([make_env])

    # apply vecnormalize with saved stats
    env = VecNormalize(vec_env, training=False)

    # copy the observation normalization stats to the environment
    env.obs_rms = vec_norm.obs_rms

    # copy the return normalization stats to the environment
    env.ret_rms = vec_norm.ret_rms

    # disable reward normalization during testing
    env.norm_reward = False

    # reset to get the initial environment configuration
    obs = env.reset()

    # initialize the done flag to false at the start
    done = False

    # initialize total reward to 0
    total_reward = 0.0

    # initialize current step to 0
    steps = 0

    # initialize lists for logging purposes
    logged_prices = []
    logged_net_worth = []
    logged_actions = []
    logged_inventory = []
    logged_rewards = []

    # disable gradient tracking as it is not needed during evaluation
    with torch.no_grad():

        # run evaluation loop
        while not done:

            # get the models next action
            action, _ = model.predict(obs, deterministic=True)

            # proceed with the given action and get the observations
            obs, reward, done, _ = env.step(action)

            # accumulate the total reward
            total_reward += reward[0]

            # increment the step
            steps += 1

            # logging the data at the current episode / timestep
            logged_prices.append(trading_env.get_current_price())
            logged_net_worth.append(trading_env.net_worth)
            logged_actions.append(int(action[0]) if isinstance(action, np.ndarray) else int(action))
            logged_inventory.append(trading_env.inventory)
            logged_rewards.append(reward[0])

    # getting the metrics after testing
    metrics = trading_env.get_metrics()

    # Logging the testing process
    logger.log(
        f"[{stock_name}] Total Reward: {total_reward:.2f} | Steps: {steps} | "
        f"Net Worth: {metrics['net_worth']:.2f} | Trades: {metrics['num_trades']} | "
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}"
    )

    # getting the buy and sell points for plotting / visualization purposes
    buy_points = [i for i, a in enumerate(logged_actions) if a > 0.05]
    sell_points = [i for i, a in enumerate(logged_actions) if a < -0.05]

    # defining aspects of the plot
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # trimming the logged data to leave out the last element (as the last element has the original setup data)
    logged_net_worth = flatten_log(logged_net_worth)[:-1]
    logged_prices = flatten_log(logged_prices)[:-1]
    logged_rewards = flatten_log(logged_rewards)[:-1]
    logged_inventory = flatten_log(logged_inventory)[:-1]
    logged_actions = flatten_log(logged_actions)[:-1]

    # safety check to avoid out of range indexing
    safe_buy_points = [i for i in buy_points if i < len(logged_prices)]
    safe_sell_points = [i for i in sell_points if i < len(logged_prices)]

    ##### BELOW IS THE ENTIRE PLOTTING LOGIC #####
    axs[0].plot(logged_prices, label="Price", color="black")
    if safe_buy_points:
        axs[0].scatter(safe_buy_points, [logged_prices[i] for i in safe_buy_points],
                       color='green', marker='^', label="Buy", alpha=0.6)
    if safe_sell_points:
        axs[0].scatter(safe_sell_points, [logged_prices[i] for i in safe_sell_points],
                       color='red', marker='v', label="Sell", alpha=0.6)
    axs[0].set_ylabel("Price")
    axs[0].legend()

    axs[1].plot(logged_net_worth, label="Net Worth", color="blue")
    axs[1].set_ylabel("Net Worth")
    axs[1].legend()

    axs[2].plot(logged_inventory, label="Inventory", color="purple")
    axs[2].set_ylabel("Inventory")
    axs[2].legend()

    axs[3].plot(logged_rewards, label="Reward", color="orange")
    axs[3].set_ylabel("Reward")
    axs[3].set_xlabel("Step")
    axs[3].legend()

    fig.suptitle(f"Evaluation: {stock_name}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(os.path.join(PLOT_DIR, f"{stock_name}_LATEST_15-40-60_lesstrading.png"))
    plt.close()

    # Logging the testing process
    logger.log(f"Saved plot: {stock_name}_LATEST_40.png")

    # get the flattened raw price list
    price_series = np.array(stock_data['raw_prices']).flatten()

    # get the initial stock price
    initial_price = float(price_series[WINDOW_SIZE])

    # get the final stock price
    final_price = float(price_series[-1])

    # get the initial portfolio value (which is the initial cash value)
    initial_portfolio = INITIAL_BALANCE

    # get the final portfolio value (after model trading)
    final_portfolio = float(logged_net_worth[-1])

    # get the buy-and-hold approach multiple
    price_change_ratio = final_price / initial_price

    # get the portfolio multiple
    portfolio_change_ratio = final_portfolio / initial_portfolio

    # print statements summarizing the trading results
    print(f"\n=== {stock_name} Evaluation Summary ===")
    print(f"Initial Price       : {initial_price:.2f}")
    print(f"Final Price         : {final_price:.2f}")
    print(f"Initial Portfolio   : {initial_portfolio:.2f}")
    print(f"Final Portfolio     : {final_portfolio:.2f}")
    print(f"Price Change Ratio  : {price_change_ratio:.4f}")
    print(f"Portfolio Change Ratio: {portfolio_change_ratio:.4f}\n")

    # get the net worth list
    net_worth_series = np.array(logged_net_worth).flatten()

    # get the raw price reutrns
    price_returns = np.diff(price_series) / price_series[:-1]

    # get the portfolio value returns
    portfolio_returns = np.diff(net_worth_series) / net_worth_series[:-1]

    # annualization factor used for calculating sharpe and sortino ratios
    annualization_factor = np.sqrt(252)

    # calculating the sharpe ratio for the stock buy-and-hold ratio
    stock_sharpe = np.mean(price_returns) / (np.std(price_returns) + 1e-8) * annualization_factor

    # calculating the sharpe ratio for the portfolio trading model
    portfolio_sharpe = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * annualization_factor

    # function that calculates the sortino ratio
    ### returns - returns list for either the stock price or portfolio
    ### annualization_factor - annualization factor used for calculating sharpe and sortino ratios
    def sortino_ratio(returns, annualization_factor=np.sqrt(252)):
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        return (np.mean(returns) / (downside_std + 1e-8)) * annualization_factor

    # calculating the sortino ratio for buy-and-hold strategy
    stock_sortino = sortino_ratio(price_returns)

    # calculating the sortino ratio for the trading model portfolio
    portfolio_sortino = sortino_ratio(portfolio_returns)

    # displaying the final statistics
    print(f"Sharpe Ratio (Buy & Hold Stock) : {stock_sharpe:.4f}")
    print(f"Sharpe Ratio (RL Portfolio)     : {portfolio_sharpe:.4f}")
    print(f"Sortino Ratio (Buy & Hold Stock): {stock_sortino:.4f}")
    print(f"Sortino Ratio (RL Portfolio)    : {portfolio_sortino:.4f}")