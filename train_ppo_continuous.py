######################################################## FILE HEADER ########################################################
# This file is the training script for a Reinforcement - Learning based trading model (specifically tech stocks)
# This file trains a model with extensive training data and saving the model with unique names based on its
# specific training style

# Below are the imports required for this file
import os
import pickle
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.trading_env import TradingEnv
from utils.logging_tools import Logger
import random

# Path of the training data
DATA_PATH = "TECH_TRAINING.pkl"  # adjust as needed

# Folder where this model will be stored after training
MODEL_SAVE_PATH = "models/ppo_agent"

# Number of timesteps to train on
TOTAL_TIMESTEPS =  10000000

# Minimum window size required to collect initial data
WINDOW_SIZE = 60

# Initial cash balance of 100,000$
INITIAL_BALANCE = 100_000

# Seed for reproducibility
SEED_LIST = [893]

# Logging the training process
logger = Logger(experiment_name="PPO_Training")

# Logging the training process
logger.log("Loading preprocessed data...")

# Loading the training data (data for multiple stocks)
with open(DATA_PATH, 'rb') as f:
    all_stock_data = pickle.load(f)

# Function that creates a multi-stock environment wrapper
### data - data for a particular stock
def make_env_from_stock(stock_data):
    return TradingEnv(
        raw_prices=np.array(stock_data['raw_prices']),
        norm_log_returns=np.array(stock_data['norm_log_returns']),
        model_preds=np.array(stock_data['model_preds']),
        window_size=WINDOW_SIZE,
        initial_balance=INITIAL_BALANCE
    )

# Function that returns a lambda function to create an environment using the make_env_from_stock function
### data - data for a particular stock
def make_env_lambda(data):
    return lambda: make_env_from_stock(data)

# Logging the training process
logger.log("Creating vectorized environments for multiple stocks...")

# Creating a list of environments for each stock from the total data
env_fns = [make_env_lambda(stock_data) for stock_data in all_stock_data.values()]

# Wrap individual trading environments into a single vectorized environment
venv = DummyVecEnv(env_fns)

# Normalize the vectorized environments
venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Training loop for multiple seeds
for seed in SEED_LIST:

    # Logging the training process
    logger.log(f"\n--- Training with seed {seed} ---")

    # Setting the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Defining the model characteristics
    model = PPO(
        policy="MlpPolicy",
        env=venv,
        verbose=1,
        tensorboard_log="logs/ppo_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        seed=seed,
        normalize_advantage=True,
        vf_coef=0.8
    )

    # Starts the training process
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Create path to save model and save the model
    model_dir = os.path.join(MODEL_SAVE_PATH, f"_seed_{seed}_NEW_pct15-40-60_lesstrading")
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "ppo_agent"))

    # Save the normalization state
    venv.save(os.path.join(model_dir, "vecnormalize.pkl"))

    # Logging the end of the training process
    logger.log(f"Model + normalization saved at {model_dir}")
