# Stock Trading ML & RL Project

A deep learning and reinforcement learning prototype for predicting stock price movements and simulating trading decisions. This project explores combining supervised models with reinforcement learning to create an intelligent trading agent.

## Overview

This project started as a simple neural network prediction task and evolved into a multi-stage pipeline that:
- Generates simulated stock price paths using Geometric Brownian Motion.
- Trains hybrid ResNet-LSTM models to predict short- and long-term stock price directions.
- Consolidates predictions via a calibrated Random Forest for stronger probability estimates.
- Trains a PPO-based reinforcement learning agent in a custom trading environment to maximize returns.
- Evaluates performance with metrics like Sharpe and Sortino ratios, comparing the agent to buy-and-hold strategies.

## Project Stages
1. Training Data Collection – Historical stock prices expanded with 350 simulated paths per stock to improve model generalization.
2. Neural Network Training – Supervised ResNet-LSTM models trained to predict binary stock movement labels at 10, 30, 60, 150 days.
3. Model Consolidation – Random Forest models combine neural network predictions and engineered features for more accurate probability estimates.
4. Pivot to Reinforcement Learning – Prepared data and features for RL training, moving from manual logic to automated learning.
5. RL Model Training – PPO agent trained in a custom OpenAI Gym trading environment using consolidated model predictions.
6. Testing & Evaluation – Tested on unseen stocks, achieving significant improvement over buy-and-hold strategies.

## Technologies & Libraries
- Python
- PyTorch (ResNet-LSTM models)
- Stable-Baselines3 (PPO RL agent)
- OpenAI Gym (custom trading environment)
- scikit-learn (Random Forest & model calibration)
- pandas & numpy (data manipulation)
- matplotlib (plots & visualizations)

## Results Highlights
- Average portfolio return: ~15.98x vs buy-and-hold ~5.49x
- Average Sharpe ratio: 1.31 vs stock price 0.62
- Average Sortino ratio: 1.43 vs stock price 0.85

## Notes
- This project is intended as a portfolio showcase and is not financial advice.

## Acknowledgements
- Thanks to Professor Fred Sala and Michael X. Bell for guidance and inspiration throughout this project.
