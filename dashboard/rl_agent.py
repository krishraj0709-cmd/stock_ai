# models/intraday/rl_agent.py

import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import os


class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for intraday trading.
    Observations : 20 normalised market features
    Actions      : 0=SELL  1=HOLD  2=BUY
    Reward       : portfolio return per step
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.0003   # 0.03% per trade
    ):
        super().__init__()
        self.df               = df.reset_index(drop=True)
        self.feature_cols     = feature_cols
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost
        self.n_feats          = len(feature_cols) + 2  # +position +cash_ratio

        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.n_feats,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx       = 0
        self.position  = 0        # number of shares held
        self.cash      = self.initial_capital
        self.prev_val  = self.initial_capital
        return self._obs(), {}

    def _obs(self):
        row      = self.df.iloc[self.idx]
        features = row[self.feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        price    = float(row["close"])
        extra    = np.array([
            self.position / 100.0,
            self.cash / self.initial_capital
        ], dtype=np.float32)
        return np.clip(np.concatenate([features, extra]), -10, 10)

    def step(self, action):
        price     = float(self.df.iloc[self.idx]["close"])
        cost      = price * self.transaction_cost

        if action == 2:   # BUY
            affordable = int((self.cash * 0.10) / (price + cost))
            if affordable > 0:
                self.cash    -= affordable * (price + cost)
                self.position += affordable

        elif action == 0:  # SELL
            if self.position > 0:
                self.cash    += self.position * (price - cost)
                self.position = 0

        self.idx  += 1
        done       = self.idx >= len(self.df) - 1
        curr_val   = self.cash + self.position * price
        reward     = (curr_val - self.prev_val) / self.initial_capital
        self.prev_val = curr_val

        return self._obs(), reward, done, False, {
            "portfolio_value": curr_val
        }

    def render(self):
        pass


def train_rl_agent(
    df: pd.DataFrame,
    feature_cols: list,
    timesteps: int = 200_000,
    save_path: str = "models/intraday/ppo_trader"
):
    def make_env():
        return TradingEnv(df, feature_cols)

    env  = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy", env,
        learning_rate     = 3e-4,
        n_steps           = 2048,
        batch_size        = 64,
        n_epochs          = 10,
        gamma             = 0.99,
        gae_lambda        = 0.95,
        clip_range        = 0.2,
        ent_coef          = 0.01,
        verbose           = 1
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = save_path,
        eval_freq            = 10_000,
        deterministic        = True,
        render               = False,
        verbose              = 0
    )

    model.learn(total_timesteps=timesteps, callback=eval_cb)
    model.save(save_path + "_final")
    print(f"  RL agent saved to {save_path}_final")
    return model


if __name__ == "__main__":
    print("RL agent definitions loaded OK.")