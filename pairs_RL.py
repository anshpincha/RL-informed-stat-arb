import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from collections import deque
import random

class PairsTradingEnvironment:
    """
    Environment for pairs trading strategy optimization using RL
    """
    
    def __init__(self, google_prices, microsoft_prices, hedge_ratio, lookback_period=200):
        self.google_prices = google_prices.values
        self.microsoft_prices = microsoft_prices.values
        self.hedge_ratio = hedge_ratio
        self.lookback_period = lookback_period
        self.total_length = len(google_prices)
        
        # Action space: window_size (30-200), entry_threshold (0.5-3.0), exit_threshold (0.0-0.5)
        self.action_space = {
            'window_sizes': list(range(30, 201, 10)),  # 30, 40, ..., 200
            'entry_thresholds': np.arange(0.5, 3.1, 0.1),  # 0.5, 0.6, ..., 3.0
            'exit_thresholds': np.arange(0.0, 0.6, 0.1)   # 0.0, 0.1, ..., 0.5
        }
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.lookback_period
        self.position = 0  # 0: no position, 1: long spread, -1: short spread
        self.entry_price_m = 0
        self.entry_price_g = 0
        self.total_pnl = 0
        self.trade_count = 0
        self.win_count = 0
        return self.get_state()
    
    def get_state(self):
        """Get current state for RL agent"""
        if self.current_step >= self.total_length - 1:
            return None
            
        # Recent price data for state representation
        recent_m = self.microsoft_prices[max(0, self.current_step-20):self.current_step+1]
        recent_g = self.google_prices[max(0, self.current_step-20):self.current_step+1]
        
        # Calculate recent spread statistics
        recent_spread = recent_m - self.hedge_ratio * recent_g
        
        state = {
            'spread_mean': np.mean(recent_spread),
            'spread_std': np.std(recent_spread),
            'current_spread': recent_spread[-1],
            'position': self.position,
            'step_ratio': self.current_step / self.total_length
        }
        
        return state
    
    def calculate_zscore(self, window_size):
        """Calculate z-score for current position"""
        if self.current_step < window_size:
            return 0
            
        end_idx = self.current_step + 1
        start_idx = end_idx - window_size
        
        m_window = self.microsoft_prices[start_idx:end_idx]
        g_window = self.google_prices[start_idx:end_idx]
        spread_window = m_window - self.hedge_ratio * g_window
        
        mean_spread = np.mean(spread_window)
        std_spread = np.std(spread_window)
        
        if std_spread == 0:
            return 0
            
        current_spread = self.microsoft_prices[self.current_step] - self.hedge_ratio * self.google_prices[self.current_step]
        return (current_spread - mean_spread) / std_spread
    
    def step(self, action):
        """Execute one step in the environment"""
        window_size, entry_threshold, exit_threshold = action
        
        if self.current_step >= self.total_length - 1:
            return None, 0, True, {}
        
        zscore = self.calculate_zscore(window_size)
        reward = 0
        
        current_m = self.microsoft_prices[self.current_step]
        current_g = self.google_prices[self.current_step]
        
        # Trading logic
        if self.position == 0:  # No position
            if zscore > entry_threshold:  # Enter short spread position
                self.position = -1
                self.entry_price_m = current_m
                self.entry_price_g = current_g
            elif zscore < -entry_threshold:  # Enter long spread position
                self.position = 1
                self.entry_price_m = current_m
                self.entry_price_g = current_g
        else:  # In position
            # Exit conditions
            exit_signal = False
            if self.position == 1 and zscore > -exit_threshold:
                exit_signal = True
            elif self.position == -1 and zscore < exit_threshold:
                exit_signal = True
            
            if exit_signal:
                # Calculate PnL
                if self.position == 1:  # Long spread
                    pnl = (current_m - self.entry_price_m) - self.hedge_ratio * (current_g - self.entry_price_g)
                else:  # Short spread
                    pnl = -(current_m - self.entry_price_m) + self.hedge_ratio * (current_g - self.entry_price_g)
                
                self.total_pnl += pnl
                reward = pnl
                self.trade_count += 1
                if pnl > 0:
                    self.win_count += 1
                
                self.position = 0
        
        self.current_step += 1
        next_state = self.get_state()
        done = (self.current_step >= self.total_length - 1)
        
        info = {
            'total_pnl': self.total_pnl,
            'trade_count': self.trade_count,
            'win_rate': self.win_count / max(1, self.trade_count),
            'zscore': zscore
        }
        
        return next_state, reward, done, info


class QLearningAgent:
    """
    Q-Learning agent for pairs trading optimization
    """
    
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Discretize continuous state space
        self.state_bins = {
            'spread_mean': np.linspace(-50, 50, 10),
            'spread_std': np.linspace(0, 50, 10),
            'current_spread': np.linspace(-100, 100, 10),
            'position': [-1, 0, 1],
            'step_ratio': np.linspace(0, 1, 10)
        }
        
        # Initialize Q-table
        self.q_table = {}
        
        # Create action combinations
        self.actions = []
        for window in action_space['window_sizes']:
            for entry_thresh in action_space['entry_thresholds']:
                for exit_thresh in action_space['exit_thresholds']:
                    if exit_thresh < entry_thresh:  # Exit threshold should be less than entry
                        self.actions.append((window, entry_thresh, exit_thresh))
    
    def discretize_state(self, state):
        """Convert continuous state to discrete state for Q-table"""
        if state is None:
            return None
            
        discrete_state = []
        for key, value in state.items():
            if key in self.state_bins:
                bins = self.state_bins[key]
                if key == 'position':
                    discrete_state.append(value)
                else:
                    bin_idx = np.digitize(value, bins) - 1
                    bin_idx = max(0, min(len(bins) - 1, bin_idx))
                    discrete_state.append(bin_idx)
        
        return tuple(discrete_state)
    
    def get_action(self, state):
        """Select action using epsilon-greedy policy"""
        discrete_state = self.discretize_state(state)
        
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        
        if random.random() < self.epsilon:
            action_idx = random.randint(0, len(self.actions) - 1)
        else:
            action_idx = np.argmax(self.q_table[discrete_state])
        
        return self.actions[action_idx], action_idx
    
    def update_q_value(self, state, action_idx, reward, next_state):
        """Update Q-value using Q-learning update rule"""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        
        if discrete_next_state is not None and discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(len(self.actions))
        
        current_q = self.q_table[discrete_state][action_idx]
        
        if discrete_next_state is not None:
            max_next_q = np.max(self.q_table[discrete_next_state])
        else:
            max_next_q = 0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q
    
    def decay_epsilon(self, decay_rate=0.995):
        """Decay exploration rate"""
        self.epsilon = max(0.01, self.epsilon * decay_rate)


def train_rl_agent(google_prices, microsoft_prices, hedge_ratio, episodes=100):
    """Train the RL agent"""
    env = PairsTradingEnvironment(google_prices, microsoft_prices, hedge_ratio)
    agent = QLearningAgent(env.action_space)
    
    episode_rewards = []
    episode_trades = []
    episode_win_rates = []
    
    print("Training RL Agent...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action, action_idx = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.update_q_value(state, action_idx, reward, next_state)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_trades.append(info['trade_count'])
        episode_win_rates.append(info['win_rate'])
        
        agent.decay_epsilon()
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_trades = np.mean(episode_trades[-20:])
            avg_win_rate = np.mean(episode_win_rates[-20:])
            print(f"Episode {episode + 1}: Avg Reward: {avg_reward:.2f}, "
                  f"Avg Trades: {avg_trades:.1f}, Avg Win Rate: {avg_win_rate:.3f}")
    
    return agent, episode_rewards


def test_strategy(agent, google_prices, microsoft_prices, hedge_ratio):
    """Test the trained strategy"""
    env = PairsTradingEnvironment(google_prices, microsoft_prices, hedge_ratio)
    agent.epsilon = 0  # No exploration during testing
    
    state = env.reset()
    actions_taken = []
    zscores = []
    positions = []
    pnls = []
    
    while True:
        action, _ = agent.get_action(state)
        actions_taken.append(action)
        
        next_state, reward, done, info = env.step(action)
        
        zscores.append(info.get('zscore', 0))
        positions.append(env.position)
        pnls.append(info['total_pnl'])
        
        state = next_state
        
        if done:
            break
    
    return {
        'actions': actions_taken,
        'zscores': zscores,
        'positions': positions,
        'pnls': pnls,
        'final_pnl': info['total_pnl'],
        'total_trades': info['trade_count'],
        'win_rate': info['win_rate']
    }


# Load and prepare data (using your existing code structure)
def main():
    # Load data
    tickers = ['MSFT', 'GOOG']
    start = '2020-03-01'
    end = '2025-07-24'
    data = yf.download(tickers, start=start, end=end, interval='1d').dropna()
    
    google = data['Close']['GOOG']
    microsoft = data['Close']['MSFT']
    
    # Calculate hedge ratio (using your existing method)
    google_with_const = sm.add_constant(google)
    model = sm.OLS(microsoft, google_with_const)
    results = model.fit()
    hedge_ratio = results.params['GOOG']
    
    print(f"Calculated hedge ratio: {hedge_ratio:.4f}")
    
    # Split data into train and test
    split_point = int(0.7 * len(google))
    
    google_train = google.iloc[:split_point]
    microsoft_train = microsoft.iloc[:split_point]
    google_test = google.iloc[split_point:]
    microsoft_test = microsoft.iloc[split_point:]
    
    # Train RL agent
    agent, training_rewards = train_rl_agent(
        google_train, microsoft_train, hedge_ratio, episodes=200
    )
    
    # Test strategy
    test_results = test_strategy(agent, google_test, microsoft_test, hedge_ratio)
    
    print("\n=== RL Strategy Results ===")
    print(f"Final PnL: ${test_results['final_pnl']:.2f}")
    print(f"Total Trades: {test_results['total_trades']}")
    print(f"Win Rate: {test_results['win_rate']:.3f}")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training progress
    ax1.plot(training_rewards)
    ax1.set_title('Training Rewards Over Episodes')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Cumulative PnL
    ax2.plot(test_results['pnls'])
    ax2.set_title('Cumulative PnL - RL Strategy')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Cumulative PnL ($)')
    ax2.grid(True)
    
    # Z-scores and positions
    ax3.plot(test_results['zscores'], alpha=0.7, label='Z-Score')
    ax3.fill_between(range(len(test_results['positions'])), 
                     0, test_results['positions'], 
                     alpha=0.3, label='Position')
    ax3.set_title('Z-Scores and Positions')
    ax3.set_xlabel('Time Steps')
    ax3.legend()
    ax3.grid(True)
    
    # Action parameters over time
    windows = [action[0] for action in test_results['actions']]
    entry_thresholds = [action[1] for action in test_results['actions']]
    
    ax4.plot(windows, label='Window Size', alpha=0.7)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(entry_thresholds, 'r-', label='Entry Threshold', alpha=0.7)
    
    ax4.set_title('Dynamic Parameters')
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Window Size', color='b')
    ax4_twin.set_ylabel('Entry Threshold', color='r')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Compare with static strategy
    print("\n=== Comparison with Static Strategy ===")
    
    # Run static strategy for comparison
    static_window = 90
    static_entry = 1.0
    static_exit = 0.0
    
    env_static = PairsTradingEnvironment(google_test, microsoft_test, hedge_ratio)
    state = env_static.reset()
    static_pnl = []
    
    while True:
        next_state, reward, done, info = env_static.step((static_window, static_entry, static_exit))
        static_pnl.append(info['total_pnl'])
        
        if done:
            static_final_pnl = info['total_pnl']
            static_trades = info['trade_count']
            static_win_rate = info['win_rate']
            break
    
    print(f"Static Strategy - Final PnL: ${static_final_pnl:.2f}")
    print(f"Static Strategy - Total Trades: {static_trades}")
    print(f"Static Strategy - Win Rate: {static_win_rate:.3f}")
    
    print(f"\nRL Improvement: ${test_results['final_pnl'] - static_final_pnl:.2f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(test_results['pnls'], label='RL Strategy', linewidth=2)
    plt.plot(static_pnl, label='Static Strategy', linewidth=2)
    plt.title('Strategy Comparison: RL vs Static')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative PnL ($)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()