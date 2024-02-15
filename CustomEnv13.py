import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class CustomEnv13(gym.Env):
    """Custom Environment that simulates a system for waveform generation, following the OpenAI Gym interface."""
    
    def __init__(self, kernel_size=20, execution_fraction=5, buffer_size=120, dt=1/10, frequency=0.5, amplitude=5):
        super(CustomEnv13, self).__init__()
        # Validate input parameters
        assert kernel_size % execution_fraction == 0, "kernel_size must be an integer multiple of execution_fraction"
        assert buffer_size % kernel_size == 0, "buffer_size must be an integer multiple of kernel_size"
        
        # Environment parameters
        self.kernel_size = kernel_size
        self.buffer_size = buffer_size
        self.stride = kernel_size // execution_fraction
        self.dt = dt
        self.frequency = frequency
        self.amplitude = amplitude
        self.action_number = (buffer_size - kernel_size) // self.stride + 1
        
        # State variables
        self.t = 0
        self.buffer = np.zeros(buffer_size)
        self.full_observation = np.zeros(buffer_size)
        self.last_full_observation = self.full_observation.copy()
        
        # Kernel for waveform generation
        time = np.arange(start=-((kernel_size + 1) / 2 - 1) * dt, stop=(kernel_size + 1) / 2 * dt, step=dt)
        self.kernel = np.sinc(time / kernel_size) * np.sinc(time)
        
        # Reference output for reward computation
        self.ref_out = self.compute_reference()
        
        # Action and observation spaces
        self.action_space = spaces.Box(low=-amplitude, high=amplitude, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.5 * amplitude, high=1.5 * amplitude, shape=(self.buffer_size,), dtype=np.float32)
        
        # Reward tracking
        self.eps = 1e-8
        self.old_reward = -10000
        self.old_old_reward = -10000

    def reset(self):
        """Resets the environment to its initial state."""
        self.buffer = np.zeros(self.buffer_size)
        self.full_observation = np.zeros(self.buffer_size)
        self.t = 0
        self.old_reward = -10000
        self.old_old_reward = -10000
        self.last_full_observation = self.full_observation.copy()
        return np.copy(self.full_observation)

    def compute_reference(self):
        """Generates the reference output waveform."""
        action_time = np.arange(start=0, stop=self.buffer_size * self.dt, step=self.stride * self.dt)
        actions = np.sin(2 * np.pi * self.frequency * action_time) * self.amplitude
        self.reset()
        for i in range(1, self.action_number):  # Skip initial action
            self._take_action(actions[i])
        return np.copy(self.last_full_observation)

    def _take_action(self, action):
        """Applies the given action to the environment."""
        self.buffer[self.t * self.stride:self.t * self.stride + self.kernel_size] += action * self.kernel
        self.full_observation[self.t * self.stride:(self.t + 1) * self.stride] = self.buffer[self.t * self.stride:(self.t + 1) * self.stride]
        self.t += 1

    def reward_computation(self, buffer):
        """Computes the reward based on the current buffer and reference output."""
        rew = np.correlate(buffer / (max(abs(buffer)) + self.eps), self.ref_out / (max(abs(self.ref_out)) + self.eps))
        return rew

    def step(self, action):
        """Executes one time step within the environment."""
        self._take_action(action)
        reward = self.reward_computation(self.full_observation)
        done = False

        if self.old_reward > reward and reward < self.old_old_reward:
            done = True
        elif self.t >= self.action_number:
            done = True

        self.old_old_reward = self.old_reward
        self.old_reward = reward
        return np.copy(self.full_observation), reward, done, {}

    def render(self):
        """Renders the environment's current state."""
        plt.plot(self.full_observation)
        plt.show()
