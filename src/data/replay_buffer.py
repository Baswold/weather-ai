"""
Experience replay buffers for RL training.

Implements both uniform and prioritized replay buffers.
Prioritized replay helps learn from rare events like hurricanes.
"""

import random
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import numpy as np
import pickle


class Transition:
    """
    A single transition in the RL training process.

    For weather prediction, a transition contains:
    - state: Historical window + current forecast
    - action: The prediction made by the model
    - reward: How accurate the prediction was
    - next_state: Updated history (for potential multi-step learning)
    """

    __slots__ = ["state", "action", "reward", "next_state", "info"]

    def __init__(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: Optional[np.ndarray] = None,
        info: Optional[Dict] = None,
    ):
        self.state = state  # Combined [history, current_forecast]
        self.action = action  # Model's prediction
        self.reward = reward  # Negative MSE or similar
        self.next_state = next_state
        self.info = info or {}


class UniformReplayBuffer:
    """
    Standard uniform experience replay buffer.

    Samples transitions uniformly at random from past experience.
    Simple and effective for most cases.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self._position = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: Optional[np.ndarray] = None,
        info: Optional[Dict] = None,
    ):
        """Add a transition to the buffer."""
        transition = Transition(state, action, reward, next_state, info)
        self.buffer.append(transition)

    def push_batch(self, transitions: List[Transition]):
        """Add multiple transitions at once."""
        self.buffer.extend(transitions)

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a random batch of transitions."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self.buffer) >= batch_size

    def save(self, path: str):
        """Save buffer to disk."""
        with open(path, "wb") as f:
            pickle.dump(list(self.buffer), f)

    def load(self, path: str):
        """Load buffer from disk."""
        with open(path, "rb") as f:
            transitions = pickle.load(f)
            self.buffer.clear()
            self.buffer.extend(transitions)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.

    Samples transitions with probability proportional to their TD-error.
    This emphasizes learning from:
    1. Rare events (hurricanes, heat waves)
    2. Mistakes the model is still making

    Based on: "Prioritized Experience Replay" (Schaul et al., 2016)
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        """
        Initialize the prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (annealed to 1)
            beta_increment: How much to increase beta each sample
            epsilon: Small constant to ensure non-zero priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.buffer: List[Transition] = []
        self.priorities: np.ndarray = np.zeros(capacity)
        self._position = 0
        self._size = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: Optional[np.ndarray] = None,
        info: Optional[Dict] = None,
    ):
        """Add a transition with maximum priority (new experiences are important)."""
        transition = Transition(state, action, reward, next_state, info)

        max_priority = self.priorities.max() if self._size > 0 else 1.0

        if self._size < self.capacity:
            self.buffer.append(transition)
            self._size += 1
        else:
            self.buffer[self._position] = transition

        self.priorities[self._position] = max_priority
        self._position = (self._position + 1) % self.capacity

    def push_batch(self, transitions: List[Transition]):
        """Add multiple transitions."""
        for t in transitions:
            self.push(t.state, t.action, t.reward, t.next_state, t.info)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        Sample a batch with prioritization.

        Returns:
            (transitions, indices, importance_weights)
        """
        if self._size == 0:
            return [], np.array([]), np.array([])

        # Calculate sampling probabilities
        priorities = self.priorities[:self._size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(self._size, size=min(batch_size, self._size), p=probs, replace=False)

        # Calculate importance sampling weights
        weights = (self._size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Anneal beta towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        transitions = [self.buffer[i] for i in indices]
        return transitions, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD-errors.

        Higher TD-error = more to learn from = higher priority
        """
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + self.epsilon

    def __len__(self) -> int:
        return self._size

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= batch_size


class ExtremeEventBuffer(UniformReplayBuffer):
    """
    Specialized buffer for extreme weather events.

    Maintains a separate buffer for rare events like:
    - Hurricanes/typhoons
    - Extreme temperatures
    - Heavy precipitation

    These are sampled with higher probability to ensure the model
    learns to predict important events.
    """

    def __init__(self, capacity: int = 100000, extreme_ratio: float = 0.3):
        """
        Initialize the extreme event buffer.

        Args:
            capacity: Total buffer capacity
            extreme_ratio: Fraction of samples that should be extreme events
        """
        super().__init__(capacity)
        self.extreme_ratio = extreme_ratio
        self.extreme_indices: deque[int] = deque()

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: Optional[np.ndarray] = None,
        info: Optional[Dict] = None,
    ):
        """Add transition and track if it's an extreme event."""
        index = len(self.buffer)
        super().push(state, action, reward, next_state, info)

        # Check if this is an extreme event
        if info and info.get("is_extreme", False):
            self.extreme_indices.append(index)

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample with extreme event oversampling."""
        n_extreme = int(batch_size * self.extreme_ratio)
        n_normal = batch_size - n_extreme

        # Sample extreme events
        extremes = []
        if self.extreme_indices and n_extreme > 0:
            extreme_idx_list = list(self.extreme_indices)
            n_extreme = min(n_extreme, len(extreme_idx_list))
            extreme_indices = random.sample(extreme_idx_list, n_extreme)
            extremes = [self.buffer[i] for i in extreme_indices if i < len(self.buffer)]

        # Sample normally for the rest
        all_indices = set(range(len(self.buffer)))
        extreme_set = set(self.extreme_indices)
        normal_indices = list(all_indices - extreme_set)
        n_normal = min(n_normal, len(normal_indices))

        normals = []
        if n_normal > 0:
            normal_indices_sampled = random.sample(normal_indices, n_normal)
            normals = [self.buffer[i] for i in normal_indices_sampled]

        return extremes + normals


class TemporalReplayBuffer(UniformReplayBuffer):
    """
    Replay buffer that respects temporal ordering.

    Instead of sampling randomly, this samples sequences of consecutive days.
    This can help the model learn temporal dependencies.
    """

    def __init__(self, capacity: int = 100000, sequence_length: int = 5):
        """
        Initialize the temporal replay buffer.

        Args:
            capacity: Maximum number of transitions
            sequence_length: Length of temporal sequences to sample
        """
        super().__init__(capacity)
        self.sequence_length = sequence_length
        self.sequences: List[List[int]] = []  # Track sequences by location/date

    def add_sequence(self, transitions: List[Transition]):
        """Add a temporal sequence of transitions."""
        start_idx = len(self.buffer)
        for t in transitions:
            self.buffer.append(t)
        # Store sequence as (start_idx, end_idx)
        self.sequences.append((start_idx, start_idx + len(transitions) - 1))

    def sample_sequence(self) -> Optional[List[Transition]]:
        """Sample a random temporal sequence."""
        if not self.sequences:
            return None

        start, end = random.choice(self.sequences)
        sequence = []
        for i in range(start, end + 1):
            if i < len(self.buffer):
                sequence.append(self.buffer[i])

        return sequence if sequence else None

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample random sequences up to batch_size."""
        sequences = []
        while len(sum([s or [] for s in sequences], [])) < batch_size:
            seq = self.sample_sequence()
            if seq:
                sequences.append(seq)

        # Flatten
        result = []
        for seq in sequences:
            result.extend(seq)
        return result[:batch_size]
