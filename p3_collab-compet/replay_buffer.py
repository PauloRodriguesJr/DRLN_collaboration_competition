from collections import deque
import random
import torch
import numpy as np
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, n_agents):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "states", "actions", "rewards", "next_states", "dones", "priority"])
        self.n_agents = n_agents

    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # TODO: Weight the probability of being selected by its priority

        # Start retrieving the priority of the experience

        experiences = random.sample(self.memory, k=self.batch_size)

        states = [torch.from_numpy(np.vstack([e.states[index] for e in experiences if e is not None])).float(
        ).to(device) for index in range(self.n_agents)]
        actions = [torch.from_numpy(np.vstack([e.actions[index] for e in experiences if e is not None])).float(
        ).to(device) for index in range(self.n_agents)]
        rewards = torch.from_numpy(np.vstack(
            [e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = [torch.from_numpy(np.vstack([e.next_states[index] for e in experiences if e is not None])).float(
        ).to(device) for index in range(self.n_agents)]
        dones = torch.from_numpy(np.vstack(
            [e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
