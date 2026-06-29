# src/main/agents/random_agent.py

import random


class RandomAgent:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def act(self, observation):
        steering = random.uniform(-0.4, 0.4)
        throttle = random.uniform(0.3, 0.7)

        brake = 0.0
        if random.random() < 0.05:
            brake = random.uniform(0.1, 0.4)
            throttle = 0.0

        return [steering, throttle, brake]

    def reset(self):
        pass