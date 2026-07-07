import random
import secrets


class RandomAgent:
    name = "Random Agent"
    agent_type = "random"
    version = "1.0"

    def __init__(self, seed=None):
        self.seed = seed if seed is not None else secrets.randbits(32)
        self._random = random.Random(self.seed)

    @property
    def config(self):
        return {
            "steering_range": [-0.4, 0.4],
            "throttle_range": [0.3, 0.7],
        }

    def act(self, _observation, _telemetry=None):
        steering = self._random.uniform(-0.4, 0.4)
        throttle = self._random.uniform(0.3, 0.7)

        # TorcsEnv is configured for steering and throttle only.
        return [steering, throttle]

    def reset(self):
        self._random.seed(self.seed)
