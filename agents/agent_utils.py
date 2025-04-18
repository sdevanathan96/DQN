import torch


class ParameterNoise:
    """
    Implements parameter space noise for exploration.
    """

    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adaptation_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        """
        Adapt noise based on the distance between perturbed and non-perturbed actions.
        """
        if distance > self.desired_action_stddev:
            self.current_stddev /= self.adaptation_coefficient
        else:
            self.current_stddev *= self.adaptation_coefficient

    def get_noisy_params(self, params):
        """
        Add noise to parameters
        """
        noisy_params = {}
        for name, param in params.items():
            noise = torch.normal(mean=0, std=self.current_stddev, size=param.size())
            noisy_params[name] = param + noise
        return noisy_params
