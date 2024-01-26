from abc import ABC, abstractmethod
import torch

class Intervention(ABC):
    def __init__(self, steering_vector, target_pos):
        # assert len(target_pos) == steering_vector.shape[0], \
        #     "Length of target_pos must match first dimension of activations"
        self.steering_vector = steering_vector
        self.target_pos = target_pos

    @abstractmethod
    def apply(self):
        pass


class AddIntervention(Intervention):
    def apply(self, activations):
        # print(activations.shape)
        activations[0, self.target_pos, :] += self.steering_vector
        return activations
    

class AblationIntervention(Intervention):
    def apply(self, activations):
        steering_vector = self.steering_vector.to(activations.dtype)
        orthogonal_space = activations[0, self.target_pos, :] - torch.dot(activations[0, self.target_pos, :], steering_vector) / torch.linalg.norm(steering_vector) * steering_vector
        activations[0, self.target_pos, :] = orthogonal_space
        return activations

