import numpy as np

# Self contained optimization algorithm
class AdaDelta:

    def __init__(self, shape, rho: float = 0.9, eps: float = 0.1):
        self.rho  = rho
        self.eps  = eps
        self.Eg2  = np.zeros(shape, dtype=np.float32)
        self.EdX2 = np.zeros(shape, dtype=np.float32)

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.Eg2  = self.rho * self.Eg2  + (1.0 - self.rho) * grad ** 2
        delta     = -(np.sqrt(self.EdX2 + self.eps)
                      / np.sqrt(self.Eg2  + self.eps)) * grad
        self.EdX2 = self.rho * self.EdX2 + (1.0 - self.rho) * delta ** 2
        return x + delta

    def reset(self):
        self.Eg2[:]  = 0.0
        self.EdX2[:] = 0.0
