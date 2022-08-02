import random

from gpflow.kernels import RBF
import tensorflow as tf

def generate_random_small_number():
    exp = random.randint(-7, -5)
    significand = 0.9 * random.random() + 0.1
    return significand * 10**exp


class SafeRBF(RBF):
    def scaled_squared_euclid_dist(self, X, X2=None) -> tf.Tensor:
        random_small_number = generate_random_small_number()
        r2 = self.square_dist(self.scale(X), self.scale(X2))
        return tf.sqrt(r2 + random_small_number) # or 1e-6. I find 1e-12 is too small