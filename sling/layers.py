import jax
import jax.numpy as jnp
import math
from flax import linen as nn
EPSILON = 1e-8

def get_delta_x(x):
    return jnp.expand_dims(x, -3) - jnp.expand_dims(x, -2)

def get_distance(x):
    x_minus_xt_norm = (
        jax.nn.relu((x ** 2).sum(axis=-1, keepdims=True))
    )
    return x_minus_xt_norm

def cosine_cutoff(x, lower=0.0, upper=5.0):
    cutoffs = 0.5 * (
        jnp.cos(
            math.pi
            * (
                2
                * (x - lower)
                / (upper - lower)
                + 1.0
            )
        )
        + 1.0
    )
    # remove contributions below the cutoff radius
    x = x * (x < upper)
    x = x * (x > lower)
    return cutoffs

class ExpNormalSmearing(nn.Module):
    cutoff_lower: float = 0.0
    cutoff_upper: float = 5.0
    num_rbf: float = 50

    def setup(self):
        self.alpha = 5.0 / (self.cutoff_upper - self.cutoff_lower)
        means, betas = self._initial_params()
        self.out_features = self.num_rbf
        self.means = self.param(
            "means",
            nn.initializers.constant(means),
            means.shape,
        )

        self.betas = self.param(
            "betas",
            nn.initializers.constant(betas),
            betas.shape,
        )

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = jnp.exp(
            -self.cutoff_upper + self.cutoff_lower
        )
        means = jnp.linspace(start_value, 1, self.num_rbf)
        betas = jnp.array(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def __call__(self, dist):
        return cosine_cutoff(dist) * jnp.exp(
            -self.betas
            * (jnp.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )
    

class SlingLayer(nn.Module):
    hidden_features : int
    out_features : int
    def setup(self):
        self.smearing = ExpNormalSmearing()
        self.fc = nn.Dense(
            self.hidden_features, use_bias=False,
        )

        self.fc_edge = nn.Dense(
            self.hidden_features, use_bias=False,
        )

        self.fc_summary = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                jax.nn.sigmoid,
                nn.Dense(1),
            ]
        )

    def __call__(self, h, x):
        # (n, c)
        h = self.fc(h)

        # (n, n, c)
        delta_x = get_delta_x(x)

        # (n, n, 1)
        distance = get_distance(delta_x) + EPSILON

        # (n, n, 3)
        delta_x_norm = delta_x / (jax.nn.relu((delta_x ** 2).sum(axis=-1, keepdims=True) + EPSILON) ** 0.5)
        
        # (n, n, c)
        h_e_vec = self.fc_edge(self.smearing(distance))

        # (n, n, c, 3)
        h_e_vec = jnp.expand_dims(h_e_vec, -1) * jnp.expand_dims(delta_x_norm, -2)

        # (n, n, c)
        edge_scalar_embedding = jnp.expand_dims(h, -2) + jnp.expand_dims(h, -3)

        # (n, n, c, 3)
        combination = jnp.expand_dims(edge_scalar_embedding, -1) * h_e_vec

        # (n, n, c)
        combination_norm = (combination ** 2).sum(-1)

        # (n, n, 1)
        combination_energy = self.fc_summary(combination_norm).sum(axis=(-2, -3))

        return combination_energy
    