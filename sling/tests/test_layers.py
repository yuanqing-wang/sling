import pytest

def test_layer():
    import numpy as onp
    import jax
    import jax.numpy as jnp
    from sling.layers import SlingLayer
    layer = SlingLayer(4, 8, 16)
    h = jnp.array(onp.random.randn(2, 4))
    x = jnp.array(onp.random.randn(2, 3))
    params = layer.init(jax.random.PRNGKey(2666), h, x)
    y = layer.apply(params, h, x)
    return y

def test_invariance(_equivariance_test_utils):
    import jax
    import jax.numpy as jnp
    from sling.layers import SlingLayer
    layer = SlingLayer(4, 8, 16)
    
    h0, x0, translation, rotation, reflection = _equivariance_test_utils
    params = layer.init(jax.random.PRNGKey(2666), h0, x0)

    h_original = layer.apply(params, h0, x0)
    h_translation = layer.apply(params, h0, translation(x0))
    h_rotation = layer.apply(params, h0, rotation(x0))
    h_reflection = layer.apply(params, h0, reflection(x0))

    assert jnp.allclose(h_translation, h_original)
    assert jnp.allclose(h_rotation, h_original)
    assert jnp.allclose(h_reflection, h_original)
