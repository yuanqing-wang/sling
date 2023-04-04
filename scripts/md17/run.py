import jax
import jax.numpy as jnp
import optax
import numpy as onp
import sling
import tqdm

def run(data_name):
    data = onp.load("%s_dft.npz" % data_name)
    onp.random.seed(2666)
    idxs = onp.random.permutation(len(data['R']))

    x = jnp.array(data['R'][idxs])
    e = jnp.array(data['E'][idxs])
    i = jnp.array(data['z'])
    f = jnp.array(data['F'][idxs])
    i = jax.nn.one_hot(i, i.max())

    batch_size = 4
    n_tr = n_vl = 1000
    x_tr = x[:n_tr]
    e_tr = e[:n_tr]
    f_tr = f[:n_tr]

    x_vl = x[n_tr:n_tr+n_vl]
    e_vl = e[n_tr:n_tr+n_vl]
    f_vl = f[n_tr:n_tr+n_vl]

    x_te = x[n_tr+n_vl:]
    e_te = e[n_tr+n_vl:]
    f_te = f[n_tr+n_vl:]

    n_batches = int(n_tr / batch_size)

    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=e_tr.mean(), std=e_tr.std())
    e_tr_mean = e_tr.mean().item()

    from sling.layers import SlingLayer    
    model = SlingLayer(
        hidden_features=16,
        out_features=1,
    )

    # @jax.jit
    def get_e_pred(params, x):
        i_tr = jnp.broadcast_to(i, (*x.shape[:-1], i.shape[-1]))
        e_pred = model.apply(params, i_tr, x)
        e_pred = e_pred.sum(axis=-2)
        e_pred = coloring(e_pred)
        return e_pred

    def get_e_pred_sum(params, x):
        e_pred = get_e_pred(params, x)
        return -e_pred.sum()

    get_f_pred = jax.jit(lambda params, x: jax.grad(get_e_pred_sum, argnums=(1,))(params, x)[0])

    def loss_fn(params, x, e, f):
        e_pred = get_e_pred(params, x)
        f_pred = get_f_pred(params, x)
        e_loss = jnp.abs(e_pred - e).mean()
        f_loss = jnp.abs(f_pred - f).mean()
        return f_loss # f_loss + e_loss * 0.001

    # @jax.jit
    def step(state, x, e, f):
        params = state.params
        loss, grads = jax.value_and_grad(loss_fn)(params, x, e, f)
        state = state.apply_gradients(grads=grads)
        return state

    # @jax.jit
    def epoch(state, x_tr, e_tr, f_tr):
        key = jax.random.PRNGKey(state.step)
        idxs = jax.random.permutation(key, jnp.arange(x_tr.shape[0]))
        x_tr, e_tr, f_tr = x_tr[idxs], e_tr[idxs], f_tr[idxs]

        x_tr = x_tr.reshape(n_batches, batch_size, x_tr.shape[-2], x_tr.shape[-1])
        e_tr = e_tr.reshape(n_batches, batch_size, e_tr.shape[-1])
        f_tr = f_tr.reshape(n_batches, batch_size, f_tr.shape[-2], f_tr.shape[-1])

        def loop_body(idx_batch, state):
            x = x_tr[idx_batch]
            e = e_tr[idx_batch]
            f = f_tr[idx_batch]
            state = step(state, x, e, f)
            return state

        state = jax.lax.fori_loop(0, n_batches, loop_body, state)
        return state

    from functools import partial

    # @partial(jax.jit, static_argnums=(6,))
    def many_epochs(state, x_tr, e_tr, f_tr, n=10):
        def loop_body(idx, state):
            state = epoch(state, x_tr, e_tr, f_tr)
            return state
        state = jax.lax.fori_loop(0, n, loop_body, state)
        return state

    key = jax.random.PRNGKey(2666)
    x0 = x_tr[:batch_size]
    i_tr = jnp.broadcast_to(i, (*x0.shape[:-1], i.shape[-1]))
    params = model.init(key, i_tr, x0)

    optimizer = optax.adam(1e-10)

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint, restore_checkpoint
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    # state = restore_checkpoint("_"+data_name, target=state)
    # import os
    # idx_step = len(os.listdir("_" + data_name))

    for idx_batch in tqdm.tqdm(range(2000)):
        import time
        # state = many_epochs(state, x_tr, e_tr, f_tr)
        state = epoch(state, x_tr, e_tr, f_tr)
        save_checkpoint("_" + data_name, target=state, step=idx_batch, keep=99999)

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
