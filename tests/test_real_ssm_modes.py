import jax
import jax.numpy as jnp

from event_ssm.seq_model import BatchClassificationModel
from event_ssm.ssm import apply_rotation_pair, init_S5SSM, rotation_gamma


def test_rotation_pair_matches_decay_rotation_matrix():
    alpha = jnp.array([0.3])
    omega = jnp.array([1.7])
    dt = 0.4
    x = jnp.array([[0.2, -0.5]])

    a_re = jnp.exp(-alpha * dt) * jnp.cos(omega * dt)
    a_im = -jnp.exp(-alpha * dt) * jnp.sin(omega * dt)
    actual = apply_rotation_pair(a_re, a_im, x)

    expected_matrix = jnp.array([
        [jnp.exp(-alpha[0] * dt) * jnp.cos(omega[0] * dt),
         jnp.exp(-alpha[0] * dt) * jnp.sin(omega[0] * dt)],
        [-jnp.exp(-alpha[0] * dt) * jnp.sin(omega[0] * dt),
         jnp.exp(-alpha[0] * dt) * jnp.cos(omega[0] * dt)],
    ])
    expected = expected_matrix @ x[0]
    assert jnp.allclose(actual[0], expected, atol=1e-6)


def test_rotation_gamma_reduces_to_real_decay_when_omega_is_zero():
    alpha = jnp.array([0.4, 0.8])
    omega = jnp.zeros_like(alpha)
    dt = 0.7

    gamma_re, gamma_im = rotation_gamma(alpha, omega, dt)
    expected = (jnp.exp(-alpha * dt) - 1.0) / (-alpha)
    assert jnp.allclose(gamma_re, expected, atol=1e-6)
    assert jnp.allclose(gamma_im, jnp.zeros_like(gamma_im), atol=1e-6)


def test_real_modes_initialize_with_real_parameters_and_outputs():
    for a_mode in ["shared_real_decay", "independent_real_decay", "real_rotation2x2"]:
        ssm = init_S5SSM(
            C_init="lecun_normal",
            dt_min=0.004,
            dt_max=0.1,
            conj_sym=False,
            clip_eigs=False,
        )
        model = BatchClassificationModel(
            ssm=ssm,
            num_classes=20,
            num_embeddings=700,
            discretization="async",
            a_mode=a_mode,
            d_model=16,
            d_ssm=16,
            ssm_block_size=8,
            num_stages=1,
            num_layers_per_stage=2,
            dropout=0.1,
            classification_mode="timepool",
            prenorm=True,
            batchnorm=False,
            pooling_stride=2,
            pooling_mode="timepool",
            state_expansion_factor=1,
        )
        x = jnp.zeros((2, 16), dtype=jnp.int32)
        integration_timesteps = jnp.ones((2, 16), dtype=jnp.float32) * 0.001
        lengths = jnp.ones((2,), dtype=jnp.int32) * 16
        variables = model.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
            x,
            integration_timesteps,
            lengths,
            True,
        )
        logits = model.apply(variables, x, integration_timesteps, lengths, False)
        complex_params = [
            param for param in jax.tree_util.tree_leaves(variables["params"])
            if jnp.iscomplexobj(param)
        ]

        assert logits.shape == (2, 20)
        assert not jnp.iscomplexobj(logits)
        assert complex_params == []
