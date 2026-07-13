from functools import partial
import jax
import jax.numpy as np
from jax.scipy.linalg import block_diag

from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal, glorot_normal

from .ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal, make_DPLR_HiPPO

from .layers import EventPooling


def discretize_zoh(Lambda, step_delta, time_delta):
    """
    Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    This is the default discretization method used by many SSM works including S5.

    :param Lambda: diagonal state matrix (P,)
    :param step_delta: discretization step sizes (P,)
    :param time_delta: (float32) discretization step sizes (P,)
    :return: discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Delta = step_delta * time_delta
    Lambda_bar = np.exp(Lambda * Delta)
    gamma_bar = (1/Lambda * (Lambda_bar-Identity))
    return Lambda_bar, gamma_bar


def discretize_dirac(Lambda, step_delta, time_delta):
    """
    Discretize a diagonalized, continuous-time linear SSM
    with dirac delta input spikes.
    :param Lambda: diagonal state matrix (P,)
    :param step_delta: discretization step sizes (P,)
    :param time_delta: (float32) discretization step sizes (P,)
    :return: discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)
    """
    Delta = step_delta * time_delta
    Lambda_bar = np.exp(Lambda * Delta)
    gamma_bar = 1.0
    return Lambda_bar, gamma_bar


def discretize_async(Lambda, step_delta, time_delta):
    """
    Discretize a diagonalized, continuous-time linear SSM
    with dirac delta input spikes and appropriate input normalization.

    :param Lambda: diagonal state matrix (P,)
    :param step_delta: discretization step sizes (P,)
    :param time_delta: (float32) discretization step sizes (P,)
    :return: discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * step_delta * time_delta)
    gamma_bar = (1/Lambda * (np.exp(Lambda * step_delta)-Identity))
    return Lambda_bar, gamma_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """
    Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.

    :param q_i: tuple containing A_i and Bu_i at position i (P,), (P,)
    :param q_j: tuple containing A_j and Bu_j at position j (P,), (P,)
    :return: new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_elements, Bu_elements, C_tilde, conj_sym, stride=1):
    """
    Compute the LxH output of discretized SSM given an LxH input.

    :param Lambda_elements: (complex64) discretized state matrix (L, P)
    :param Bu_elements: (complex64) discretized inputs projected to state space (L, P)
    :param C_tilde: (complex64) output matrix (H, P)
    :param conj_sym: (bool) whether conjugate symmetry is enforced
    :return: ys: (float32) the SSM outputs (S5 layer preactivations) (L, H)
    """
    remaining_timesteps = (Bu_elements.shape[0] // stride) * stride

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    xs = xs[:remaining_timesteps:stride]

    if conj_sym:
        return jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(xs)


def apply_real_diagonal_ssm(Lambda_elements, Bu_elements, C, stride=1):
    """
    Compute the SSM output for a real diagonal state matrix.
    """
    remaining_timesteps = (Bu_elements.shape[0] // stride) * stride
    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
    xs = xs[:remaining_timesteps:stride]
    return jax.vmap(lambda x: C @ x)(xs)


def apply_rotation_pair(a_re, a_im, x):
    """
    Apply a 2x2 real block represented by a complex-like pair.
    The represented matrix is [[a_re, -a_im], [a_im, a_re]].
    """
    return np.stack((
        a_re * x[..., 0] - a_im * x[..., 1],
        a_im * x[..., 0] + a_re * x[..., 1],
    ), axis=-1)


@jax.vmap
def binary_operator_rotation(q_i, q_j):
    """
    Binary operator for block-diagonal 2x2 rotation-decay recurrences.
    """
    (a_re_i, a_im_i), b_i = q_i
    (a_re_j, a_im_j), b_j = q_j

    a_re = a_re_j * a_re_i - a_im_j * a_im_i
    a_im = a_re_j * a_im_i + a_im_j * a_re_i
    b = apply_rotation_pair(a_re_j, a_im_j, b_i) + b_j
    return (a_re, a_im), b


def apply_rotation2x2_ssm(A_elements, Bu_elements, C, stride=1):
    """
    Compute the SSM output for real 2x2 rotation-decay blocks.
    """
    remaining_timesteps = (Bu_elements.shape[0] // stride) * stride
    _, xs = jax.lax.associative_scan(binary_operator_rotation, (A_elements, Bu_elements))
    xs = xs[:remaining_timesteps:stride]
    xs = xs.reshape(xs.shape[0], -1)
    return jax.vmap(lambda x: C @ x)(xs)


def inverse_softplus(x):
    return np.log(np.expm1(x))


def rotation_gamma(alpha, omega, delta):
    """
    Compute (exp(A * delta) - I) A^{-1} for 2x2 rotation-decay blocks.
    The block A is [[-alpha, omega], [-omega, -alpha]].
    """
    real = -alpha
    imag = -omega
    exp_real = np.exp(real * delta) * np.cos(imag * delta)
    exp_imag = np.exp(real * delta) * np.sin(imag * delta)
    numerator_real = exp_real - 1.0
    numerator_imag = exp_imag
    denominator = real ** 2 + imag ** 2
    gamma_real = (numerator_real * real + numerator_imag * imag) / denominator
    gamma_imag = (numerator_imag * real - numerator_real * imag) / denominator
    return gamma_real, gamma_imag


class S5SSM(nn.Module):
    H_in: int
    H_out: int
    P: int
    block_size: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    step_rescale: float = 1.0
    stride: int = 1
    pooling_mode: str = "last"
    a_mode: str = "complex_diagonal"

    """
    Event-based S5 module
    
    :param H_in: int, SSM input dimension
    :param H_out: int, SSM output dimension
    :param P: int, SSM state dimension
    :param block_size: int, block size for block-diagonal state matrix
    :param C_init: str, initialization method for output matrix C
    :param discretization: str, discretization method for event-based SSM
    :param dt_min: float, minimum value of log timestep
    :param dt_max: float, maximum value of log timestep
    :param conj_sym: bool, whether to enforce conjugate symmetry in the state space operator
    :param clip_eigs: bool, whether to clip eigenvalues of the state space operator
    :param step_rescale: float, rescale factor for step size
    :param stride: int, stride for subsampling layer
    :param pooling_mode: str, pooling mode for subsampling layer
    :param a_mode: str, state matrix parameterization
    """

    def setup(self):
        """
        Initializes parameters once and performs discretization each time the SSM is applied to a sequence
        """

        self.real_mode = self.a_mode in [
            "shared_real_decay",
            "independent_real_decay",
            "real_rotation2x2",
        ]
        if self.a_mode not in [
            "complex_diagonal",
            "shared_real_decay",
            "independent_real_decay",
            "real_rotation2x2",
        ]:
            raise NotImplementedError(f"A mode {self.a_mode} not implemented")

        if self.P % self.block_size != 0:
            raise ValueError(f"P={self.P} must be divisible by block_size={self.block_size}")

        # Initialize state matrix A using approximation to HiPPO-LegS matrix.
        Lambda_base, _, _, V_base, _ = make_DPLR_HiPPO(self.block_size)

        if self.real_mode:
            blocks = self.P // self.block_size
            Lambda = (Lambda_base * np.ones((blocks, self.block_size))).ravel()
            local_P = self.P
        else:
            blocks = self.P // self.block_size
            block_size = self.block_size // 2 if self.conj_sym else self.block_size
            local_P = self.P // 2 if self.conj_sym else self.P

            Lambda = Lambda_base[:block_size]
            V = V_base[:, :block_size]
            Vc = V.conj().T

            # If initializing state matrix A as block-diagonal, put HiPPO approximation
            # on each block
            Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
            V = block_diag(*([V] * blocks))
            Vinv = block_diag(*([Vc] * blocks))

        state_str = f"SSM: {self.H_in} -> {self.P} -> {self.H_out} ({self.a_mode})"
        if self.stride > 1:
            state_str += f" (stride {self.stride} with pooling mode {self.pooling_mode})"
        print(state_str)

        B_init = lecun_normal()
        if self.real_mode:
            if self.P % 2 != 0:
                raise ValueError("Real-valued comparison modes require an even state size P")

            alpha_init = np.maximum(-Lambda.real, 1e-4)
            if self.a_mode == "shared_real_decay":
                raw_alpha_init = inverse_softplus(np.array([alpha_init.mean()]))
            elif self.a_mode == "independent_real_decay":
                raw_alpha_init = inverse_softplus(alpha_init)
            else:
                raw_alpha_init = inverse_softplus(alpha_init.reshape(-1, 2).mean(axis=-1))
                omega_init = np.abs(Lambda.imag[:self.P // 2])
                self.omega = self.param("omega", lambda rng, shape: omega_init, (None,))

            self.raw_alpha = self.param("raw_alpha", lambda rng, shape: raw_alpha_init, (None,))
            # One time scale per real-state pair keeps the diagonal and 2x2
            # parameterizations equally sized while retaining S5's stable,
            # learnable async discretization scale.
            self.log_step = self.param("log_step",
                                       init_log_steps,
                                       (self.P // 2, self.dt_min, self.dt_max))
            self.B = self.param("B", B_init, (self.P, self.H_in))
            self.C = self.param("C", lecun_normal(), (self.H_out, self.P))
        else:
            # Initialize diagonal state to state matrix Lambda (eigenvalues)
            self.Lambda_re = self.param("Lambda_re", lambda rng, shape: Lambda.real, (None,))
            self.Lambda_im = self.param("Lambda_im", lambda rng, shape: Lambda.imag, (None,))

            if self.clip_eigs:
                self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
            else:
                self.Lambda = self.Lambda_re + 1j * self.Lambda_im

            # Initialize input to state (B) matrix
            B_shape = (self.P, self.H_in)
            self.B = self.param("B",
                                lambda rng, shape: init_VinvB(B_init, rng, shape, Vinv),
                                B_shape)

            # Initialize state to output (C) matrix
            if self.C_init in ["trunc_standard_normal"]:
                C_init = trunc_standard_normal
                C_shape = (self.H_out, self.P, 2)
            elif self.C_init in ["lecun_normal"]:
                C_init = lecun_normal()
                C_shape = (self.H_out, self.P, 2)
            elif self.C_init in ["complex_normal"]:
                C_init = normal(stddev=0.5 ** 0.5)
            else:
                raise NotImplementedError(
                       "C_init method {} not implemented".format(self.C_init))

            if self.C_init in ["complex_normal"]:
                C = self.param("C", C_init, (self.H_out, local_P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

            else:
                self.C = self.param("C",
                                    lambda rng, shape: init_CV(C_init, rng, shape, V),
                                    C_shape)

                self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

            # Initialize learnable discretization timescale value
            self.log_step = self.param("log_step",
                                       init_log_steps,
                                       (local_P, self.dt_min, self.dt_max))

        # Initialize feedthrough (D) matrix
        if self.H_in == self.H_out:
            self.D = self.param("D", normal(stddev=1.0), (self.H_in,))
        else:
            self.D = self.param("D", glorot_normal(), (self.H_out, self.H_in))

        # pooling layer
        self.pool = EventPooling(stride=self.stride, mode=self.pooling_mode)

        # Discretize
        if self.discretization in ["zoh"]:
            self.discretize_fn = discretize_zoh
        elif self.discretization in ["dirac"]:
            self.discretize_fn = discretize_dirac
        elif self.discretization in ["async"]:
            self.discretize_fn = discretize_async
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

    def get_alpha(self):
        alpha = jax.nn.softplus(self.raw_alpha) + 1e-6
        if self.a_mode == "shared_real_decay":
            return np.ones((self.P,)) * alpha[0]
        elif self.a_mode == "independent_real_decay":
            return alpha
        elif self.a_mode == "real_rotation2x2":
            return alpha
        else:
            raise NotImplementedError(f"A mode {self.a_mode} not implemented")

    def get_real_step(self):
        step = self.step_rescale * np.exp(self.log_step[:, 0])
        if self.a_mode in ["shared_real_decay", "independent_real_decay"]:
            return np.repeat(step, 2)
        return step

    def diagonal_real_discretize(self, alpha, step, time_delta):
        Lambda = -alpha

        if self.discretization == "zoh":
            Delta = step * time_delta
            Lambda_bar = np.exp(Lambda * Delta)
            gamma_bar = (Lambda_bar - 1.0) / Lambda
        elif self.discretization == "dirac":
            Lambda_bar = np.exp(Lambda * step * time_delta)
            gamma_bar = 1.0
        elif self.discretization == "async":
            Lambda_bar = np.exp(Lambda * step * time_delta)
            gamma_bar = (np.exp(Lambda * step) - 1.0) / Lambda
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

        return Lambda_bar, gamma_bar

    def rotation2x2_discretize(self, alpha, omega, step, time_delta):
        def exp_rotation(delta):
            decay = np.exp(-alpha * delta)
            theta = omega * delta
            return decay * np.cos(theta), -decay * np.sin(theta)

        if self.discretization == "zoh":
            A_re, A_im = exp_rotation(step * time_delta)
            gamma_re, gamma_im = rotation_gamma(alpha, omega, step * time_delta)
        elif self.discretization == "dirac":
            A_re, A_im = exp_rotation(step * time_delta)
            gamma_re = np.ones_like(alpha)
            gamma_im = np.zeros_like(alpha)
        elif self.discretization == "async":
            A_re, A_im = exp_rotation(step * time_delta)
            gamma_re, gamma_im = rotation_gamma(alpha, omega, step)
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

        return (A_re, A_im), (gamma_re, gamma_im)

    def __call__(self, input_sequence, integration_timesteps):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence using a parallel scan.

        :param input_sequence: (float32) input sequence (L, H)
        :param integration_timesteps: (float32) integration timesteps (L,)
        :return: (float32) output sequence (L, H)
        """

        if self.a_mode in ["shared_real_decay", "independent_real_decay"]:
            alpha = self.get_alpha()
            step = self.get_real_step()

            def discretize_and_project_inputs(u, _timestep):
                Lambda_bar, gamma_bar = self.diagonal_real_discretize(alpha, step, _timestep)
                Bu = gamma_bar * (self.B @ u)
                return Lambda_bar, Bu

            Lambda_bar_elements, Bu_bar_elements = jax.vmap(discretize_and_project_inputs)(
                input_sequence, integration_timesteps
            )
            ys = apply_real_diagonal_ssm(
                Lambda_bar_elements,
                Bu_bar_elements,
                self.C,
                stride=self.stride
            )

        elif self.a_mode == "real_rotation2x2":
            alpha = self.get_alpha()
            omega = self.omega
            step = self.get_real_step()

            def discretize_and_project_inputs(u, _timestep):
                A, gamma = self.rotation2x2_discretize(alpha, omega, step, _timestep)
                projected = (self.B @ u).reshape(-1, 2)
                Bu = apply_rotation_pair(gamma[0], gamma[1], projected)
                return A, Bu

            A_elements, Bu_elements = jax.vmap(discretize_and_project_inputs)(
                input_sequence, integration_timesteps
            )
            ys = apply_rotation2x2_ssm(
                A_elements,
                Bu_elements,
                self.C,
                stride=self.stride
            )

        else:
            # discretize on the fly
            B = self.B[..., 0] + 1j * self.B[..., 1]

            def discretize_and_project_inputs(u, _timestep):
                step = self.step_rescale * np.exp(self.log_step[:, 0])
                Lambda_bar, gamma_bar = self.discretize_fn(self.Lambda, step, _timestep)
                Bu = gamma_bar * (B @ u)
                return Lambda_bar, Bu

            Lambda_bar_elements, Bu_bar_elements = jax.vmap(discretize_and_project_inputs)(
                input_sequence, integration_timesteps
            )

            ys = apply_ssm(
                Lambda_bar_elements,
                Bu_bar_elements,
                self.C_tilde,
                self.conj_sym,
                stride=self.stride
            )

        if self.stride > 1:
            input_sequence, _ = self.pool(input_sequence, integration_timesteps)

        if self.H_in == self.H_out:
            Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        else:
            Du = jax.vmap(lambda u: self.D @ u)(input_sequence)

        return ys + Du


def init_S5SSM(
        C_init,
        dt_min,
        dt_max,
        conj_sym,
        clip_eigs,
        a_mode="complex_diagonal",
):
    """
    Convenience function that will be used to initialize the SSM.
    Same arguments as defined in S5SSM above.
    """
    return partial(S5SSM,
                   C_init=C_init,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   conj_sym=conj_sym,
                   clip_eigs=clip_eigs,
                   a_mode=a_mode,
                   )
