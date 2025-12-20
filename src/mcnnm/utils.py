from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .types import Array, Scalar

jax.config.update("jax_enable_x64", True)


def convert_inputs(
    Y: pd.DataFrame,
    W: pd.DataFrame,
    X: pd.DataFrame | None = None,
    Z: pd.DataFrame | None = None,
    V: list[pd.DataFrame] | None = None,
    Omega: pd.DataFrame | None = None,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray | None,
    jnp.ndarray | None,
    jnp.ndarray | None,
    jnp.ndarray | None,
]:
    """
    Convert input DataFrames to JAX arrays for the MC-NNM model.
    """
    Y_arr = jnp.array(Y.values)
    W_arr = jnp.array(W.values)
    N, T = Y_arr.shape

    X_arr: jnp.ndarray | None = None
    Z_arr: jnp.ndarray | None = None
    V_arr: jnp.ndarray | None = None
    Omega_arr: jnp.ndarray | None = None

    if X is not None:
        X_arr = jnp.array(X.values)
        if X_arr.shape[0] != N:
            raise ValueError(
                f"The first dimension of X ({X_arr.shape[0]}) must match the first dimension of Y ({N}).",
            )

    if Z is not None:
        Z_arr = jnp.array(Z.values)
        if Z_arr.shape[0] != T:
            raise ValueError(
                f"The first dimension of Z ({Z_arr.shape[0]}) must match the second dimension of Y ({T}).",
            )

    if V is not None:
        if len(V) == 0:
            raise ValueError("V cannot be an empty list.")
        V_list = [jnp.array(df.values) for df in V]
        V_arr = jnp.stack(V_list, axis=2)
        if V_arr.shape[:2] != (N, T):
            raise ValueError(f"The shape of V must match the shape of Y ({N}, {T}).")

    if Omega is not None:
        Omega_arr = jnp.array(Omega.values)
        if Omega_arr.shape != (T, T):
            raise ValueError(f"The shape of Omega ({Omega_arr.shape}) must be ({T}, {T}).")

    return Y_arr, W_arr, X_arr, Z_arr, V_arr, Omega_arr


def check_inputs(
    Y: Array,
    W: Array,
    X: Array | None = None,
    Z: Array | None = None,
    V: Array | None = None,
    Omega: Array | None = None,
) -> tuple[Array, Array, Array, Array]:
    """
    Check and preprocess input arrays for the MC-NNM model.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        W (Array): The binary treatment matrix of shape (N, T).
        X (Optional[Array]): The unit-specific covariates matrix of shape (N, P). Default is None.
        Z (Optional[Array]): The time-specific covariates matrix of shape (T, Q). Default is None.
        V (Optional[Array]): The unit-time specific covariates tensor of shape (N, T, J). Default is None.
        Omega (Optional[Array]): The autocorrelation matrix of shape (T, T). Default is None.

    Returns:
        Tuple: A tuple containing preprocessed X, Z, V, and Omega arrays.

    Raises:
        ValueError: If the input array dimensions are mismatched or invalid.

    """
    N, T = Y.shape

    if W.shape != (N, T):
        raise ValueError(f"The shape of W ({W.shape}) must match the shape of Y ({Y.shape}).")

    if not jnp.all((W == 0) | (W == 1)):
        raise ValueError("The mask must be binary where 0 denotes observed and 1 treated values")

    if X is not None:
        if X.shape[0] != N:
            raise ValueError(
                f"The first dimension of X ({X.shape[0]}) must match the first dimension of Y ({N}).",
            )
    else:
        X = jnp.zeros((N, 0))

    if Z is not None:
        if Z.shape[0] != T:
            raise ValueError(
                f"The first dimension of Z ({Z.shape[0]}) must match the second dimension of Y ({T}).",
            )
    else:
        Z = jnp.zeros((T, 0))

    if V is not None:
        if V.shape[:2] != (N, T):
            raise ValueError(
                f"The first two dimensions of V ({V.shape[:2]}) must match the shape of Y ({Y.shape}).",
            )
    else:
        V = jnp.zeros((N, T, 1))  # Add a dummy dimension if V is None, otherwise causes issues

    if Omega is not None:
        if Omega.shape != (T, T):
            raise ValueError(f"The shape of Omega ({Omega.shape}) must be ({T}, {T}).")
    else:
        Omega = jnp.eye(T)

    return X, Z, V, Omega


def generate_data(
    nobs: int = 500,
    nperiods: int = 100,
    Y_mean: float = 10.0,
    treatment_probability: float = 0.5,
    rank: int = 5,
    treatment_effect: float = 5.0,
    unit_fe: bool = True,
    time_fe: bool = True,
    X_cov: bool = True,
    Z_cov: bool = True,
    V_cov: bool = True,
    fixed_effects_scale: float = 0.5,
    covariates_scale: float = 0.5,
    noise_scale: float = 1,
    assignment_mechanism: Literal[
        "staggered",
        "block",
        "single_treated_period",
        "single_treated_unit",
        "last_periods",
    ] = "last_periods",
    treated_fraction: float = 0.2,
    last_treated_periods: int = 2,
    autocorrelation: float = 0.0,
    seed: int | None = None,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray | None,
    jnp.ndarray | None,
    jnp.ndarray | None,
    dict,
]:
    if seed is not None:
        np.random.seed(seed)

    if not 0 <= autocorrelation < 1:
        raise ValueError("Autocorrelation must be between 0 and 1 (exclusive).")

    # Generate low-rank matrix L
    U = np.random.normal(0, 1, (nobs, rank))
    V = np.random.normal(0, 1, (nperiods, rank))
    L = U @ V.T

    # Generate fixed effects
    unit_fe_values = np.random.normal(0, fixed_effects_scale, nobs) if unit_fe else np.zeros(nobs)
    time_fe_values = np.random.normal(0, fixed_effects_scale, nperiods) if time_fe else np.zeros(nperiods)

    # generate random offsets to y_mean
    X_mean = Y_mean - np.random.normal(2, 1)
    Z_mean = Y_mean + np.random.normal(2, 1)
    V_mean = Y_mean - np.random.normal(4, 1)

    # Generate covariates and their coefficients
    X = jnp.array(np.random.normal(X_mean, covariates_scale, (nobs, 2))) if X_cov else None
    X_coef = np.random.normal(0, covariates_scale, 2) if X_cov else np.array([])
    Z = jnp.array(np.random.normal(Z_mean, covariates_scale, (nperiods, 2))) if Z_cov else None
    Z_coef = np.random.normal(0, covariates_scale, 2) if Z_cov else np.array([])
    V = (
        jnp.array(np.random.normal(V_mean, covariates_scale, (nobs, nperiods, 2)))  # type: ignore
        if V_cov
        else None
    )  # type: ignore
    V_coef = np.random.normal(0, covariates_scale, 2) if V_cov else np.array([])

    # Generate autocorrelated errors
    errors = np.zeros((nobs, nperiods))
    for i in range(nobs):
        errors[i, 0] = np.random.normal(0, noise_scale)
        for t in range(1, nperiods):
            errors[i, t] = autocorrelation * errors[i, t - 1] + np.random.normal(
                0,
                noise_scale * np.sqrt(1 - autocorrelation**2),
            )

    # Generate outcome
    Y = (
        L
        + Y_mean
        + np.outer(unit_fe_values, np.ones(nperiods))
        + np.outer(np.ones(nobs), time_fe_values)
        + (np.repeat(X @ X_coef, nperiods).reshape(nobs, nperiods) if X_cov else 0)  # type: ignore
        + (np.tile((Z @ Z_coef).reshape(1, -1), (nobs, 1)) if Z_cov else 0)  # type: ignore
        + (np.sum(V * V_coef, axis=2) if V_cov and V is not None else 0)  # type: ignore[operator]
        + errors
    )

    # Generate treatment assignment based on the specified mechanism
    if assignment_mechanism == "staggered":
        treat = np.zeros((nobs, nperiods), dtype=int)
        min_adoption_time = nperiods // 4  # Earliest possible adoption time

        # Generate adoption times starting from min_adoption_time
        adoption_times = np.random.geometric(p=treatment_probability, size=nobs) + min_adoption_time - 1

        for i in range(nobs):  # pragma: no cover
            if adoption_times[i] < nperiods:
                treat[i, adoption_times[i] :] = 1
    elif assignment_mechanism == "block":
        treated_units = np.random.choice(nobs, size=int(nobs * treated_fraction), replace=False)
        treat = np.zeros((nobs, nperiods), dtype=int)
        treat[treated_units, nperiods // 3 :] = 1
    elif assignment_mechanism == "single_treated_period":
        treated_units = np.random.choice(nobs, size=int(nobs * treated_fraction), replace=False)
        treat = np.zeros((nobs, nperiods), dtype=int)
        treat[treated_units, -1] = 1
    elif assignment_mechanism == "single_treated_unit":
        treated_unit = np.random.choice(nobs)
        treat = np.zeros((nobs, nperiods), dtype=int)
        treat[treated_unit, nperiods // 2 :] = 1
    elif assignment_mechanism == "last_periods":
        treat = np.zeros((nobs, nperiods), dtype=int)
        treat[:, -last_treated_periods:] = 1
    else:
        raise ValueError("Invalid assignment mechanism specified.")

    Y_0 = Y.copy()  # untreated potential outcome
    Y += treat * treatment_effect

    # Convert Y, W, and covariates to JAX arrays
    Y = jnp.array(Y)
    W = jnp.array(treat, dtype=jnp.float64)
    X = jnp.array(X) if X is not None else None
    Z = jnp.array(Z) if Z is not None else None
    V = jnp.array(V) if V is not None else None  # type: ignore[assignment]

    true_params = {
        "L": L,
        "unit_fe": unit_fe_values,
        "time_fe": time_fe_values,
        "X": X,
        "X_coef": X_coef,
        "Z": Z,
        "Z_coef": Z_coef,
        "V": V,
        "V_coef": V_coef,
        "treatment_effect": treatment_effect,
        "noise_scale": noise_scale,
        "assignment_mechanism": assignment_mechanism,
        "autocorrelation": autocorrelation,
        "Y(0)": jnp.array(Y_0),
    }

    return Y, W, X, Z, V, true_params  # type: ignore


def propose_lambda_values(
    max_lambda: Scalar,
    min_lambda: Scalar | None = None,
    n_lambdas: int = 6,
) -> Array:
    """
    Creates a decreasing, log-spaced list of proposed lambda values between max_lambda and min_lambda,
    and appends 0 to the end of the list.

    Args:
        max_lambda: The maximum lambda value.
        min_lambda: The minimum lambda value. If None, it is set to max_lambda - 3 (in log10 scale).
        n_lambdas: The number of lambda values to generate (excluding the appended 0).

    Returns:
        Array: The decreasing sequence of proposed lambda values, including 0 at the end.

    Raises:
        ValueError: If max_lambda is smaller than the default minimum lambda value (1e-10).

    """
    min_log_lambda = jnp.log10(max_lambda) - 3 if min_lambda is None else jnp.log10(min_lambda)
    max_log_lambda = jnp.log10(max_lambda)

    if min_lambda and max_lambda < min_lambda:
        raise ValueError("max_lambda must be greater than or equal to min_lambda.")

    if max_log_lambda < -10:
        raise ValueError(
            f"max_lambda ({max_lambda}) is too small. It should be greater than or equal to 1e-10.",
        )
    if n_lambdas < 2:
        raise ValueError("n_lambdas must be greater than or equal to 2.")
    # Ensure min_log_lambda is not smaller than a small positive value to avoid zero or negative lambdas
    min_log_lambda = jnp.maximum(min_log_lambda, -10)

    lambda_values = jnp.logspace(max_log_lambda, min_log_lambda, n_lambdas - 1)
    lambda_values = jnp.append(lambda_values, 0.0)

    return lambda_values


def generate_lambda_grid(lambda_L_values: Array, lambda_H_values: Array) -> jnp.ndarray:
    """
    Generates a grid of lambda values for the MC-NNM model.

    This function creates a 2D grid of lambda values by forming a meshgrid from the provided sequences
    for both lambda_L and lambda_H.

    Args:
        lambda_L_values (Array): The decreasing sequence of lambda values for the L dimension.
        lambda_H_values (Array): The decreasing sequence of lambda values for the H dimension.

    Returns:
        jnp.ndarray: A 2D array where each row represents a pair of lambda values (lambda_L, lambda_H).

    """
    lambda_grid = jnp.array(jnp.meshgrid(lambda_L_values, lambda_H_values, indexing="ij")).reshape(2, -1).T
    return lambda_grid


def extract_shortest_path(lambda_grid):
    """
    Extracts the shortest path along the edges of a 2D lambda grid.

    This function traverses the edges of the given lambda grid starting from the top-right corner
    (highest lambda_L and lambda_H), moving left to the top-left corner (lowest lambda_L, highest lambda_H),
    and then down to the bottom-left corner (lowest lambda_L and lambda_H).

    Args:
        lambda_grid (jnp.ndarray): A 2D array representing the flattened lambda grid.

    Returns:
        jnp.ndarray: A 2D array containing the lambda pairs along the shortest path.

    """
    # Get unique lambda values
    unique_lambda_L = jnp.unique(lambda_grid[:, 0])
    unique_lambda_H = jnp.unique(lambda_grid[:, 1])

    # Sort unique values in descending order
    unique_lambda_L = jnp.sort(unique_lambda_L)[::-1]
    unique_lambda_H = jnp.sort(unique_lambda_H)[::-1]

    # Create the path
    path = []

    # Move from highest to lowest lambda_L, keeping lambda_H at its maximum
    for lambda_L in unique_lambda_L:
        path.append([lambda_L, unique_lambda_H[0]])

    # Move from highest to lowest lambda_H, keeping lambda_L at its minimum
    for lambda_H in unique_lambda_H[1:]:
        path.append([unique_lambda_L[-1], lambda_H])

    return jnp.array(path)


def generate_holdout_val_defaults(Y: Array):
    """
    Generates default parameters for time-based validation.

    This function calculates default values for various parameters used in time-based validation,
    including the initial window size, step size, horizon, and lambda grid.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).

    Returns:
        dict: A dictionary containing the default parameters for time-based validation.
            - initial_window (int): Number of initial time periods to use for the first training set.
            - step_size (int): Number of time periods to move forward for each split.
            - horizon (int): Number of future time periods to predict (forecast horizon).

    """
    N, T = Y.shape
    T = int(T)

    initial_window = int(0.8 * T)
    K = 5
    step_size = max(1, (T - initial_window) // K)
    horizon = step_size

    return initial_window, step_size, horizon, K


def validate_holdout_config(
    initial_window: int,
    step_size: int,
    horizon: int,
    K: int,
    max_window_size: int | None,
    T: int,
) -> tuple[int, int, int, int, int | None]:
    """
    Validate the configuration of initial_window, step_size, horizon, K, and max_window_size for holdout validation.

    Args:
        initial_window (int): The size of the initial time window for holdout validation.
        step_size (int): The step size for moving the time window in each holdout fold.
        horizon (int): The size of the holdout horizon (number of time steps to predict).
        K (int): The number of holdout folds.
        max_window_size (int, optional): The maximum size of the time window. If specified, it limits the
            size of the time window used for initializing the model configurations in each holdout fold.
        T (int): The total number of time steps in the data.

    Returns:
        Tuple[int, int, int, int, Optional[int]]: A tuple containing the validated or default values for
            initial_window, step_size, horizon, K, and max_window_size.

    Raises:
        ValueError: If the configuration is invalid or inconsistent and cannot be adjusted to sensible defaults.

    """
    if initial_window <= 0:
        raise ValueError("initial_window must be greater than 0.")
    if step_size <= 0:
        raise ValueError("step_size must be greater than 0.")
    if horizon <= 0:
        raise ValueError("horizon must be greater than 0.")
    if K <= 0:
        raise ValueError("K must be greater than 0.")
    if max_window_size is not None and max_window_size <= 0:
        raise ValueError("max_window_size must be greater than 0 if specified.")

    total_steps = initial_window + (K - 1) * step_size + horizon
    if total_steps > T:  # pragma: no cover
        # Adjust the configuration to sensible defaults
        initial_window, step_size, horizon, K = generate_holdout_val_defaults(jnp.zeros((1, T)))
        total_steps = initial_window + (K - 1) * step_size + horizon
        if total_steps > T:
            raise ValueError(
                "Cannot generate a valid holdout configuration. Please adjust the parameters manually.",
            )

    if max_window_size is not None and max_window_size < horizon:
        max_window_size = horizon

    # Check for non-overlapping folds
    end_index = initial_window
    for _ in range(K - 1):
        start_index = end_index
        end_index = start_index + step_size
        if end_index > T:
            raise ValueError(  # pragma: no cover
                "The holdout folds are overlapping. Please adjust the step_size or reduce the number of folds (K).",
            )

    return initial_window, step_size, horizon, K, max_window_size
