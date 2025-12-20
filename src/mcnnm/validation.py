import jax
import jax.numpy as jnp

from .core import (
    compute_objective_value,
    fit,
    initialize_fixed_effects_and_H,
    initialize_matrices,
)
from .types import Array, Scalar
from .utils import extract_shortest_path, generate_lambda_grid, propose_lambda_values


def cross_validate(
    Y: Array,
    X: Array | None,
    Z: Array | None,
    V: Array | None,
    W: Array,
    Omega_inv: Array | None,
    use_unit_fe: bool,
    use_time_fe: bool,
    num_lam: int,
    max_iter: int | None = 1000,
    tol: float | None = 1e-5,
    cv_ratio: float | None = 0.8,
    K: int | None = 5,
) -> tuple[Array, Array, Array, Array]:
    """
    Perform K-fold cross-validation to select the best regularization parameters for the model.

    This function splits the data into K folds, trains the model on K-1 folds, and evaluates it
    on the remaining fold. The process is repeated for each fold and for different combinations
    of regularization parameters (lambda_L and lambda_H) specified in the lambda grid. The best
    lambda values are selected based on the minimum average root mean squared error (RMSE) across
    all folds.

    Steps:
        1. Create K-fold masks using the `create_folds` function, which randomly assigns observations to folds based on\
        the `cv_ratio`.
        2. Initialize the low-rank matrix L and the augmented covariate matrices X_tilde, Z_tilde, and V
           using the `initialize_matrices` function.
        3. Initialize the model parameters (gamma, delta, beta, H_tilde) and compute the maximum lambda
           values for each fold using the `initialize_fold` function and `jax.vmap`.
        4. Determine the overall maximum lambda_L and lambda_H values across all folds.
        5. Generate lambda_L and lambda_H value ranges using the `propose_lambda_values` function.
        6. Create a lambda grid by combining the lambda_L and lambda_H value ranges using the
           `generate_lambda_grid` function.
        7. Define the `fold_loss` function that computes the validation RMSE for each lambda combination
           within a fold:
           - Split the data into training and validation sets based on the fold mask.
           - Use `jax.lax.scan` to iterate over the lambda grid and compute the RMSE for each combination.
           - Train the model using the `fit` function on the training set for each lambda combination.
           - Compute the validation RMSE using the `compute_objective_value` function on the validation set.
           - Return the validation RMSE for each lambda combination.
        8. Apply the `fold_loss` function to each fold using `jax.vmap` to compute the validation RMSE for
           each lambda combination across all folds.
        9. Compute the average validation RMSE for each lambda combination across all folds.
        10. Select the best lambda_L and lambda_H values based on the minimum average RMSE.
        11. Determine the optimal lambda_L and lambda_H ranges by slicing the corresponding value ranges
            based on the best lambda values.
        12. Return the best lambda_L and lambda_H values along with their optimal ranges.

    Args:
        Y (Array): The target variable matrix of shape (N, T).
        X (Array): The feature matrix for unit-specific covariates of shape (N, P).
        Z (Array): The feature matrix for time-specific covariates of shape (T, Q).
        V (Array): The feature matrix for unit-time covariates of shape (N, T, R).
        W (Array): The binary matrix indicating the presence of observations of shape (N, T).
        Omega_inv (Array, optional): The inverse of the covariance matrix of shape (T, T). If not provided,
            the identity matrix is used.
        use_unit_fe (bool): Whether to include unit fixed effects in the model.
        use_time_fe (bool): Whether to include time fixed effects in the model.
        num_lam (int): The number of lambda values to include in the lambda grid.
        max_iter (int, optional): The maximum number of iterations for model fitting. Default is 1000.
        tol (float, optional): The tolerance for convergence in model fitting. Default is 1e-5.
        cv_ratio (float, optional): The ratio of data to use for training in each fold. Default is 0.8.
        K (int, optional): The number of folds for cross-validation. Default is 5.

    Returns:
        Tuple[Array, Array, Array, Array]: A tuple containing the following elements:
            - best_lambda_L (Array): The best lambda_L value based on the minimum average RMSE.
            - best_lambda_H (Array): The best lambda_H value based on the minimum average RMSE.
            - lambda_L_opt_range (Array): The optimal lambda_L range.
            - lambda_H_opt_range (Array): The optimal lambda_H range.

    Raises:
        ValueError: If the input arrays have inconsistent shapes.

    Note:
        - The function assumes that the input arrays are of type `jax.numpy.ndarray`.
        - The function uses `jax.vmap` and `jax.lax.scan` for parallelization and efficient computation.
        - The function initializes the model parameters using the `initialize_matrices` and
          `initialize_fixed_effects_and_H` functions.
        - The function generates a lambda grid using the `propose_lambda_values` and `generate_lambda_grid`
          functions.
        - The function computes the RMSE for each fold and lambda combination using the `fit` function.
        - The function selects the best lambda values based on the minimum average RMSE across all folds.

    """
    N, T = Y.shape

    # Ensure cv_ratio has a default value
    cv_ratio_val = cv_ratio if cv_ratio is not None else 0.8

    def create_folds(key):
        def create_fold_mask(key):
            return jax.random.bernoulli(key, cv_ratio_val, shape=(N, T))

        # Ensure K is not None
        k_folds = K if K is not None else 5
        keys = jax.random.split(key, k_folds)
        fold_masks = jax.vmap(create_fold_mask)(keys)
        return fold_masks * W

    fold_masks = create_folds(jax.random.PRNGKey(2024))
    L, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    def initialize_fold(fold_mask):
        Y_train = Y * fold_mask
        W_train = W * fold_mask

        (
            gamma_init,
            delta_init,
            beta_init,
            H_tilde_init,
            T_mat_init,
            in_prod_T_init,
            in_prod_init,
            lambda_L_max,
            lambda_H_max,
        ) = initialize_fixed_effects_and_H(
            Y_train,
            L,
            X_tilde,
            Z_tilde,
            V,
            W_train,
            use_unit_fe,
            use_time_fe,
            verbose=False,
        )
        return (
            gamma_init,
            delta_init,
            beta_init,
            H_tilde_init,
            T_mat_init,
            in_prod_T_init,
            in_prod_init,
            lambda_L_max,
            lambda_H_max,
            fold_mask,
        )

    fold_configs = jax.vmap(initialize_fold)(fold_masks)

    max_lambda_L = jnp.max(fold_configs[7])
    max_lambda_H = jnp.max(fold_configs[8])

    lambda_L_values = propose_lambda_values(max_lambda=max_lambda_L, n_lambdas=num_lam)
    lambda_H_values = propose_lambda_values(max_lambda=max_lambda_H, n_lambdas=num_lam)

    lambda_grid = generate_lambda_grid(lambda_L_values, lambda_H_values)

    def fold_loss(
        gamma_init,
        delta_init,
        beta_init,
        H_tilde_init,
        T_mat_init,
        in_prod_T_init,
        in_prod_init,
        lambda_L_max,
        lambda_H_max,
        holdout_mask,
    ):
        Y_train = Y * holdout_mask
        W_train = W * holdout_mask

        Y_val = Y * (1 - holdout_mask)
        W_val = W * (1 - holdout_mask)

        def compute_rmse(carry, lambda_L_H):
            lambda_L, lambda_H = lambda_L_H
            L, H_tilde_init, in_prod_init, gamma_init, delta_init, beta_init = carry
            H_new, L_new, gamma_new, delta_new, beta_new, in_prod_new, loss = fit(
                Y=Y_train,
                X_tilde=X_tilde,
                Z_tilde=Z_tilde,
                V=V,
                H_tilde=H_tilde_init,
                T_mat=T_mat_init,
                in_prod=in_prod_init,
                in_prod_T=in_prod_T_init,
                W=W_train,
                L=L,
                gamma=gamma_init,
                delta=delta_init,
                beta=beta_init,
                lambda_L=lambda_L,
                lambda_H=lambda_H,
                use_unit_fe=use_unit_fe,
                use_time_fe=use_time_fe,
                Omega_inv=Omega_inv,
                niter=max_iter,
                rel_tol=tol,
            )

            # get sum of singular values of L_new
            _, singular_values, _ = jnp.linalg.svd(L, full_matrices=False)
            sum_sigma = jnp.sum(singular_values)

            rmse = compute_objective_value(
                Y=Y_val,
                X_tilde=X_tilde,
                Z_tilde=Z_tilde,
                V=V,
                H_tilde=H_new,
                W=W_val,
                L=L_new,
                gamma=gamma_new,
                delta=delta_new,
                beta=beta_new,
                sum_sing_vals=sum_sigma,
                lambda_L=lambda_L,
                lambda_H=lambda_H,
                use_unit_fe=use_unit_fe,
                use_time_fe=use_time_fe,
                inv_omega=Omega_inv,
            )

            def valid_rmse(rmse):
                return jnp.sqrt(rmse)

            def return_inf(rmse):
                return jnp.inf

            fold_val_rmse = jax.lax.cond(loss >= 0, valid_rmse, return_inf, rmse)
            new_carry = (L_new, H_new, in_prod_new, gamma_new, delta_new, beta_new)
            return new_carry, fold_val_rmse

        init_state = (L, H_tilde_init, in_prod_init, gamma_init, delta_init, beta_init)
        _, fold_val_rmses = jax.lax.scan(compute_rmse, init_state, lambda_grid)
        return fold_val_rmses

    # Unpack fold_configs explicitly to satisfy type checker
    (
        gamma_inits,
        delta_inits,
        beta_inits,
        H_tilde_inits,
        T_mat_inits,
        in_prod_T_inits,
        in_prod_inits,
        lambda_L_maxs,
        lambda_H_maxs,
        holdout_masks,
    ) = fold_configs
    fold_rmses = jax.vmap(fold_loss, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
        gamma_inits,
        delta_inits,
        beta_inits,
        H_tilde_inits,
        T_mat_inits,
        in_prod_T_inits,
        in_prod_inits,
        lambda_L_maxs,
        lambda_H_maxs,
        holdout_masks,
    )
    mean_rmses = jnp.mean(
        fold_rmses,
        axis=0,
    )  # validation RMSE for each lambda pair averaged across all folds
    min_index = jnp.argmin(mean_rmses)  # index of the lambda pair with the lowest average RMSE

    # Get the corresponding lambda values from fold_configs
    best_lambda_L, best_lambda_H = lambda_grid[min_index]

    # Slice lambda_L_values and lambda_H_values
    lambda_L_opt_range = lambda_L_values[lambda_L_values >= best_lambda_L - 1e-8]
    lambda_H_opt_range = lambda_H_values[lambda_H_values >= best_lambda_H - 1e-8]

    return best_lambda_L, best_lambda_H, lambda_L_opt_range, lambda_H_opt_range


def holdout_validate(
    Y: Array,
    X: Array | None,
    Z: Array | None,
    V: Array | None,
    W: Array,
    Omega_inv: Array | None,
    use_unit_fe: bool,
    use_time_fe: bool,
    num_lam: int,
    initial_window: int,
    step_size: int,
    horizon: int,
    K: int,
    max_window_size: int | None = None,
    max_iter: int | None = 1000,
    tol: float | None = 1e-5,
) -> tuple[Array, Array, Array, Array]:
    """
    Perform holdout validation to select the optimal regularization parameters for the MC-NNM model.

    This function splits the data into K holdout folds along the time dimension, initializes the model
    configurations for each fold based on the observed data within the specified time window, and computes
    the holdout loss and RMSE for each fold and lambda pair. The lambda pair that yields the lowest average
    RMSE across all folds is selected as the optimal regularization parameters.

    Steps:
        1. Create K holdout masks using the `create_holdout_masks` function, which generates masks based on the \
        specified time windows determined by `initial_window`, `step_size`, and `horizon`.
        2. Initialize the low-rank matrix L and the augmented covariate matrices X_tilde, Z_tilde, and V
           using the `initialize_matrices` function.
        3. Initialize the model configurations (gamma, delta, beta, H_tilde) and compute the maximum lambda
           values for each holdout fold using the `initialize_holdout` function and `jax.vmap`.
        4. Determine the overall maximum lambda_L and lambda_H values across all holdout folds.
        5. Generate lambda_L and lambda_H value ranges using the `propose_lambda_values` function.
        6. Create a lambda grid by combining the lambda_L and lambda_H value ranges using the
           `generate_lambda_grid` function.
        7. Define the `holdout_fold_loss` function that computes the holdout RMSE for each lambda combination
           within a fold:
           - Split the data into training and validation sets based on the holdout mask.
           - Use `jax.lax.scan` to iterate over the lambda grid and compute the RMSE for each combination.
           - Train the model using the `fit` function on the training set for each lambda combination.
           - Compute the holdout RMSE using the `compute_objective_value` function on the validation set.
           - Return the holdout RMSE for each lambda combination.
        8. Apply the `holdout_fold_loss` function to each holdout fold using `jax.vmap` to compute the holdout
           RMSE for each lambda combination across all folds.
        9. Compute the average holdout RMSE for each lambda combination across all folds.
        10. Select the best lambda_L and lambda_H values based on the minimum average RMSE.
        11. Return the best lambda_L and lambda_H values along with the maximum lambda_L and lambda_H values.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X (Array): The unit-specific covariates matrix of shape (N, P).
        Z (Array): The time-specific covariates matrix of shape (T, Q).
        V (Array): The unit-time specific covariates tensor of shape (N, T, J).
        W (Array): The binary matrix indicating observed (0) and missing (1) entries in Y, shape (N, T).
        Omega_inv (Array, optional): The autocorrelation matrix of shape (T, T). If not provided, an identity matrix \
        is used.
        use_unit_fe (bool): Whether to use unit fixed effects.
        use_time_fe (bool): Whether to use time fixed effects.
        num_lam (int): The number of lambda values to generate in the grid.
        initial_window (int): The size of the initial time window for holdout validation. It determines the \
        number of time steps used to initialize the model configurations for each holdout fold.
        step_size (int): The step size for moving the time window in each holdout fold. It determines the \
        number of time steps to shift the window for each subsequent fold.
        horizon (int): The size of the holdout horizon (number of time steps to predict). It determines the number of \
        time steps used for evaluating the model's performance in each holdout fold.
        K (int): The number of holdout folds.
        max_window_size (int, optional): The maximum size of the time window. If specified, it limits the size of the \
        time window used for initializing the model configurations in each holdout fold. The window size will not \
        exceed `max_window_size` even if `initial_window` + `horizon` is larger. Defaults to None, meaning no limit on \
        the window size.
        max_iter (int, optional): Maximum number of iterations for fitting the model. Defaults to 1000.
        tol (float, optional): Convergence tolerance for fitting the model. Defaults to 1e-5.

    Returns:
        Tuple[Array, Array, Array, Array]: A tuple containing the following elements:
            - best_lambda_L (Array): The best lambda_L value based on the minimum average RMSE.
            - best_lambda_H (Array): The best lambda_H value based on the minimum average RMSE.
            - max_lambda_L (Array): The maximum lambda_L value across all holdout folds.
            - max_lambda_H (Array): The maximum lambda_H value across all holdout folds.

    Raises:
        ValueError: If the input arrays have inconsistent shapes or if the time window parameters are invalid.

    Note:
        - The binary matrix W indicates observed (0) and missing (1) entries in Y. The missing entries (1)
          are relevant for the loss computation.
        - The function uses JAX's vmap and scan operations to efficiently compute the holdout losses and
          RMSEs for multiple holdout folds and lambda pairs in parallel.
        - The function initializes the model configurations using the `initialize_matrices` and
          `initialize_fixed_effects_and_H` functions.
        - The function generates a lambda grid using the `propose_lambda_values` and `generate_lambda_grid`
          functions.
        - The function computes the RMSE for each holdout fold and lambda combination using the `fit` function.
        - The function selects the best lambda values based on the minimum average RMSE across all holdout folds.

    """
    N, T = Y.shape

    def create_holdout_masks(W, initial_window, step_size, horizon, K, max_window_size):
        N, T = W.shape
        masks = []
        start_index = initial_window
        for _ in range(K):
            end_index = min(start_index + horizon, T)
            if max_window_size is not None:
                train_start = max(end_index - max_window_size, 0)
            else:
                train_start = 0

            mask = jnp.zeros((N, T), dtype=bool)
            mask = mask.at[:, train_start:start_index].set(True)  # Training period

            n_train = jnp.sum(W * mask)
            if n_train > 0:
                masks.append(mask)

            start_index += step_size
            if end_index >= T:
                break

        if not masks:  # pragma: no cover
            # If no valid masks were created, create a single mask using all available data
            mask = jnp.ones((N, T), dtype=bool)  # pragma: no cover
            masks.append(mask)  # pragma: no cover

        return jnp.array(masks)

    holdout_masks = create_holdout_masks(W, initial_window, step_size, horizon, K, max_window_size)
    if holdout_masks.shape[0] < K:
        print("Warning: Not enough data for holdout validation. Using fewer folds.")
    if holdout_masks.shape[0] == 0:  # pragma: no cover
        print("Error: No data available for holdout validation. Exiting.")  # pragma: no  cover
        return (
            jnp.array(jnp.nan),
            jnp.array(jnp.nan),
            jnp.array(jnp.nan),
            jnp.array(jnp.nan),
        )  # pragma: no  cover
    L, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    def initialize_holdout(holdout_mask):
        Y_train = Y * holdout_mask
        W_train = W * holdout_mask

        (
            gamma_init,
            delta_init,
            beta_init,
            H_tilde_init,
            T_mat_init,
            in_prod_T_init,
            in_prod_init,
            lambda_L_max,
            lambda_H_max,
        ) = initialize_fixed_effects_and_H(
            Y_train,
            L,
            X_tilde,
            Z_tilde,
            V,
            W_train,
            use_unit_fe,
            use_time_fe,
            verbose=False,
        )
        return (
            gamma_init,
            delta_init,
            beta_init,
            H_tilde_init,
            T_mat_init,
            in_prod_T_init,
            in_prod_init,
            lambda_L_max,
            lambda_H_max,
            holdout_mask,
        )

    holdout_configs = jax.vmap(initialize_holdout)(holdout_masks)

    max_lambda_L = jnp.max(holdout_configs[7])
    max_lambda_H = jnp.max(holdout_configs[8])

    lambda_L_values = propose_lambda_values(max_lambda=max_lambda_L, n_lambdas=num_lam)
    lambda_H_values = propose_lambda_values(max_lambda=max_lambda_H, n_lambdas=num_lam)

    lambda_grid = generate_lambda_grid(lambda_L_values, lambda_H_values)

    def holdout_fold_loss(
        gamma_init,
        delta_init,
        beta_init,
        H_tilde_init,
        T_mat_init,
        in_prod_T_init,
        in_prod_init,
        lambda_L_max,
        lambda_H_max,
        holdout_mask,
    ):
        Y_train = Y * holdout_mask
        W_train = W * holdout_mask

        Y_val = Y * (1 - holdout_mask)
        W_val = W * (1 - holdout_mask)

        def compute_holdout_rmse(carry, lambda_L_H):
            lambda_L, lambda_H = lambda_L_H
            L, H_tilde_init, in_prod_init, gamma_init, delta_init, beta_init = carry
            H_new, L_new, gamma_new, delta_new, beta_new, in_prod_new, loss = fit(
                Y=Y_train,
                X_tilde=X_tilde,
                Z_tilde=Z_tilde,
                V=V,
                H_tilde=H_tilde_init,
                T_mat=T_mat_init,
                in_prod=in_prod_init,
                in_prod_T=in_prod_T_init,
                W=W_train,
                L=L,
                gamma=gamma_init,
                delta=delta_init,
                beta=beta_init,
                lambda_L=lambda_L,
                lambda_H=lambda_H,
                use_unit_fe=use_unit_fe,
                use_time_fe=use_time_fe,
                Omega_inv=Omega_inv,
                niter=max_iter,
                rel_tol=tol,
            )

            # get sum of singular values of L_new
            _, singular_values, _ = jnp.linalg.svd(L, full_matrices=False)
            sum_sigma = jnp.sum(singular_values)

            rmse = compute_objective_value(
                Y=Y_val,
                X_tilde=X_tilde,
                Z_tilde=Z_tilde,
                V=V,
                H_tilde=H_new,
                W=W_val,
                L=L_new,
                gamma=gamma_new,
                delta=delta_new,
                beta=beta_new,
                sum_sing_vals=sum_sigma,
                lambda_L=lambda_L,
                lambda_H=lambda_H,
                use_unit_fe=use_unit_fe,
                use_time_fe=use_time_fe,
                inv_omega=Omega_inv,
            )

            def valid_rmse(rmse):
                return jnp.sqrt(rmse)

            def return_inf(rmse):
                return jnp.inf

            fold_val_rmse = jax.lax.cond(loss >= 0, valid_rmse, return_inf, rmse)
            new_carry = (L_new, H_new, in_prod_new, gamma_new, delta_new, beta_new)
            return new_carry, fold_val_rmse

        init_state = (L, H_tilde_init, in_prod_init, gamma_init, delta_init, beta_init)
        _, fold_val_rmses = jax.lax.scan(compute_holdout_rmse, init_state, lambda_grid)
        return fold_val_rmses

    # Unpack holdout_configs explicitly to satisfy type checker
    (
        gamma_inits,
        delta_inits,
        beta_inits,
        H_tilde_inits,
        T_mat_inits,
        in_prod_T_inits,
        in_prod_inits,
        lambda_L_maxs,
        lambda_H_maxs,
        holdout_masks,
    ) = holdout_configs
    holdout_rmses = jax.vmap(holdout_fold_loss, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
        gamma_inits,
        delta_inits,
        beta_inits,
        H_tilde_inits,
        T_mat_inits,
        in_prod_T_inits,
        in_prod_inits,
        lambda_L_maxs,
        lambda_H_maxs,
        holdout_masks,
    )
    mean_rmses = jnp.mean(
        holdout_rmses,
        axis=0,
    )  # validation RMSE for each lambda pair averaged across all folds
    min_index = jnp.argmin(mean_rmses)  # index of the lambda pair with the lowest average RMSE

    # Get the corresponding lambda values from fold_configs
    best_lambda_L, best_lambda_H = lambda_grid[min_index]

    # Slice lambda_L_values and lambda_H_values
    lambda_L_opt_range = lambda_L_values[lambda_L_values >= best_lambda_L - 1e-8]
    lambda_H_opt_range = lambda_H_values[lambda_H_values >= best_lambda_H - 1e-8]

    return best_lambda_L, best_lambda_H, lambda_L_opt_range, lambda_H_opt_range


def final_fit(
    Y: Array,
    X: Array | None,
    Z: Array | None,
    V: Array | None,
    W: Array,
    Omega_inv: Array | None,
    use_unit_fe: bool,
    use_time_fe: bool,
    best_lambda_L,
    best_lambda_H,
    lambda_L_opt_range,
    lambda_H_opt_range,
    max_iter: int | None = 1000,
    tol: float | None = 1e-5,
) -> tuple[Array, Array, Array, Array, Array, Array, Scalar]:
    """
    Perform the final fit of the MC-NNM model using the optimal regularization parameters.

    This function fits the model using a sequence of lambda values, starting from the largest
    values in the optimal ranges and moving towards the best lambda values. This approach,
    known as warm-starting, helps to improve convergence and stability of the final fit.

    This function is similar to NNM_H, with one key difference. Instead of fitting all models
    on the grid described by lambda_Ls and lambda_Hs, it only considers the shortest path from
    the point on the grid with highest lambda_L and lambda_H to the point on the grid with
    smallest values of lambda_L and lambda_H. The key benefit of using this function is that,
    for chosen values of lambda_L and lambda_H, training can be much faster as the number of
    trained models is M+N-1 compared to M*N, where M is the length of lambda_L and N is the
    length of lambda_H.

    Steps:
        1. Initialize matrices and model parameters using the `initialize_matrices` and
           `initialize_fixed_effects_and_H` functions.
        2. Generate a lambda grid using the optimal lambda ranges.
        3. Extract the shortest path of lambda values from the largest to the best values.
        4. Iteratively fit the model for each lambda pair in the shortest path, using the
           results from the previous fit as initial values for the next fit.
        5. Return the final model parameters and loss.

    Args:
        Y (Array): The target variable matrix of shape (N, T).
        X (Array): The feature matrix for unit-specific covariates of shape (N, P).
        Z (Array): The feature matrix for time-specific covariates of shape (T, Q).
        V (Array): The feature matrix for unit-time covariates of shape (N, T, R).
        W (Array): The binary matrix indicating the presence of observations of shape (N, T).
        Omega_inv (Array, optional): The inverse of the covariance matrix of shape (T, T). If not provided, an\
         identity matrix is used.
        use_unit_fe (bool): Whether to include unit fixed effects in the model.
        use_time_fe (bool): Whether to include time fixed effects in the model.
        best_lambda_L: The best lambda_L value determined by cross-validation or holdout validation.
        best_lambda_H: The best lambda_H value determined by cross-validation or holdout validation.
        lambda_L_opt_range (Array): The optimal range of lambda_L values.
        lambda_H_opt_range (Array): The optimal range of lambda_H values.
        max_iter (int, optional): The maximum number of iterations for model fitting. Default is 1000.
        tol (float, optional): The tolerance for convergence in model fitting. Default is 1e-5.

    Returns:
        Tuple[Array, Array, Array, Array, Array, Array, float]: A tuple containing the following elements:
            - L_final (Array): The final low-rank matrix.
            - H_final (Array): The final interactive fixed effects matrix.
            - in_prod_final (Array): The final in-product matrix.
            - gamma_final (Array): The final unit fixed effects.
            - delta_final (Array): The final time fixed effects.
            - beta_final (Array): The final coefficients for covariates.
            - loss_final (float): The final loss value.

    Note:
        - The function uses the `fit` function to perform the model fitting for each lambda pair.
        - The function uses `jax.lax.scan` for efficient iteration over the lambda pairs.
        - The warm-starting approach helps to improve the stability and convergence of the final fit.

    """
    # Initialize matrices
    L_init, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    # Initialize coefficients
    (
        gamma_init,
        delta_init,
        beta_init,
        H_tilde_init,
        T_mat_init,
        in_prod_T_init,
        in_prod_init,
        lambda_L_max,
        lambda_H_max,
    ) = initialize_fixed_effects_and_H(
        Y,
        L_init,
        X_tilde,
        Z_tilde,
        V,
        W,
        use_unit_fe,
        use_time_fe,
        verbose=False,
    )

    lambda_grid = generate_lambda_grid(lambda_L_opt_range, lambda_H_opt_range)
    shortest_path = extract_shortest_path(lambda_grid)

    def fit_sequence(carry, lambda_L_H):
        lambda_L, lambda_H = lambda_L_H
        (
            L_previous,
            H_tilde_previous,
            in_prod_previous,
            gamma_previous,
            delta_previous,
            beta_previous,
        ) = carry
        H_new, L_new, gamma_new, delta_new, beta_new, in_prod_new, loss = fit(
            Y=Y,
            X_tilde=X_tilde,
            Z_tilde=Z_tilde,
            V=V,
            H_tilde=H_tilde_previous,
            T_mat=T_mat_init,
            in_prod=in_prod_previous,
            in_prod_T=in_prod_T_init,
            W=W,
            L=L_previous,
            gamma=gamma_previous,
            delta=delta_previous,
            beta=beta_previous,
            lambda_L=lambda_L,
            lambda_H=lambda_H,
            use_unit_fe=use_unit_fe,
            use_time_fe=use_time_fe,
            Omega_inv=Omega_inv,
            niter=max_iter,
            rel_tol=tol,
        )

        new_carry = (L_new, H_new, in_prod_new, gamma_new, delta_new, beta_new)

        return new_carry, loss

    init_state = (L_init, H_tilde_init, in_prod_init, gamma_init, delta_init, beta_init)
    results, loss_sequence = jax.lax.scan(fit_sequence, init_state, shortest_path)

    # unpack results for last lambda pair in shortest path
    L_final, H_final, in_prod_final, gamma_final, delta_final, beta_final = results
    loss_final = loss_sequence[-1]
    return L_final, H_final, in_prod_final, gamma_final, delta_final, beta_final, loss_final
