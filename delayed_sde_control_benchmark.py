"""
delayed_sde_control_benchmark.py

A Reproducible Numerical Study and Differentiable Monte Carlo Benchmark 
for a Time-Delayed Stochastic First-Order Dynamical System with Discrete PD Control.
"""

from typing import NamedTuple, Tuple, Dict, Any, List
import time
import jax

# Enable 64-bit double precision for scientific-grade numerical differentiation
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt


# ==============================================================================
# 1. PARAMETER DATA STRUCTURES
# ==============================================================================

class ContinuousModelParams(NamedTuple):
    gamma: float        # Natural dissipation/relaxation rate (1/s)
    sigma: float        # Brownian diffusion/disturbance intensity
    target: float       # Constant reference state setpoint x*
    q_cost: float       # Quadratic penalty weight on state tracking error
    r_cost: float       # Quadratic penalty weight on commanded control effort


class DiscretizationParams(NamedTuple):
    dt: float           # Discrete integration time step (s)
    tau: float          # Physical transport delay (s)
    delay_steps: int    # Discrete buffer length D = tau / dt (integer invariant)
    horizon_steps: int  # Total simulation horizon steps
    tail_fraction: float  # Final fraction of horizon used to compute tail-window mean stage cost


class OptimizationConfig(NamedTuple):
    learning_rate: float
    max_iterations: int
    mc_batch_size: int
    tol_projected_grad_norm: float  # Empirical stationarity threshold: ||G_eta(theta)||_2 < tol
    min_param_bounds: jnp.ndarray   # Lower-bound box constraint [Kp_min, Kd_min]
    eps_safeguard: float            # Zero-division numerical safeguard


class NumericalValidationConfig(NamedTuple):
    fd_perturbation_h_list: List[float]
    fd_acceptance_tol: float
    test_thetas: List[Tuple[str, jnp.ndarray]]
    mc_sample_sizes: List[int]
    mc_target_relative_ci_halfwidth_pct: float
    z_score_95: float
    dt_candidates: List[float]
    dt_ref: float
    dt_study_mc_samples: int
    invariant_eps: float
    out_of_sample_eval_n: int
    num_optimization_runs: int
    stability_x0_list: List[float]
    long_horizon_multiplier: int


class VisualizationConfig(NamedTuple):
    grid_resolution: int
    kp_range: Tuple[float, float]
    kd_range: Tuple[float, float]
    dpi: int


class SystemState(NamedTuple):
    x: float
    prev_e: float
    u_buffer: jnp.ndarray


# ==============================================================================
# 2. PRE-FLIGHT MATHEMATICAL & INVARIANT VALIDATION
# ==============================================================================

def validate_system_invariants(
    m: ContinuousModelParams,
    d: DiscretizationParams,
    opt: OptimizationConfig,
    val: NumericalValidationConfig
) -> None:
    assert m.gamma > 0.0, "Model Error: gamma must be strictly positive."
    assert m.sigma >= 0.0, "Model Error: sigma must be non-negative."
    assert m.q_cost >= 0.0 and m.r_cost >= 0.0, "Model Error: cost weights must be non-negative."
    assert d.dt > 0.0, "Discretization Error: dt must be strictly positive."
    assert d.tau > 0.0, "Discretization Error: tau must be strictly positive."
    assert d.delay_steps >= 1, "Discretization Error: delay_steps must be at least 1."
    assert abs(d.delay_steps * d.dt - d.tau) < val.invariant_eps, \
        f"Invariant Violation: tau ({d.tau}) != delay_steps * dt ({d.delay_steps * d.dt})"
    assert 0.0 < d.tail_fraction <= 1.0, "Discretization Error: tail_fraction must reside in (0, 1]."
    assert opt.learning_rate > 0.0, "Optimization Error: learning_rate must be strictly positive."
    assert jnp.all(opt.min_param_bounds >= 0.0), "Optimization Error: parameter bounds must be non-negative."


# ==============================================================================
# 3. STEP INTEGRATOR (Euler-Maruyama + Discrete-Time PD)
# ==============================================================================

def make_step_fn(m: ContinuousModelParams, d: DiscretizationParams):
    def step(state: SystemState, dW_k: float, theta: jnp.ndarray):
        kp, kd = theta[0], theta[1]
        
        e = state.x - m.target
        de_dt = (e - state.prev_e) / d.dt
        u_commanded = -(kp * e + kd * de_dt)
        
        u_delayed = state.u_buffer[0]
        next_u_buffer = jnp.roll(state.u_buffer, -1).at[-1].set(u_commanded)
        
        drift = (-m.gamma * state.x + u_delayed) * d.dt
        diffusion = m.sigma * dW_k
        next_x = state.x + drift + diffusion
        
        stage_cost = m.q_cost * (e ** 2) + m.r_cost * (u_commanded ** 2)
        
        next_state = SystemState(x=next_x, prev_e=e, u_buffer=next_u_buffer)
        return next_state, (next_x, u_commanded, stage_cost)
    return step


def simulate_trajectory_tail_cost(
    theta: jnp.ndarray, 
    dW_sequence: jnp.ndarray, 
    m: ContinuousModelParams, 
    d: DiscretizationParams,
    initial_x: float = 0.0
) -> Tuple[float, Tuple[jnp.ndarray, jnp.ndarray]]:
    step_fn = make_step_fn(m, d)
    
    init_state = SystemState(
        x=initial_x,
        prev_e=initial_x - m.target,
        u_buffer=jnp.zeros(d.delay_steps)
    )
    
    def scan_wrapper(carry_state, dW_k):
        return step_fn(carry_state, dW_k, theta)
    
    _, (x_hist, u_hist, cost_hist) = jax.lax.scan(
        scan_wrapper, init_state, dW_sequence, length=d.horizon_steps
    )
    
    tail_start = int((1.0 - d.tail_fraction) * d.horizon_steps)
    tail_mean_stage_cost = jnp.mean(cost_hist[tail_start:])
    return tail_mean_stage_cost, (x_hist, u_hist)


def compute_empirical_cost_batch(
    theta: jnp.ndarray, 
    batch_dW: jnp.ndarray, 
    m: ContinuousModelParams, 
    d: DiscretizationParams
) -> float:
    def single_eval(dW_seq):
        cost, _ = simulate_trajectory_tail_cost(theta, dW_seq, m, d)
        return cost
    return jnp.mean(jax.vmap(single_eval)(batch_dW))


# ==============================================================================
# 4. NUMERICAL AUDITS
# ==============================================================================

def verify_gradient_multi_regime(
    test_thetas: List[Tuple[str, jnp.ndarray]],
    h_list: List[float],
    batch_dW: jnp.ndarray, 
    m: ContinuousModelParams, 
    d: DiscretizationParams, 
    v_cfg: NumericalValidationConfig,
    opt_cfg: OptimizationConfig
) -> List[Dict[str, Any]]:
    grad_ad_fn = jax.grad(compute_empirical_cost_batch, argnums=0)
    results = []
    
    for regime_name, theta in test_thetas:
        grad_ad = grad_ad_fn(theta, batch_dW, m, d)
        
        for h in h_list:
            grad_fd = []
            for i in range(len(theta)):
                e_i = jnp.zeros_like(theta).at[i].set(1.0)
                c_plus = compute_empirical_cost_batch(theta + h * e_i, batch_dW, m, d)
                c_minus = compute_empirical_cost_batch(theta - h * e_i, batch_dW, m, d)
                grad_fd.append((c_plus - c_minus) / (2.0 * h))
            grad_fd = jnp.array(grad_fd)
            
            rel_error = jnp.linalg.norm(grad_ad - grad_fd) / (jnp.linalg.norm(grad_fd) + opt_cfg.eps_safeguard)
            results.append({
                'regime': regime_name,
                'theta': theta,
                'h': h,
                'grad_ad': grad_ad,
                'grad_fd': grad_fd,
                'relative_error': float(rel_error),
                'passed': bool(rel_error < v_cfg.fd_acceptance_tol)
            })
    return results


def study_mc_sample_nested_scaling(
    theta: jnp.ndarray,
    key: jax.random.PRNGKey,
    m: ContinuousModelParams,
    d: DiscretizationParams,
    v_cfg: NumericalValidationConfig
) -> Tuple[List[Dict[str, Any]], float]:
    n_max = max(v_cfg.mc_sample_sizes)
    master_dW_batch = jax.random.normal(key, shape=(n_max, d.horizon_steps)) * jnp.sqrt(d.dt)
    
    def single_eval(dW_seq):
        cost, _ = simulate_trajectory_tail_cost(theta, dW_seq, m, d)
        return cost
    all_individual_costs = jax.vmap(single_eval)(master_dW_batch)
    
    results = []
    log_n_list = []
    log_sem_list = []
    
    for n in v_cfg.mc_sample_sizes:
        sub_costs = all_individual_costs[:n]
        mean_val = float(jnp.mean(sub_costs))
        assert mean_val >= 0.0, "Cost estimator must be non-negative."
        
        sem_val = float(jnp.std(sub_costs, ddof=1) / jnp.sqrt(n)) if n > 1 else 0.0
        ci_half_width = v_cfg.z_score_95 * sem_val
        rel_margin_pct = (ci_half_width / (mean_val + v_cfg.invariant_eps)) * 100.0
        
        results.append({
            'n_samples': n,
            'mean_cost': mean_val,
            'sem': sem_val,
            'wald_ci95_lower': mean_val - ci_half_width,
            'wald_ci95_upper': mean_val + ci_half_width,
            'relative_margin_pct': rel_margin_pct,
            'meets_precision_target': bool(rel_margin_pct <= v_cfg.mc_target_relative_ci_halfwidth_pct)
        })
        
        if n > 1 and sem_val > 0.0:
            log_n_list.append(jnp.log(n))
            log_sem_list.append(jnp.log(sem_val))
            
    x = jnp.array(log_n_list)
    y = jnp.array(log_sem_list)
    x_mean, y_mean = jnp.mean(x), jnp.mean(y)
    empirical_slope = float(jnp.sum((x - x_mean) * (y - y_mean)) / jnp.sum((x - x_mean) ** 2))
    
    return results, empirical_slope


def study_timestep_discretization_error(
    theta: jnp.ndarray,
    key: jax.random.PRNGKey,
    m: ContinuousModelParams,
    tau: float,
    total_time: float,
    v_cfg: NumericalValidationConfig,
    tail_fraction: float
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dt_eval_list = sorted(v_cfg.dt_candidates, reverse=True)
    all_dts = sorted(list(set(dt_eval_list + [v_cfg.dt_ref])))
    dt_ref = v_cfg.dt_ref
    
    base_horizon = int(round(total_time / dt_ref))
    n_eval = v_cfg.dt_study_mc_samples
    
    dW_ref_batch = jax.random.normal(key, shape=(n_eval, base_horizon)) * jnp.sqrt(dt_ref)
    
    raw_evals = {}
    for dt in all_dts:
        delay_steps = int(round(tau / dt))
        assert abs(delay_steps * dt - tau) < v_cfg.invariant_eps
        ratio = int(round(dt / dt_ref))
        assert abs(ratio * dt_ref - dt) < v_cfg.invariant_eps
        
        horizon_steps = int(round(total_time / dt))
        d = DiscretizationParams(
            dt=dt,
            tau=tau,
            delay_steps=delay_steps,
            horizon_steps=horizon_steps,
            tail_fraction=tail_fraction
        )
        
        dW_coarse_batch = dW_ref_batch.reshape(n_eval, horizon_steps, ratio).sum(axis=-1)
        cost_val = float(compute_empirical_cost_batch(theta, dW_coarse_batch, m, d))
        
        raw_evals[dt] = {
            'delay_steps': delay_steps,
            'horizon_steps': horizon_steps,
            'cost': cost_val
        }
    
    ref_cost = raw_evals[dt_ref]['cost']
    results = []
    prev_error = None
    prev_dt = None
    
    for dt in dt_eval_list:
        cost_val = raw_evals[dt]['cost']
        abs_error = abs(cost_val - ref_cost)
        rel_error = abs_error / (ref_cost + v_cfg.invariant_eps)
        
        apparent_eoc = None
        if prev_error is not None and prev_error > 1e-12 and abs_error > 1e-12:
            apparent_eoc = float(jnp.log(prev_error / abs_error) / jnp.log(prev_dt / dt))
            
        results.append({
            'dt': dt,
            'delay_steps': raw_evals[dt]['delay_steps'],
            'horizon_steps': raw_evals[dt]['horizon_steps'],
            'cost': cost_val,
            'abs_error_vs_ref': abs_error,
            'rel_error_vs_ref': rel_error,
            'apparent_eoc': apparent_eoc
        })
        
        prev_error = abs_error
        prev_dt = dt
        
    return results, {'dt_ref': dt_ref, 'ref_cost': ref_cost}


# ==============================================================================
# 5. DISCRETE SPECTRAL STABILITY & STOCHASTIC DIAGNOSTICS
# ==============================================================================

def compute_discrete_closed_loop_spectral_radius(
    theta: jnp.ndarray,
    m: ContinuousModelParams,
    d: DiscretizationParams
) -> Tuple[float, jnp.ndarray]:
    kp, kd = theta[0], theta[1]
    D = d.delay_steps
    dim = D + 2
    
    M = jnp.zeros((dim, dim))
    
    # Row 0: x_{n+1} = (1 - gamma*dt)*x_n + dt*b_0
    M = M.at[0, 0].set(1.0 - m.gamma * d.dt)
    M = M.at[0, 2].set(d.dt)
    
    # Row 1: x_prev_{n+1} = x_n
    M = M.at[1, 0].set(1.0)
    
    # Rows 2 to D: FIFO delay buffer shift
    for k in range(D - 1):
        M = M.at[2 + k, 2 + k + 1].set(1.0)
        
    # Row D+1: b_{D-1, n+1} = u_{commanded, n}
    M = M.at[dim - 1, 0].set(-(kp + kd / d.dt))
    M = M.at[dim - 1, 1].set(kd / d.dt)
    
    eigenvalues = jnp.linalg.eigvals(M)
    spectral_radius = float(jnp.max(jnp.abs(eigenvalues)))
    return spectral_radius, eigenvalues


def run_closed_loop_stability_diagnostics(
    theta: jnp.ndarray,
    key: jax.random.PRNGKey,
    m: ContinuousModelParams,
    d: DiscretizationParams,
    v_cfg: NumericalValidationConfig
) -> Dict[str, Any]:
    total_time = d.horizon_steps * d.dt
    
    # 1. Exact Spectral Radius Analysis (Schur Stability)
    spectral_radius, _ = compute_discrete_closed_loop_spectral_radius(theta, m, d)
    is_schur_stable = bool(spectral_radius < 1.0)
    
    # 2. Deterministic Closed-Loop Transient Decay Proxy (sigma = 0)
    m_deterministic = ContinuousModelParams(
        gamma=m.gamma, sigma=0.0, target=m.target, q_cost=m.q_cost, r_cost=m.r_cost
    )
    dW_zero = jnp.zeros(d.horizon_steps)
    x0_test = 2.0
    _, (x_det, _) = simulate_trajectory_tail_cost(
        theta, dW_zero, m_deterministic, d, initial_x=x0_test
    )
    
    final_error_det = abs(float(x_det[-1]) - m.target)
    init_error_det = abs(x0_test - m.target)
    
    if final_error_det > 1e-12 and init_error_det > 1e-12:
        lambda_emp = float((1.0 / total_time) * jnp.log(final_error_det / init_error_det))
    else:
        lambda_emp = -float('inf')
        
    # 3. Displaced Initial Conditions Recovery Audit
    recovery_results = []
    for x0 in v_cfg.stability_x0_list:
        _, (x_rec, _) = simulate_trajectory_tail_cost(
            theta, dW_zero, m_deterministic, d, initial_x=x0
        )
        final_err = abs(float(x_rec[-1]) - m.target)
        max_excursion = float(jnp.max(jnp.abs(x_rec - m.target)))
        recovery_criterion_met = bool(final_err < 0.05)
        recovery_results.append({
            'x0': x0,
            'final_error': final_err,
            'max_excursion': max_excursion,
            'final_error_recovery_criterion_met': recovery_criterion_met
        })
        
    # 4. Stochastic Long-Horizon Boundedness Diagnostic
    long_horizon_steps = d.horizon_steps * v_cfg.long_horizon_multiplier
    d_long = DiscretizationParams(
        dt=d.dt, tau=d.tau, delay_steps=d.delay_steps, 
        horizon_steps=long_horizon_steps, tail_fraction=d.tail_fraction
    )
    key, subk = jax.random.split(key)
    dW_long_batch = jax.random.normal(subk, shape=(64, long_horizon_steps)) * jnp.sqrt(d.dt)
    
    def eval_long(dW_seq):
        _, (x_h, _) = simulate_trajectory_tail_cost(theta, dW_seq, m, d_long, initial_x=0.0)
        return x_h
        
    all_x_long = jax.vmap(eval_long)(dW_long_batch)
    
    mid_pt = long_horizon_steps // 2
    var_first_half = float(jnp.var(all_x_long[:, :mid_pt]))
    var_second_half = float(jnp.var(all_x_long[:, mid_pt:]))
    max_state_long = float(jnp.max(jnp.abs(all_x_long)))
    
    variance_ratio = var_second_half / (var_first_half + 1e-12)
    empirical_boundedness_indicator = bool(variance_ratio < 2.5 and max_state_long < 10.0)
    
    return {
        'spectral_radius_rho': spectral_radius,
        'is_schur_stable': is_schur_stable,
        'transient_decay_proxy_lambda': lambda_emp,
        'recovery_tests': recovery_results,
        'long_horizon_time_s': total_time * v_cfg.long_horizon_multiplier,
        'var_first_half': var_first_half,
        'var_second_half': var_second_half,
        'variance_ratio': variance_ratio,
        'max_state_long': max_state_long,
        'empirical_long_horizon_boundedness_indicator': empirical_boundedness_indicator
    }


# ==============================================================================
# 6. OPTIMIZER: STOCHASTIC PROJECTED GRADIENT DESCENT (SPGD)
# ==============================================================================

def run_stochastic_projected_gradient_descent(
    theta_init: jnp.ndarray,
    key: jax.random.PRNGKey,
    m: ContinuousModelParams,
    d: DiscretizationParams,
    opt_cfg: OptimizationConfig
) -> Tuple[jnp.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    val_and_grad_fn = jax.jit(
        jax.value_and_grad(compute_empirical_cost_batch, argnums=0),
        static_argnums=(2, 3)
    )
    
    eta = opt_cfg.learning_rate
    theta = theta_init
    history = []
    stop_reason = "max_iterations_reached"
    
    t_start = time.time()
    for it in range(opt_cfg.max_iterations):
        key, subkey = jax.random.split(key)
        batch_dW = jax.random.normal(subkey, shape=(opt_cfg.mc_batch_size, d.horizon_steps)) * jnp.sqrt(d.dt)
        
        cost_val, grad_raw = val_and_grad_fn(theta, batch_dW, m, d)
        
        unconstrained_step = theta - eta * grad_raw
        next_theta = jnp.maximum(unconstrained_step, opt_cfg.min_param_bounds)
        
        proj_grad_mapping = (theta - next_theta) / eta
        proj_grad_norm = float(jnp.linalg.norm(proj_grad_mapping))
        step_norm = float(jnp.linalg.norm(next_theta - theta))
        
        history.append({
            'iteration': it + 1,
            'theta': theta,
            'cost': float(cost_val),
            'proj_grad_norm': proj_grad_norm,
            'step_norm': step_norm
        })
        
        if proj_grad_norm < opt_cfg.tol_projected_grad_norm:
            stop_reason = (
                f"empirical_projected_stationarity_reached "
                f"(||G_eta||_2 = {proj_grad_norm:.4f} < {opt_cfg.tol_projected_grad_norm})"
            )
            theta = next_theta
            break
            
        theta = next_theta

    opt_metadata = {
        'elapsed_seconds': time.time() - t_start,
        'total_iterations': len(history),
        'stop_reason': stop_reason,
        'final_theta': theta
    }
    return theta, history, opt_metadata


# ==============================================================================
# 7. MAIN EXECUTION ROUTINE
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("NUMERICAL BENCHMARK: DELAYED SDE STOCHASTIC CONTROL (Double Precision)")
    print("=" * 80)

    val_cfg = NumericalValidationConfig(
        fd_perturbation_h_list=[1e-2, 1e-3, 1e-4, 1e-5],
        fd_acceptance_tol=1e-2,
        test_thetas=[
            ("Nominal Interior", jnp.array([2.5, 0.4])),
            ("Near-Boundary / Low-Gain", jnp.array([0.05, 0.01])),
            ("High-Gain / Delay-Sensitive", jnp.array([5.0, 0.8])),
        ],
        mc_sample_sizes=[16, 32, 64, 128, 256, 512, 1024],
        mc_target_relative_ci_halfwidth_pct=2.5,
        z_score_95=1.95996,
        dt_candidates=[0.025, 0.01, 0.005, 0.0025],
        dt_ref=0.00125,
        dt_study_mc_samples=128,
        invariant_eps=1e-10,
        out_of_sample_eval_n=1000,
        num_optimization_runs=5,
        stability_x0_list=[-3.0, -1.0, 0.0, 2.0, 4.0],
        long_horizon_multiplier=5
    )

    model_params = ContinuousModelParams(
        gamma=0.5,
        sigma=0.15,
        target=1.0,
        q_cost=1.0,
        r_cost=0.005
    )

    tau_physical = 0.05
    dt_base = 0.01
    delay_steps_base = int(round(tau_physical / dt_base))

    disc_params = DiscretizationParams(
        dt=dt_base,
        tau=tau_physical,
        delay_steps=delay_steps_base,
        horizon_steps=600,
        tail_fraction=0.50
    )

    opt_cfg = OptimizationConfig(
        learning_rate=0.08,
        max_iterations=50,
        mc_batch_size=64,
        tol_projected_grad_norm=0.08,
        min_param_bounds=jnp.array([1e-3, 1e-4]),
        eps_safeguard=1e-8
    )

    vis_cfg = VisualizationConfig(
        grid_resolution=35,
        kp_range=(0.5, 7.0),
        kd_range=(0.01, 1.0),
        dpi=200
    )

    validate_system_invariants(model_params, disc_params, opt_cfg, val_cfg)

    master_key = jax.random.PRNGKey(2026)
    k_grad, k_mc, k_dt, k_opt_base, k_oos, k_stab = jax.random.split(master_key, 6)

    # A. Gradient Verification
    print("\n[Audit 1/6] Gradient Verification (AD vs. Central FD across Regimes & Step Sizes)...")
    dW_verify = jax.random.normal(k_grad, shape=(16, disc_params.horizon_steps)) * jnp.sqrt(disc_params.dt)
    grad_checks = verify_gradient_multi_regime(
        val_cfg.test_thetas, val_cfg.fd_perturbation_h_list, dW_verify, 
        model_params, disc_params, val_cfg, opt_cfg
    )
    for chk in grad_checks:
        status_str = "PASSED" if chk['passed'] else "FAILED"
        print(f"      Regime: {chk['regime']:<28} | h={chk['h']:.0e} | Rel Error: {chk['relative_error']:.2e} [{status_str}]")
        assert chk['passed'], f"Gradient verification failed for {chk['regime']} at h={chk['h']}!"

    # B. Timestep Discretization Error
    print(f"\n[Audit 2/6] Timestep Discretization Error & Apparent EOC (vs. Reference dt_ref = {val_cfg.dt_ref:.5f} s)...")
    dt_results, ref_info = study_timestep_discretization_error(
        jnp.array([2.0, 0.2]), k_dt, model_params, tau_physical, 
        total_time=6.0, v_cfg=val_cfg, tail_fraction=disc_params.tail_fraction
    )
    print(f"      Reference Tail Stage-Cost at dt_ref ({val_cfg.dt_ref:.5f} s): {ref_info['ref_cost']:.5f}")
    for r in dt_results:
        eoc_str = f"Apparent EOC: {r['apparent_eoc']:.2f}" if r['apparent_eoc'] is not None else "Baseline Coarse Level"
        print(f"      dt={r['dt']:.4f} s (Buffer D={r['delay_steps']:2d}) -> Cost: {r['cost']:.5f} | "
              f"Abs Error vs Ref: {r['abs_error_vs_ref']:.5f} | {eoc_str}")

    # C. Monte Carlo Scaling
    print("\n[Audit 3/6] Nested-Prefix Monte Carlo Scaling & Slope Evaluation d(ln SEM)/d(ln N)...")
    mc_results, empirical_slope = study_mc_sample_nested_scaling(
        jnp.array([2.0, 0.2]), k_mc, model_params, disc_params, val_cfg
    )
    for r in mc_results:
        print(f"      N_MC={r['n_samples']:4d} -> Mean Cost: {r['mean_cost']:.5f} ± {r['sem']:.5f} | "
              f"Wald CI Half-Width: ±{r['relative_margin_pct']:.2f}% | Target (<={val_cfg.mc_target_relative_ci_halfwidth_pct}%): {r['meets_precision_target']}")
    print(f"      Empirical Log-Log Scaling Slope : {empirical_slope:.4f} (Expected CLT asymptotic scaling: -0.5000)")

    # D. Multi-Seed SPGD
    print(f"\n[Audit 4/6] Multi-Seed Robustness Evaluation ({val_cfg.num_optimization_runs} Independent Stochastic Runs)...")
    opt_seed_keys = jax.random.split(k_opt_base, val_cfg.num_optimization_runs)
    theta_0 = jnp.array([1.0, 0.05])
    
    multi_run_finals = []
    primary_history = None
    
    for run_idx, s_key in enumerate(opt_seed_keys):
        th_final, history, meta = run_stochastic_projected_gradient_descent(
            theta_0, s_key, model_params, disc_params, opt_cfg
        )
        multi_run_finals.append(th_final)
        if run_idx == 0:
            primary_history = history
        print(f"      Run {run_idx+1}: Final theta = [{th_final[0]:.3f}, {th_final[1]:.3f}] | "
              f"Iter: {meta['total_iterations']} | Stop: {meta['stop_reason']}")
        
    multi_run_finals = jnp.array(multi_run_finals)
    theta_mean = jnp.mean(multi_run_finals, axis=0)
    theta_dispersion_std = jnp.std(multi_run_finals, axis=0)
    print(f"      Multi-Seed Parameter Dispersion (Inter-Seed Spread): Kp = {theta_mean[0]:.3f} ± {theta_dispersion_std[0]:.3f}, Kd = {theta_mean[1]:.3f} ± {theta_dispersion_std[1]:.3f}")

    # E. Paired OOS Inference
    print(f"\n[Audit 5/6] Paired Out-of-Sample (OOS) Statistical Inference (N = {val_cfg.out_of_sample_eval_n} Unseen Paths)...")
    dW_oos = jax.random.normal(k_oos, shape=(val_cfg.out_of_sample_eval_n, disc_params.horizon_steps)) * jnp.sqrt(disc_params.dt)
    
    def eval_oos_batch(th):
        return jax.vmap(lambda dW_seq: simulate_trajectory_tail_cost(th, dW_seq, model_params, disc_params)[0])(dW_oos)
    
    costs_init = eval_oos_batch(theta_0)
    mean_c0 = float(jnp.mean(costs_init))
    sem_c0 = float(jnp.std(costs_init, ddof=1) / jnp.sqrt(val_cfg.out_of_sample_eval_n))
    print(f"      Initial Candidate theta_0 : OOS Mean Tail Stage-Cost = {mean_c0:.5f} ± {sem_c0:.5f} (SEM)")
    
    for s_idx, th_s in enumerate(multi_run_finals):
        c_s = eval_oos_batch(th_s)
        m_s = float(jnp.mean(c_s))
        s_s = float(jnp.std(c_s, ddof=1) / jnp.sqrt(val_cfg.out_of_sample_eval_n))
        
        paired_diff = costs_init - c_s
        mean_diff = float(jnp.mean(paired_diff))
        sem_diff = float(jnp.std(paired_diff, ddof=1) / jnp.sqrt(val_cfg.out_of_sample_eval_n))
        ci_diff_half = val_cfg.z_score_95 * sem_diff
        ci_lower = mean_diff - ci_diff_half
        ci_upper = mean_diff + ci_diff_half
        descriptive_reduct_pct = (mean_diff / mean_c0) * 100.0
        
        if ci_lower > 0.0:
            decision_str = "Statistically distinguishable improvement (p < 0.05)"
        elif ci_upper < 0.0:
            decision_str = "Statistically distinguishable degradation (p < 0.05)"
        else:
            decision_str = "No statistically significant difference detected (95% CI contains 0)"
            
        print(f"      Candidate Seed {s_idx+1} ({th_s[0]:.3f}, {th_s[1]:.3f}) : Mean Cost = {m_s:.5f} ± {s_s:.5f} | "
              f"Paired Mean Reduction = +{mean_diff:.5f} ± {sem_diff:.5f} (95% CI: [{ci_lower:.5f}, {ci_upper:.5f}]) | "
              f"Descriptive Reduction: {descriptive_reduct_pct:.2f}% | [{decision_str}]")

    # F. Spectral Stability & Stochastic Diagnostics
    print("\n[Audit 6/6] Exact Discrete Spectral Stability & Stochastic Diagnostics (Representative Seed 1)...")
    representative_candidate_theta = multi_run_finals[0]
    stab_diag = run_closed_loop_stability_diagnostics(
        representative_candidate_theta, k_stab, model_params, disc_params, val_cfg
    )
    print(f"      1. Discrete Companion Matrix Spectral Radius rho(M) : {stab_diag['spectral_radius_rho']:.4f} | Schur Stable (rho < 1): {stab_diag['is_schur_stable']}")
    print(f"      2. Deterministic Closed-Loop Decay Proxy lambda_emp   : {stab_diag['transient_decay_proxy_lambda']:.4f} 1/s")
    print(f"      3. Stochastic Long-Horizon Boundedness (T = {stab_diag['long_horizon_time_s']:.1f} s) : "
          f"Var(1st Half)={stab_diag['var_first_half']:.4f}, Var(2nd Half)={stab_diag['var_second_half']:.4f} | "
          f"Ratio={stab_diag['variance_ratio']:.2f} | Boundedness Indicator: {stab_diag['empirical_long_horizon_boundedness_indicator']}")
    print("      4. Displaced Initial Condition Recovery Audit (Deterministic):")
    for rec in stab_diag['recovery_tests']:
        print(f"         x(0) = {rec['x0']:+4.1f} -> Final Error = {rec['final_error']:.4f}, Max Excursion = {rec['max_excursion']:.4f} | Criterion Met: {rec['final_error_recovery_criterion_met']}")

    # G. Graphical Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    kp_vals = jnp.linspace(vis_cfg.kp_range[0], vis_cfg.kp_range[1], vis_cfg.grid_resolution)
    kd_vals = jnp.linspace(vis_cfg.kd_range[0], vis_cfg.kd_range[1], vis_cfg.grid_resolution)
    KP, KD = jnp.meshgrid(kp_vals, kd_vals)
    
    grid_dW = jax.random.normal(k_grad, shape=(opt_cfg.mc_batch_size, disc_params.horizon_steps)) * jnp.sqrt(disc_params.dt)
    
    @jax.jit
    def eval_grid_flat(kp_f, kd_f):
        return jax.vmap(lambda kp, kd: compute_empirical_cost_batch(
            jnp.array([kp, kd]), grid_dW, model_params, disc_params
        ))(kp_f, kd_f)
        
    cost_surface = eval_grid_flat(KP.flatten(), KD.flatten()).reshape((vis_cfg.grid_resolution, vis_cfg.grid_resolution))
    
    cs = ax1.contourf(KP, KD, cost_surface, levels=30, cmap='viridis_r')
    cbar = plt.colorbar(cs, ax=ax1)
    cbar.set_label(f'Empirical Tail-Window Mean Stage-Cost $\\hat{{J}}$ (MC N={opt_cfg.mc_batch_size})')

    path_thetas = jnp.array([h['theta'] for h in primary_history] + [representative_candidate_theta])
    ax1.plot(path_thetas[:, 0], path_thetas[:, 1], 'r.-', linewidth=2, markersize=6, label='SPGD Path (Seed 1)')
    ax1.plot(theta_0[0], theta_0[1], 'wo', markersize=8, label='Initial Candidate $\\theta_0$')
    ax1.plot(multi_run_finals[:, 0], multi_run_finals[:, 1], 'k^', markersize=7, label=f'Final Candidates ({val_cfg.num_optimization_runs} Seeds)')
    ax1.plot(representative_candidate_theta[0], representative_candidate_theta[1], 'm*', markersize=12, label='Representative $\\theta_{\\mathrm{final}}$ (Seed 1)')

    ax1.set_title('Empirical Cost Surface & SPGD Multi-Seed Candidates', fontsize=11)
    ax1.set_xlabel('Proportional Gain ($K_p$)')
    ax1.set_ylabel('Derivative Gain ($K_d$)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    costs_seed1 = eval_oos_batch(representative_candidate_theta)
    paired_diff_seed1 = costs_init - costs_seed1
    mean_diff1 = float(jnp.mean(paired_diff_seed1))

    ax2.hist(paired_diff_seed1, bins=40, alpha=0.7, color='teal', density=True, label='Pathwise Cost Reduction $D_i$')
    ax2.axvline(0.0, color='red', linestyle='--', linewidth=1.5, label='Null Effect Threshold ($D=0$)')
    ax2.axvline(mean_diff1, color='navy', linestyle='-', linewidth=2, label=f'Paired Mean Reduction (+{mean_diff1:.4f})')

    ax2.set_title(f'Paired OOS Reduction Distribution ($N={val_cfg.out_of_sample_eval_n}$, Seed 1 vs Init)', fontsize=11)
    ax2.set_xlabel('Pathwise Reduction: $J(\\theta_0) - J(\\theta_{\\mathrm{final}})$ (Higher = Better)')
    ax2.set_ylabel('Empirical Probability Density')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('delayed_sde_control_benchmark.png', dpi=vis_cfg.dpi)
    print("\nValidation figures exported successfully to 'delayed_sde_control_benchmark.png'.")
