
# Delayed SDE Stochastic Control Benchmark (JAX)

A rigorous, reproducible numerical study and differentiable Monte Carlo benchmark for a time-delayed stochastic first-order dynamical system with discrete proportional-derivative (PD) control, implemented in pure functional **JAX** with double-precision arithmetic (`float64`).

---

## 1. Executive Summary & Objective

### Single Goal Statement
> **"Determine whether the implemented delayed stochastic-control benchmark produces numerically consistent, reproducible, and statistically distinguishable results over a specified parameter domain, while rigorously verifying the Schur stability of the corresponding deterministic discrete-time closed-loop dynamics."**

This repository serves as a **canonical scientific benchmark** designed to evaluate Automatic Differentiation (AD) through stochastic delay differential integrators, sample-average approximation (SAA) optimization landscapes, and closed-loop spectral stability.

---

## 2. Mathematical Formulation

### A. Continuous-Time Open-Loop Dynamics
The open-loop plant is governed by a one-dimensional stochastic Langevin / Ornstein–Uhlenbeck drift-diffusion process with physical transport delay $\tau > 0$:

$$dx(t) = \left( -\gamma x(t) + u(t - \tau) \right) dt + \sigma dW(t)$$

where:
- $x(t) \in \mathbb{R}$: System state.
- $\gamma > 0$: Natural linear relaxation/damping coefficient ($0.5\text{ s}^{-1}$).
- $\sigma \ge 0$: Brownian diffusion intensity ($0.15$).
- $W(t)$: Standard one-dimensional Wiener process ($dW(t) \sim \mathcal{N}(0, dt)$).
- $\tau > 0$: Actuator transport delay ($\tau = 0.05\text{ s}$).

### B. Discrete-Time State-Augmented Representation
Under an Euler–Maruyama discretization with time step $\Delta t$ such that $\tau = D \cdot \Delta t$ ($D \in \mathbb{N}$), the system evolves on a discrete $(D+2)$-dimensional state space:

$$x_{k+1} = x_k + \left( -\gamma x_k + u_{k-D} \right) \Delta t + \sigma \sqrt{\Delta t} \, \xi_k, \quad \xi_k \sim \mathcal{N}(0, 1)$$

$$e_k = x_k - x^*, \quad \dot{e}_k \approx \frac{e_k - e_{k-1}}{\Delta t}$$

$$u_k = -\left( K_p e_k + K_d \frac{e_k - e_{k-1}}{\Delta t} \right)$$

where $u_{k-D}$ is retrieved from an exact $D$-stage first-in-first-out (FIFO) circular delay buffer $\mathbf{b}_k \in \mathbb{R}^D$.

### C. Objective Functional
The objective is the expected **tail-window mean quadratic stage cost** evaluated over the terminal half of the finite simulation horizon ($T = 6.0\text{ s}$, $N_{\text{steps}} = 600$):

$$\hat{J}(\theta) = \mathbb{E}_{\xi} \left[ \frac{1}{N_{\text{tail}}} \sum_{k=N/2}^{N} \left( q (x_k - x^*)^2 + r u_k^2 \right) \right]$$

with tracking penalty $q = 1.0$, commanded effort penalty $r = 0.005$, and setpoint $x^* = 1.0$.

---

## 3. Methodological Scope Matrix

```
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                                   SCOPE MATRIX (LOCKED)                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║ IN-SCOPE (Explicitly Audited & Mathematically Characterized)                                 ║
║  1. Implementation Correctness: Euler-Maruyama SDE, discrete PD, FIFO delay buffer.          ║
║  2. Differentiation Accuracy: JAX AD vs. Central FD across 3 regimes and 4 decades of h.     ║
║  3. Temporal Refinement: Nested Brownian motion and apparent EOC vs. dt_ref = 0.00125 s.     ║
║  4. Monte Carlo Estimator Scaling: Nested-prefix sampling and empirical N^-1/2 slope check.  ║
║  5. Optimization Robustness: SPGD with exact projected gradient mapping & multi-seed spread. ║
║  6. Exact Discrete Stability: Companion matrix spectral radius Schur stability rho(M) < 1.   ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║ OUT-OF-SCOPE (Explicitly Not Claimed)                                                        ║
║  ❌ Global Optimality: SPGD guarantees only local empirical stationarity.                    ║
║  ❌ Continuous Infinite-Dimensional DDE Operator Resolvent Convergence.                      ║
║  ❌ Mean-Square / Almost-Sure Formal Stochastic Lyapunov Stability Theorems.                 ║
║  ❌ Universal Central Limit Theorem Proof (Evaluated only empirically over sample range).    ║
║  ❌ Real-World Physical Plant Modeling (This is an idealized canonical mathematical problem).║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 4. Verification & Audit Results

Executing `delayed_sde_control_benchmark.py` runs the full 6-stage verification suite:

### Audit 1: Gradient Verification (AD vs. Central FD)
Evaluated across three dynamic regimes (*Nominal Interior*, *Near-Boundary*, *High-Gain*) across $h \in [10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}]$:
- **Result:** Exact $\mathcal{O}(h^2)$ truncation error decay down to $\text{Rel Error} \approx 2.08 \times 10^{-10}$ at $h = 10^{-5}$ in `float64`.
- **Verdict:** **PASSED (All regimes).**

### Audit 2: Timestep Refinement & Apparent EOC
Nested Brownian increments aggregated from the finest reference resolution ($\Delta t_{\text{ref}} = 0.00125\text{ s}$):
- $\Delta t = 0.0250\text{ s} \implies \text{Error} = 0.00341$
- $\Delta t = 0.0100\text{ s} \implies \text{Error} = 0.00324 \quad (\text{Apparent EOC} = 0.06)$
- $\Delta t = 0.0050\text{ s} \implies \text{Error} = 0.00280 \quad (\text{Apparent EOC} = 0.21)$
- $\Delta t = 0.0025\text{ s} \implies \text{Error} = 0.00187 \quad (\text{Apparent EOC} = 0.58)$

### Audit 3: Monte Carlo Estimator Scaling
- Relative Wald 95% Confidence Interval half-width scales monotonically from $\pm 14.55\%$ ($N=16$) down to $\pm 1.75\%$ ($N=1024$), satisfying the $\le 2.5\%$ precision threshold.
- **Empirical Log-Log Decay Slope:** $\frac{d \ln(\text{SEM})}{d \ln(N)} = \mathbf{-0.5213}$ (Consistent with the theoretical CLT scaling $\beta = -0.5000$).

### Audit 4: Multi-Seed SPGD Robustness
Five independent stochastic optimization trajectories from $\theta_0 = [1.00, 0.05]$:
- **Terminal Parameter Candidate:** $K_p = 1.374 \pm 0.015, \quad K_d = 0.054 \pm 0.000$ (Inter-seed parameter spread $\approx 1.1\%$).
- **Stopping Condition:** All runs satisfied projected stationarity $\|G_\eta(\theta)\|_2 < 0.08$ within 40–46 iterations.

### Audit 5: Paired Out-of-Sample (OOS) Statistical Inference
Evaluated on $N = 1000$ unseen Monte Carlo noise realizations:
- **Baseline Cost ($\theta_0$):** $\hat{J} = 0.12186 \pm 0.00111$
- **Representative Candidate Cost ($\theta^*$):** $\hat{J} = 0.07810 \pm 0.00072$
- **Pathwise Paired Mean Cost Reduction:** $\bar{D} = +0.04376 \pm 0.00040$
- **Asymptotic 95% Wald CI:** $[0.04298, 0.04454]$
- **Descriptive Reduction:** $\mathbf{35.91\%}$ ($\mathbf{p < 0.05}$, Statistically Distinguishable Improvement).

### Audit 6: Discrete-Time Closed-Loop Spectral Stability
- **Companion Transition Matrix Spectral Radius:** $\mathbf{\rho(M) = 0.9809 < 1.0}$
- **Mathematical Implication:** The deterministic linear discrete closed-loop system is **provably Schur stable** (all poles reside strictly inside the open unit disk in the complex $z$-plane).

---

## 5. Theoretical Clarifications & FAQ

### Q1: Why does the Initial Condition Recovery Test report `Final Error = 0.2647`?
**Answer (Analytic Steady-State Offset of PD Regulation):**  
In a first-order system with linear damping $\gamma > 0$ under proportional-derivative feedback without an integral term ($K_i = 0$), the unforced continuous steady state $x_{\text{ss}}$ satisfies:

$$0 = -\gamma x_{\text{ss}} + u_{\text{ss}} \implies -\gamma x_{\text{ss}} - K_p (x_{\text{ss}} - x^*) = 0$$

$$x_{\text{ss}} = \frac{K_p}{\gamma + K_p} x^* = \frac{1.389}{0.5 + 1.389} (1.0) = \mathbf{0.7353}$$

The theoretical steady-state tracking error is:

$$e_{\text{ss}} = |x_{\text{ss}} - x^*| = |0.7353 - 1.0| = \mathbf{0.2647}$$

All displaced initial conditions ($x(0) \in \{-3, -1, 0, 2, 4\}$) converge asymptotically to this exact equilibrium point ($e_{\text{final}} \approx 0.2647$). This confirms deterministic asymptotic stability ($\rho(\mathbf{M}) < 1$); the non-zero offset is a fundamental mathematical property of pure PD regulation on a damped plant, not a dynamical instability.

### Q2: What is the exact ontological status of the model?
**Answer:** The benchmark is defined as an **exact discrete-time state-augmented linear dynamical system** $\mathbf{s}_{k+1} = \mathbf{M} \mathbf{s}_k + \mathbf{B} \xi_k$ of dimension $D+2 = 7$. Spectral radius proofs and JAX automatic differentiation are exact with respect to this finite-dimensional computational graph.

### Q3: Is this calibrated against a real-world physical device?
**Answer:** No. This is an idealized **canonical synthetic benchmark problem** designed for numerical methodology and stochastic control research.

---

## 6. How to Reproduce

### Dependencies
- Python $\ge 3.9$
- JAX $\ge 0.4.0$
- NumPy, Pandas, Matplotlib

### Run the Benchmark
```bash
python delayed_sde_control_benchmark.py
```
Upon completion, validation logs will be printed to `stdout` and validation plots will be exported to `delayed_sde_control_benchmark.png`.
```

