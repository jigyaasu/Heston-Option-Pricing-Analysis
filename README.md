# Heston Stochastic Volatility Model — Pricing, Calibration & Market Application

Implementation, benchmarking, and market calibration of the **Heston (1993) stochastic volatility model** for European option pricing, with a full comparison of pricing engines and a two-stage global/local calibration pipeline validated on real TSLA options data.

## Overview

The Black–Scholes model assumes constant volatility, which fails to reproduce the volatility "smile" observed in real option markets. The **Heston model** treats variance itself as a mean-reverting stochastic process correlated with the underlying asset's returns, giving it the flexibility to fit market-observed smiles.

This project implements the model end-to-end:

1. **Three independent pricing engines** — Monte Carlo simulation, direct numerical integration of the semi-closed-form solution, and the Carr–Madan (1999) FFT method — cross-validated against each other and benchmarked for speed.
2. **A robust calibration framework** that combines a global, derivative-free optimizer (differential evolution) with a local gradient-based polish (least squares) to recover model parameters reliably, even from poor initial guesses.
3. **Application to real market data** — TSLA daily equity options quotes (2019–2022) — comparing calibration objectives (price error vs. implied-volatility error) and pricing input conventions (bid-ask midpoint vs. depth-weighted price).

## The Heston Model

The stock price and its variance evolve according to a pair of correlated stochastic differential equations:

```
dS_t = r S_t dt + sqrt(v_t) S_t dW1_t
dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2_t
corr(dW1_t, dW2_t) = rho
```

| Parameter | Meaning |
|---|---|
| `v0`    | Initial variance |
| `kappa` | Rate of mean reversion of variance |
| `theta` | Long-run mean variance |
| `xi`    | Volatility of volatility |
| `rho`   | Correlation between asset and variance shocks |

The variance process is analogous to a damped spring (Hooke's law) with added noise — `kappa` acts as the spring's stiffness pulling variance back toward its equilibrium `theta`. The **Feller condition**, `2*kappa*theta > xi^2`, must hold to keep variance strictly positive.

## Pricing Methods

### 1. Monte Carlo Simulation
Simulates correlated stock and variance paths directly (Euler discretization with a full-truncation scheme for variance) and averages discounted payoffs. Serves as the ground-truth benchmark, at the cost of runtime and Monte Carlo sampling error.

### 2. Direct Numerical Integration
Evaluates the semi-closed-form Heston pricing formula via the two probability integrals `P1` and `P2`, each requiring a numerical quadrature of the characteristic function. Fast and exact up to quadrature tolerance, but must be re-evaluated per strike.

### 3. Carr–Madan FFT Method
Applies a damping factor to the option price to make it Fourier-transformable, expresses the transform in closed form using Heston's characteristic function, then recovers prices for an **entire strike grid in a single FFT call** using Simpson's-rule discretization. This is the method of choice for calibration, where the model must be repriced hundreds of times across a full strike/maturity surface.

### Benchmark Results

| Pricing Method | Price (ATM Call) | Runtime (1 strike) | Runtime (1,000 strikes) |
|---|---|---|---|
| Monte Carlo (100k paths) | $32.13 (SE $0.17) | 6.63 s | — |
| Direct numerical integration | $32.25 | 0.0057 s | 1.066 s |
| Carr–Madan (FFT) | $32.25 | 0.0058 s | 0.0069 s (**153× faster**) |

All three methods agree to within $0.000282 (0.00000016% of spot) on synthetic data.

## Calibration Framework

Calibration fits the five Heston parameters to observed market option prices or implied volatilities by minimizing a residual error function (`error_price`, `error_iv`, or `error_vega`-normalized). Two optimizers are combined:

- **`scipy.optimize.least_squares`** (local, gradient-based) — fast and precise once near the true minimum, but can converge to the wrong local minimum from a poor starting guess.
- **`scipy.optimize.differential_evolution`** (global, population-based, derivative-free) — needs no initial guess and reliably finds the correct basin of attraction, but converges slowly and imprecisely near the optimum.

**Two-stage pipeline (recommended):** differential evolution locates the right basin; a least-squares polish then refines within it in just a handful of function evaluations. Each stage does only the part it's good at.

| Approach | Result on synthetic data | Runtime |
|---|---|---|
| Least squares (good initial guess) | Recovers true parameters exactly | ~0.8 s, 13 evals |
| Least squares (bad initial guess) | Converges to wrong local minimum | ~0.8 s, 13 evals |
| Differential evolution alone | Recovers true parameters | ~15 s, 1,036 evals |
| **Differential evolution + LS polish** | **Recovers true parameters exactly, immune to bad guesses** | ~15 s (DE) + ~0.2 s (polish) |

Calibrating with the Carr–Madan pricer instead of direct integration gives a **5–12× speedup** with identical recovered parameters.

## Application to TSLA Options Data

The model is calibrated to real end-of-day TSLA option quotes (2019–2022, [Kaggle dataset](https://www.kaggle.com/datasets/kylegraupe/tsla-daily-eod-options-quotes-2019-2022)):

- **Data cleaning:** filters for liquidity (bid/ask depth > 10 contracts), moderate maturity (< 300 DTE), and near-the-money strikes (< 50% strike distance).
- **Market price construction:** both the bid-ask midpoint and a size-weighted average price are computed and calibrated separately.
- **Implied volatility:** a vectorized Newton–bisection solver converts model and market prices into Black–Scholes implied volatilities for smile comparison.

**Key findings:**
- Calibrating to **implied-volatility error** fits the volatility smile (especially the wings) far better than calibrating to **raw price error**, which over-weights expensive in-the-money contracts.
- Carr–Madan and direct-integration calibrations produce nearly identical parameters on real, noisy data — confirming the FFT speedup carries no accuracy cost.
- Mid-price and depth-weighted-price calibrations produce broadly similar fits.
- Calibrated parameters are **not stable day-to-day** (e.g., mean-reversion speed ranged from ~1.5 to ~13.7 across consecutive quote dates), consistent with the known difficulty of daily single-name equity Heston calibration.

## Project Structure

```
.
├── heston_pricing.py        # Characteristic function, Carr–Madan FFT, direct integration, Monte Carlo
├── black_scholes.py         # BS price/delta/vega and vectorized implied-volatility solver
├── calibration.py           # Residual functions, DE + least-squares pipeline
├── data_preprocessing.py    # TSLA options data cleaning and market price/IV construction
├── visualization.py         # Parameter comparison and smile-fit plotting
└── Heston_Project_Final.ipynb  # Full analysis notebook
```

## Requirements

```
numpy
pandas
scipy
matplotlib
seaborn
kagglehub
```

Install with:
```bash
pip install numpy pandas scipy matplotlib seaborn kagglehub
```

## Usage

```python
from heston_pricing import heston_call_carr_madan

price = heston_call_carr_madan(
    S0=175, K=175, v0=0.32**2, r=0.04, t=1,
    kappa=1, theta=0.56**2, xi=0.1, rho=-0.8,
    alpha=1.5, N=4096, eta=0.25
)
```

Calibrating to a set of market quotes:

```python
from scipy.optimize import differential_evolution, least_squares
from calibration import sse_price_carr_madan, error_price_carr_madan

bounds = [[.1**2, .2, .1**2, .5, -.99], [.9**2, 15, .9**2, 5, -.01]]
de_result = differential_evolution(sse_price_carr_madan, bounds=list(zip(*bounds)),
                                    args=(S0, strikes, ttes, r, market_prices))
polished = least_squares(error_price_carr_madan, x0=de_result.x, bounds=bounds,
                          args=(S0, strikes, ttes, r, market_prices))
```

## References

- Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.* Review of Financial Studies.
- Carr, P., & Madan, D. (1999). *Option Valuation Using the Fast Fourier Transform.* Journal of Computational Finance.

## Contributors

Arindam Bhattacharyya , Ankit Kumar, Ethan Zhou, Santiago Neira Lopez
