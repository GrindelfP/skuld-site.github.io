# Skuld — Neural Numerical Integration

A Python library that uses neural networks to compute definite integrals analytically — no quadrature, no mesh, just trained weights and corner sums.

## Overview

Skuld supports two fundamentally different approaches to neural numerical integration:

- **Lloyd et al. (2020)** — Approximate the *integrand* with a shallow MLP, then integrate the network analytically using polylogarithms.
- **Maître et al. (2022)** — Train a network to approximate the *antiderivative*, then evaluate at hypercube corners with an alternating-sign sum.

Five architectures are available under a unified API: MLP, SIREN, WIRE, KAN, and BrokNet.

## Pages

| Page | Description |
|------|-------------|
| [Home](index.html) | Library overview and quick start |
| [Documentation](docs.html) | Full API reference |
| [Research](research.html) | Physics integrand, methods, and results |
| [License](license.html) | MIT License |
| [Author](author.html) | Contact links |

## Installation

```bash
pip install -e skuld-lib/
```

## Quick Start

```python
from skuld.siren import SirenIntegrator

integrator = SirenIntegrator(n_params=4, n_int_vars=3)
history, norm_cache = integrator.train(
    integrand_fn=integrand_fn,
    param_sets=[(0, 0, 1, 2), (1, 1, 2, 3)],
    n_epochs=8000,
)
result = integrator.integrate((0, 0, 1, 2), norm_cache=norm_cache)
```

## License

This website is released under the [MIT License](LICENSE).
