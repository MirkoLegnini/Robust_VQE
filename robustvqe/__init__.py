"""
robustvqe
=========

Numerical experiments of Sec. VII of "Noise Resilience and Robust Convergence
Guarantees for the Variational Quantum Eigensolver" (M. Legnini, J. Berberich).
See README.md; the entry point is reproduce.py.
"""

import jax

jax.config.update("jax_enable_x64", True)        # double precision throughout
jax.config.update("jax_platform_name", "cpu")
