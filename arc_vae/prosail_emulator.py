"""
ARC PROSAIL Emulator — PyTorch re-implementation
=================================================
Loads the pre-trained MLP weights from the ARC package and re-implements
the forward pass in PyTorch, making it differentiable via autograd.

Architecture (from foward_prosail_model_weights.npz):
    Linear(15 → 256) → ReLU
    Linear(256 → 256) → ReLU
    Linear(256 → 256) → ReLU
    Linear(256 → 256) → ReLU
    Linear(256 → 10)

Inputs (15-dim, after pre-processing — see prepare_input()):
    (N-1)/2.5, exp(-Cab/100), exp(-Car/100), Cbrown,
    exp(-50*Cw), exp(-50*Cm), exp(-LAI/2), cos(ALA°),
    cos(SZA°), cos(VZA°), RAA%360/360, p0, p1, p2, p3

Outputs (10-dim):
    S2 surface reflectance for bands B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
"""

import os
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_ARC_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "ARC", "arc", "data"
)
_WEIGHTS_PATH = os.path.join(_ARC_DATA, "foward_prosail_model_weights.npz")


# ---------------------------------------------------------------------------
# PyTorch MLP
# ---------------------------------------------------------------------------
class PROSAILEmulator(nn.Module):
    """
    Differentiable re-implementation of the ARC JAX PROSAIL emulator.

    Call with pre-processed inputs of shape (B, 15).
    Returns S2 reflectance of shape (B, 10).
    """

    def __init__(self, weights_path: str = _WEIGHTS_PATH):
        super().__init__()
        weights = self._load_weights(weights_path)

        # Build layers from stored weight arrays
        # weights[0::2] = weight matrices, weights[1::2] = bias vectors
        layers = []
        n_layers = len(weights) // 2       # = 5
        for i in range(n_layers - 1):      # 4 hidden affine+ReLU layers
            W = torch.from_numpy(weights[2 * i])        # (in, out)
            b = torch.from_numpy(weights[2 * i + 1])    # (out,)
            lin = nn.Linear(W.shape[0], W.shape[1])
            lin.weight = nn.Parameter(W.T.clone())      # nn.Linear stores (out, in)
            lin.bias   = nn.Parameter(b.clone())
            layers += [lin, nn.ReLU()]

        # Output layer (no activation)
        W_out = torch.from_numpy(weights[-2])            # (256, 10)
        b_out = torch.from_numpy(weights[-1])            # (10,)
        out_lin = nn.Linear(W_out.shape[0], W_out.shape[1])
        out_lin.weight = nn.Parameter(W_out.T.clone())
        out_lin.bias   = nn.Parameter(b_out.clone())
        layers.append(out_lin)

        self.net = nn.Sequential(*layers)

    @staticmethod
    def _load_weights(path: str):
        f = np.load(path, allow_pickle=True)
        return f["model_weights"].tolist()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 15) pre-processed inputs

        Returns
        -------
        (B, 10) S2 reflectance
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Input pre-processing  (mirrors prepare_final_input in arc_sample_generator)
# ---------------------------------------------------------------------------
def prepare_input(
    N, cab, cw, cm, lai, ala, cbrown,
    sza, vza, raa,
    p0, p1, p2, p3,
    *,
    as_tensor: bool = True,
) -> torch.Tensor | np.ndarray:
    """
    Apply the same input transformations as ARC's prepare_final_input().

    All scalar or array arguments are broadcast-compatible.
    Car is derived internally as cab / 4.

    Returns a (..., 15) tensor or array.
    """
    car = cab / 4.0

    inp = np.stack([
        (np.asarray(N)      - 1.0) / 2.5,
        np.exp(-np.asarray(cab)  / 100.0),
        np.exp(-np.asarray(car)  / 100.0),
        np.asarray(cbrown),
        np.exp(-50.0 * np.asarray(cw)),
        np.exp(-50.0 * np.asarray(cm)),
        np.exp(-np.asarray(lai)  / 2.0),
        np.cos(np.deg2rad(np.asarray(ala))),
        np.cos(np.deg2rad(np.asarray(sza))),
        np.cos(np.deg2rad(np.asarray(vza))),
        np.asarray(raa) % 360.0 / 360.0,
        np.asarray(p0),
        np.asarray(p1),
        np.asarray(p2),
        np.asarray(p3),
    ], axis=-1).astype(np.float32)

    if as_tensor:
        return torch.from_numpy(inp)
    return inp


# ---------------------------------------------------------------------------
# Numerical verification against JAX original
# ---------------------------------------------------------------------------
def verify_against_jax(n_samples: int = 128, rtol: float = 1e-4):
    """
    Generate random inputs, run both JAX and PyTorch versions,
    and check they agree to within rtol.

    This is the FIRST test that must pass before any training code is written.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ARC"))
    from arc.NN_predict_jax import predict as jax_predict
    from arc.arc_sample_generator import load_model

    rng = np.random.default_rng(42)

    # Random biophysical inputs in physiological ranges
    N       = rng.uniform(1.0,  3.0,   n_samples)
    cab     = rng.uniform(20.0, 80.0,  n_samples)
    cw      = rng.uniform(0.001,0.06,  n_samples)
    cm      = rng.uniform(0.001,0.04,  n_samples)
    lai     = rng.uniform(0.5,  8.0,   n_samples)
    ala     = rng.uniform(50.0, 80.0,  n_samples)
    cbrown  = rng.uniform(0.0,  1.5,   n_samples)
    sza     = rng.uniform(15.0, 60.0,  n_samples)
    vza     = rng.uniform(0.0,  10.0,  n_samples)
    raa     = rng.uniform(0.0,  180.0, n_samples)
    p0      = rng.uniform(0.1,  0.7,   n_samples)
    p1      = rng.uniform(0.0,  0.3,   n_samples)
    p2      = rng.uniform(0.0,  0.6,   n_samples)
    p3      = rng.uniform(0.0,  1.0,   n_samples)

    # Pre-processed input array  (n_samples, 15)
    x_np = prepare_input(N, cab, cw, cm, lai, ala, cbrown,
                         sza, vza, raa, p0, p1, p2, p3, as_tensor=False)

    # --- JAX prediction ---
    arc_weights = load_model(_WEIGHTS_PATH)
    jax_out = np.array(jax_predict(x_np, arc_weights, cal_jac=False))  # (10, n_samples)
    jax_out = jax_out.T   # → (n_samples, 10)

    # --- PyTorch prediction ---
    emulator = PROSAILEmulator()
    emulator.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x_np)
        pt_out = emulator(x_t).numpy()   # (n_samples, 10)

    # --- Compare ---
    max_abs_err = np.abs(jax_out - pt_out).max()
    max_rel_err = (np.abs(jax_out - pt_out) / (np.abs(jax_out) + 1e-8)).max()

    passed = max_rel_err < rtol
    status = "PASSED ✓" if passed else "FAILED ✗"

    print(f"\nPROSAIL emulator verification: {status}")
    print(f"  Samples tested : {n_samples}")
    print(f"  Max abs error  : {max_abs_err:.2e}")
    print(f"  Max rel error  : {max_rel_err:.2e}  (threshold: {rtol:.0e})")
    print(f"  Output range   : [{jax_out.min():.4f}, {jax_out.max():.4f}]")

    if not passed:
        raise AssertionError(
            f"JAX vs PyTorch mismatch: max_rel_err={max_rel_err:.2e} > rtol={rtol}"
        )
    return True


def verify_gradients():
    """
    Check that gradients flow through the PyTorch emulator correctly.
    Uses torch.autograd.gradcheck with double precision.
    """
    emulator = PROSAILEmulator().double()

    rng = np.random.default_rng(0)
    x = torch.from_numpy(
        prepare_input(
            N=2.0, cab=50.0, cw=0.02, cm=0.01, lai=3.0, ala=65.0, cbrown=0.1,
            sza=30.0, vza=5.0, raa=90.0, p0=0.3, p1=0.1, p2=0.3, p3=0.5,
            as_tensor=False,
        ).reshape(1, 15).astype(np.float64)
    ).requires_grad_(True)

    passed = torch.autograd.gradcheck(
        lambda inp: emulator(inp),
        (x,),
        eps=1e-4, atol=1e-3, rtol=1e-3,
    )
    status = "PASSED ✓" if passed else "FAILED ✗"
    print(f"\nGradient check: {status}")
    return passed


if __name__ == "__main__":
    print("=" * 55)
    print("Step 1: Verify PyTorch PROSAIL emulator")
    print("=" * 55)
    verify_against_jax()
    verify_gradients()
    print("\nAll checks passed. Proceed to Step 2.")
