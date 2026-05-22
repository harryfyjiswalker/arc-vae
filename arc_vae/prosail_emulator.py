"""
Differentiable Pytorch Re-Implementation of ARC PROSAIL Emulator 

This module loads the pre-trained Multi-Layer Perceptron (MLP) weights from the 
ARC package and re-implements the forward radiative transfer pass inside PyTorch. 
The code translaties the original JAX model array operations into native PyTorch tensor 
manipulations to allow the canopy reflectance generation step to become fully trackable by 
autograd. This allows end-to-end gradient backpropagation during encoder training.

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
    A 10-dimensional tensor of shape (B, 10) or (B*T, 10) containing predicted 
    Sentinel-2 top-of-canopy surface reflectances mapped to bands:
    [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
"""

import os
import numpy as np
import torch
import torch.nn as nn


# Path helpers
import arc as _arc
_ARC_DATA    = os.path.join(os.path.dirname(os.path.abspath(_arc.__file__)), "data") # Establish the path to the internal data folder within the repo structure
_WEIGHTS_PATH = os.path.join(_ARC_DATA, "foward_prosail_model_weights.npz") # Build the absolute system path to the pre-trained JAX model weights archive


# PyTorch MLP
class PROSAILEmulator(nn.Module):
    """
    Differentiable re-implementation of the ARC JAX PROSAIL emulator.

    Call with pre-processed inputs of shape (B, 15).
    Returns S2 reflectance of shape (B, 10).
    """

    def __init__(self, weights_path: str = _WEIGHTS_PATH):
        # Initialise the parent nn.Module class structure
        super().__init__()
        # Unpack the pre-trained weight dictionary layers from the array file
        weights = self._load_weights(weights_path)

        # Container array to hold linear transformations and activations
        layers = []
        # Quantify total layers by splitting weight-bias pair count
        n_layers = len(weights) // 2       # = 5
        
        # Sequentially rebuild the 4 hidden affine layer stages
        for i in range(n_layers - 1):      # 4 hidden affine+ReLU layers
            W = torch.from_numpy(weights[2 * i])        # Isolate layer weight matrix
            b = torch.from_numpy(weights[2 * i + 1])    # Isolate subsequent layer bias vector
            lin = nn.Linear(W.shape[0], W.shape[1])     # Define the standard Linear execution block
            lin.weight = nn.Parameter(W.T.clone())      # Transpose weights to fit PyTorch layout criteria
            lin.bias   = nn.Parameter(b.clone())        # Load pre-trained layer biases into memory parameters
            layers += [lin, nn.ReLU()]                  # Append transformed linear block and activation layer

        # Build the final unactivated spectral band projection layer
        W_out = torch.from_numpy(weights[-2])            # (256, 10)
        b_out = torch.from_numpy(weights[-1])            # (10,)
        out_lin = nn.Linear(W_out.shape[0], W_out.shape[1])
        out_lin.weight = nn.Parameter(W_out.T.clone())    # Load cloned output weight transformation matrix
        out_lin.bias   = nn.Parameter(b_out.clone())      # Load cloned output bias vector configurations
        layers.append(out_lin)                            # Append final layer block to the structure list

        # Compile linear steps into an automated execution sequence
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _load_weights(path: str):
        f = np.load(path, allow_pickle=True) # Open weight file using picking settings
        return f["model_weights"].tolist() # Extract weight parameters list object from file dictionary

    def forward(self, x: torch.Tensor) -> torch.Tensor: # Pass input features straight through the compiled linear sequence
        """
        Parameters
        ----------
        x : (B, 15) pre-processed inputs

        Returns
        -------
        (B, 10) S2 reflectance
        """
        return self.net(x)


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

    # Stack transformed features along last axis to match the model input space
    inp = np.stack([
        (np.asarray(N)      - 1.0) / 2.5,  # Normalise structural dimension N
        np.exp(-np.asarray(cab)  / 100.0),  # Exponential transform for chlorophyll absorption tracking
        np.exp(-np.asarray(car)  / 100.0),  # Exponential transform for carotenoid absorption tracking
        np.asarray(cbrown),                 # Bind raw brown pigment values directly
        np.exp(-50.0 * np.asarray(cw)),     # Scale and apply exponential attenuation to equivalent water thickness
        np.exp(-50.0 * np.asarray(cm)),     # Scale and apply exponential attenuation to dry matter content
        np.exp(-np.asarray(lai)  / 2.0),    # Apply exponential decay transform to Leaf Area Index values
        np.cos(np.deg2rad(np.asarray(ala))),  # Convert average leaf angle to radians for cosine mapping
        np.cos(np.deg2rad(np.asarray(sza))),  # Convert solar zenith angle to radians for cosine mapping
        np.cos(np.deg2rad(np.asarray(vza))),  # Convert view zenith angle to radians for cosine mapping
        np.asarray(raa) % 360.0 / 360.0,      # Map relative azimuth angle onto a proportional unit circle fraction
        np.asarray(p0),  # Bind first adjusted soil parameter             
        np.asarray(p1),  # Bind second adjusted soil parameter
        np.asarray(p2),  # Bind third adjusted soil parameter
        np.asarray(p3),  # Bind fourth adjusted soil parameter
    ], axis=-1).astype(np.float32)

    # Conditionally cast the finalised array configuration into a native PyTorch tensor
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

    """
    import sys
    from arc.NN_predict_jax import predict as jax_predict
    from arc.arc_sample_generator import load_model

    rng = np.random.default_rng(42)

    # Sample random test parameters covering valid physiological intervals
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

   # Execute original frozen model calculations using JAX execution path
    arc_weights = load_model(_WEIGHTS_PATH)
    jax_out = np.array(jax_predict(x_np, arc_weights, cal_jac=False))  # (10, n_samples)
    jax_out = jax_out.T   # (n_samples, 10)

    # Execute corresponding calculations inside PyTorch environment
    emulator = PROSAILEmulator()
    emulator.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x_np)
        pt_out = emulator(x_t).numpy()   # (n_samples, 10)

    # Quantify absolute and relative errors between runtime frameworks
    max_abs_err = np.abs(jax_out - pt_out).max()
    max_rel_err = (np.abs(jax_out - pt_out) / (np.abs(jax_out) + 1e-8)).max()

    passed = max_rel_err < rtol # Terminate application loop if framework values diverge past threshold
    status = "PASSED" if passed else "FAILED"

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
    # Construct a reference input block with gradient tracking activated
    x = torch.from_numpy(
        prepare_input(
            N=2.0, cab=50.0, cw=0.02, cm=0.01, lai=3.0, ala=65.0, cbrown=0.1,
            sza=30.0, vza=5.0, raa=90.0, p0=0.3, p1=0.1, p2=0.3, p3=0.5,
            as_tensor=False,
        ).reshape(1, 15).astype(np.float64)
    ).requires_grad_(True)

    # Execute analytical checking step to confirm gradient consistency
    passed = torch.autograd.gradcheck(
        lambda inp: emulator(inp),
        (x,),
        eps=1e-4, atol=1e-3, rtol=1e-3,
    )
    status = "PASSED ✓" if passed else "FAILED ✗"
    print(f"\nGradient check: {status}")
    return passed


if __name__ == "__main__":
    print("Verifying PyTorch PROSAIL emulator")
    verify_against_jax()
    verify_gradients()
    print("\nWorking.")
