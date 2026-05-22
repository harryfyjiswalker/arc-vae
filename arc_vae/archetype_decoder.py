"""
ARC Archetype Decoder — Differentiable PyTorch Implementation
__

Implements the mapping  (p, h, DOYs, angles, soil) --> S2 reflectance
following the ARC forward model, but entirely in PyTorch so that gradients 
flow back through the decoder to update the encoder.

The ARC forward model in calendar time works as follows:
  1. From phenology parameters h = (growth_speed, start, senes_speed, end),
     compute a normalised double-logistic curve L(t; h) in [0,1] for every
     calendar DOY t.
  2. Compute a fixed reference logistic L_ref from the archetype median LAI
     trajectory.  This defines the archetype's own phenological clock.
  3. For each calendar DOY t, find the archetype clock value τ = L_ref_inv(L(t;h)).
     τ is itself a calendar DOY in the archetype's reference frame.
  4. Look up each parameter's archetype median at τ:
       a^j(τ) = meds[τ, j]    (the archetype median trajectory)
  5. Scale:  x_j(t) = clip(p_j  *  a^j(τ(t; h)),  lo_j,  hi_j)
  6. Feed x(t) + soil + geometry into the PROSAIL emulator to get r̂(t).

Steps 3–4 are made differentiable via linear interpolation into meds.

Parameters
----------
p  : (B, 7)  scaling parameters  [p_N, p_Cab, p_Cm, p_Cw, p_LAI, p_ALA, p_Cbrown]
h  : (B, 4)  phenology parameters [growth_speed, start_of_season,
                                    senescence_speed, end_of_season]
doys   : (T,)      integer day-of-year values for the observations
angles : (B, T, 3) or (T, 3)  [sza, vza, raa] in degrees
soil   : (B, 4)    soil parameters [p0_raw, p1_raw, p2_raw, p3_raw]

Returns
-------
r_hat  : (B, T, 10)  predicted S2 surface reflectance
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from prosail_emulator import PROSAILEmulator, prepare_input

# Paths
_ARC_DATA = Path(__file__).parent.parent / "ARC" / "arc" / "data"

# Physiological limits (order: N, Cab, Cm, Cw, LAI, ALA, Cbrown)
_PARAM_NAMES = ["N", "Cab", "Cm", "Cw", "LAI", "ALA", "Cbrown"]
_LO = torch.tensor([1.0,  0.0,   0.0,   0.0,  0.0,  20.0, 0.0],  dtype=torch.float32)
_HI = torch.tensor([3.0, 140.0,  0.04,  0.1,  10.0, 90.0, 1.5],  dtype=torch.float32)



# Differentiable double-logistic
# Evaluate cyclical seasonal green-up and senescence curves using continuous sigmoids
def double_logistic(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor: 
    """
    Compute the ARC double-logistic curve in a differentiable way.

    Parameters
    ----------
    h : (B, 4)  [growth_speed, start_of_season, senescence_speed, end_of_season_ABSOLUTE]
                IMPORTANT: ARC sample_logistic modifies pheo_samples[:,3] in-place by
                adding start_of_season before returning. So pheo_s[:,3] from
                generate_arc_refs is already the ABSOLUTE end DOY — do NOT add
                start_of_season again.
    t : (T,)    calendar DOYs as 0-based array indices (DOY 150 -> t=149)

    Returns
    -------
    L : (B, T)  double-logistic values clipped to >= 0, before normalisation
    """
    k1 = h[:, 0:1]   # growth speed        (B, 1)
    t1 = h[:, 1:2]   # start of season     (B, 1) absolute DOY index
    k2 = h[:, 2:3]   # senescence speed    (B, 1)
    t2 = h[:, 3:4]   # end of season       (B, 1) ABSOLUTE — start already added by ARC

    t_ = t.unsqueeze(0)   # (1, T)

    # ARC formula: L = 1 - sigma1 - sigma2  where
    #   sigma1 = 1/(1+exp(k1*(t-t1)))  → high BEFORE t1, low after (growth phase marker)
    #   sigma2 = 1/(1+exp(-k2*(t-t2))) → low BEFORE t2, high after (senescence marker)
    sigma1 = torch.sigmoid(-k1 * (t_ - t1))   # = 1/(1+exp(k1*(t-t1)))
    sigma2 = torch.sigmoid( k2 * (t_ - t2))   # = 1/(1+exp(-k2*(t-t2)))

    L = 1.0 - sigma1 - sigma2
    return L.clamp(min=0.0)   # ARC clips negatives to 0; (B, T)


def normalise_logistic(L: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalise each sample's logistic curve to [0, 1]."""
    Lmin = L.amin(dim=1, keepdim=True)
    Lmax = L.amax(dim=1, keepdim=True)
    return (L - Lmin) / (Lmax - Lmin + eps)   # (B, T)



# Archetype decoder module
class ARCDecoder(nn.Module):
    """
    Differentiable ARC archetype forward model + PROSAIL emulator.

    The decoder is fixed.  Gradients flow through it to update the encoder.

    Parameters
    ----------
    crop_type : str
        One of {'maize', 'wheat', 'soybean', 'rice'}
    weights_path : str | Path
        Path to foward_prosail_model_weights.npz
    n_ref_points : int
        Resolution of the reference logistic inverse lookup table (default 1000)
    """

    def __init__(
        self,
        crop_type: str = "maize",
        weights_path: str | Path = _ARC_DATA / "foward_prosail_model_weights.npz",
        n_ref_points: int = 1000,
    ):
        super().__init__()

        # PROSAIL emulator (no grad — weights are fixed)
        self.prosail = PROSAILEmulator(str(weights_path))
        for param in self.prosail.parameters():
            param.requires_grad_(False)

        # Load archetype median trajectories  (365, 7)
        crop_files = {
            "maize":   "US_001.npz",
            "soybean": "US_005.npz",
            "rice":    "China_000.npz",
            "wheat":   "US_024.npz",
        }
        arc_data = np.load(_ARC_DATA / crop_files[crop_type.lower()])
        meds  = arc_data["meds"].astype(np.float32)   # (365, 7)
        maxs  = np.nanmax(meds, axis=0)               # (7,)  archetype peak per param

        # Register as buffers so they move with .to(device) but are not trained
        self.register_buffer("meds",  torch.from_numpy(meds))   # (365, 7)
        self.register_buffer("maxs",  torch.from_numpy(maxs))   # (7,)
        self.register_buffer("lo",    _LO)
        self.register_buffer("hi",    _HI)

        # Pre-compute the reference logistic and its inverse lookup table.
        # L_ref(t) is derived from the archetype median LAI (column 4),
        # matching exactly what ARC's compute_reference_parameters does.
        self._build_ref_inverse(meds, n_ref_points)

    # Reference logistic inverse (fixed, pre-computed)
    def _build_ref_inverse(self, meds: np.ndarray, n_ref_points: int):
        """
        Build a fixed lookup:  normalised logistic value v → archetype DOY τ.

        ARC fits a double-logistic to the archetype median LAI, then inverts it
        to map any sample's logistic value back to archetype calendar time.
        We pre-compute this inverse as a piecewise-linear table.
        """
        import sys, os
        sys.path.insert(0, str(Path(__file__).parent.parent / "ARC"))
        from arc.arc_sample_generator import compute_reference_parameters, logistic_function

        days  = np.arange(365, dtype=np.float64)
        ref_p = compute_reference_parameters(meds)

        # Reference logistic values over 365 days
        y_ref = logistic_function(ref_p, days)
        y_ref = (y_ref - y_ref.min()) / (y_ref.max() - y_ref.min() + 1e-9)

        # Mirror the descending part (ARC flips after peak so the mapping is monotone)
        argmax = int(np.argmax(y_ref))
        y_ref_mono = y_ref.copy()
        y_ref_mono[argmax:] = y_ref_mono[argmax:] * -1 + 2   # range [1, 2] on descent

        # Build a fine grid of v values --> corresponding archetype DOY
        v_grid = np.linspace(y_ref_mono.min(), y_ref_mono.max(), n_ref_points)
        doy_grid = np.interp(v_grid, y_ref_mono, days)  # monotone → invertible

        self.register_buffer("v_grid",   torch.from_numpy(v_grid.astype(np.float32)))
        self.register_buffer("doy_grid", torch.from_numpy(doy_grid.astype(np.float32)))
        self._ref_argmax = argmax   # needed to mirror sample logistic the same way

  
    # Core forward: (p, h, doys, angles, soil) --> reflectance
    
    def forward(
        self,
        p:      torch.Tensor,   # (B, 7)
        h:      torch.Tensor,   # (B, 4)
        doys:   torch.Tensor,   # (T,)   calendar DOYs (1-indexed, e.g. 150)
        angles: torch.Tensor,   # (B, T, 3) or (T, 3)  [sza, vza, raa]
        soil:   torch.Tensor,   # (B, 4)   raw soil params
    ) -> torch.Tensor:          # (B, T, 10)
        """
        Decode latent (p, h) to S2 reflectance time series.
        """
        B  = p.shape[0]
        T  = doys.shape[0]

        # 1. Compute normalised sample logistic over ALL 365 days
 
        t_all = torch.arange(365, dtype=torch.float32, device=p.device)  # (365,)
        L_all = double_logistic(h, t_all)       # (B, 365)

        # Normalise each sample's curve to [0, 1] over the full year
        Lmin = L_all.amin(dim=1, keepdim=True)
        Lmax = L_all.amax(dim=1, keepdim=True)
        L_norm_all = (L_all - Lmin) / (Lmax - Lmin + 1e-8)   # (B, 365)

        # Mirror descending part to make curve monotone [0-->1, 1-->2].
        #
        # ARC uses hard argmax to find the peak, which is non-differentiable.
        # We replace it with a differentiable soft-peak estimate:
        #   t_peak ≈ (k2*t1 + k1*t2) / (k1+k2)  (weighted midpoint of the logistic)
        # and a smooth sigmoid mask instead of a binary step.
        #
        # This approximation enables ∂L/∂h to be non-zero so the encoder learns phenology.
        k1 = h[:, 0:1]     # (B,1)
        t1 = h[:, 1:2]
        k2 = h[:, 2:3]
        t2 = h[:, 3:4]
        t_peak = (k2 * t1 + k1 * t2) / (k1 + k2 + 1e-8)   # (B, 1)  differentiable

        # Soft mask: ≈0 before peak, ≈1 after peak.  alpha=20 gives approx 10-DOY transition.
        t_grid = t_all.unsqueeze(0)                            # (1, 365)
        soft_mask = torch.sigmoid(20.0 * (t_grid - t_peak))   # (B, 365) ∈ (0,1)

        # L_mono: ascending part stays [0,1], descending part mapped to [1,2]
        L_mono_all = (1.0 - soft_mask) * L_norm_all + soft_mask * (2.0 - L_norm_all)

        # 2. Extract logistic values at observation DOYs
        #    ARC uses doys as 1-indexed --> array index = doy - 1
        doy_idx = (doys.long() - 1).clamp(0, 364)        # (T,) 0-indexed
        L_mono_obs = L_mono_all[:, doy_idx]               # (B, T)

        # 3. Map to archetype calendar DOY via inverse reference logistic
        tau = self._interp1d(L_mono_obs.reshape(-1),
                              self.v_grid, self.doy_grid)  # (B*T,)
        tau = tau.reshape(B, T)                             # (B, T)  archetype DOY ∈ [0, 364]

        # 3. Look up archetype median at tau  (differentiable interp)
        # meds is (365, 7); tau is (B, T); output (B, T, 7)
        a = self._lookup_meds(tau)   # (B, T, 7)

        # 4. Scale: x_j(t) = p_j * a_j(tau)  then clip to physical bounds
        x = p.unsqueeze(1) * a                             # (B, T, 7)
        x = torch.clamp(x, self.lo, self.hi)               # (B, T, 7)

        # 5. Normalise soil params (same as ARC adjust_soil_params)
        soil_norm = self._normalise_soil(soil, angles, doys.float(), B, T)  # (B, T, 4)

        # 6. Apply PROSAIL input pre-processing and run emulator
        N       = x[..., 0]   # (B, T)
        cab     = x[..., 1]
        cm      = x[..., 2]
        cw      = x[..., 3]
        lai     = x[..., 4]
        ala     = x[..., 5]
        cbrown  = x[..., 6]
        car     = cab / 4.0

        if angles.dim() == 2:   # (T, 3)  broadcast over batch
            angles = angles.unsqueeze(0).expand(B, -1, -1)  # (B, T, 3)
        sza = angles[..., 0]   # (B, T)
        vza = angles[..., 1]
        raa = angles[..., 2]

        p0 = soil_norm[..., 0]
        p1 = soil_norm[..., 1]
        p2 = soil_norm[..., 2]
        p3 = soil_norm[..., 3]

        # Assemble PROSAIL input  (B*T, 15)
        inp = torch.stack([
            (N   - 1.0)  / 2.5,
            torch.exp(-cab    / 100.0),
            torch.exp(-car    / 100.0),
            cbrown,
            torch.exp(-50.0 * cw),
            torch.exp(-50.0 * cm),
            torch.exp(-lai    / 2.0),
            torch.cos(torch.deg2rad(ala)),
            torch.cos(torch.deg2rad(sza)),
            torch.cos(torch.deg2rad(vza)),
            raa % 360.0 / 360.0,
            p0, p1, p2, p3,
        ], dim=-1).reshape(B * T, 15)   # (B*T, 15)

        r = self.prosail(inp)           # (B*T, 10)
        r = r.reshape(B, T, 10)         # (B, T, 10)
        return r

    # Helper: 1D linear interpolation (differentiable)
    @staticmethod
    def _interp1d(
        x: torch.Tensor,        # (N,) query points
        xp: torch.Tensor,       # (M,) grid (sorted ascending)
        fp: torch.Tensor,       # (M,) values
    ) -> torch.Tensor:
        """Differentiable piecewise-linear interpolation."""
        # Clamp to grid range
        x_ = x.clamp(xp[0], xp[-1])

        # Find lower index
        idx = torch.searchsorted(xp.contiguous(), x_.contiguous())
        idx = idx.clamp(1, len(xp) - 1)
        idx_lo = idx - 1

        x_lo = xp[idx_lo]
        x_hi = xp[idx]
        f_lo = fp[idx_lo]
        f_hi = fp[idx]

        # Linear interpolation weight
        w = (x_ - x_lo) / (x_hi - x_lo + 1e-12)
        return f_lo + w * (f_hi - f_lo)

    # Helper: look up meds at continuous DOY tau  (B, T) --> (B, T, 7)
    def _lookup_meds(self, tau: torch.Tensor) -> torch.Tensor:
        """Differentiable linear interpolation into the 365-day meds table."""
        B, T = tau.shape
        tau_ = tau.clamp(0, 364).reshape(-1)   # (B*T,)

        idx_lo = tau_.long().clamp(0, 363)
        idx_hi = (idx_lo + 1).clamp(0, 364)
        w      = (tau_ - idx_lo.float()).unsqueeze(-1)   # (B*T, 1)

        a_lo = self.meds[idx_lo]   # (B*T, 7)
        a_hi = self.meds[idx_hi]   # (B*T, 7)
        a    = a_lo + w * (a_hi - a_lo)   # (B*T, 7)
        return a.reshape(B, T, 7)

    # Helper: normalise soil parameters (as in ARC adjust_soil_params)
    def _normalise_soil(
        self,
        soil: torch.Tensor,     # (B, 4)  raw [brightness, shape_p1, shape_p2, moisture]
        angles: torch.Tensor,   # (B, T, 3) or (T, 3)
        t: torch.Tensor,        # (T,) DOYs
        B: int, T: int,
    ) -> torch.Tensor:          # (B, T, 4)
        """
        Apply the same soil normalisation as ARC's adjust_soil_params.
        The Walthall coefficient is computed per (sza, vza, raa) per DOY.
        """
        if angles.dim() == 2:
            angles_b = angles.unsqueeze(0).expand(B, -1, -1)
        else:
            angles_b = angles   # (B, T, 3)

        sza = torch.deg2rad(angles_b[..., 0])   # (B, T)
        vza = torch.deg2rad(angles_b[..., 1])
        raa = torch.deg2rad(angles_b[..., 2])

        walthall = (1.0 / 16.41) * (
            sza * vza * torch.cos(raa) * 7.363
            - 4.3 * (vza**2 + sza**2)
            + 7.702 * sza**2 * vza**2
        )  # (B, T)

        p0 = soil[:, 0:1] / 1.5                        # (B, 1)
        p1 = (soil[:, 1:2] - 10.0) / 70.0
        p2 = (soil[:, 2:3] - 22.0) / (130.0 - 22.0)
        p3 = (soil[:, 3:4] -  2.0) / (100.0 - 2.0)

        p0 = p0 + p0 * walthall   # (B, T)  broadcast
        p1 = p1.expand(B, T)
        p2 = p2.expand(B, T)
        p3 = p3.expand(B, T)

        return torch.stack([p0, p1, p2, p3], dim=-1)   # (B, T, 4)


# Verification
def verify_decoder(n_samples: int = 16, rtol: float = 0.05):
    """
    Generate synthetic samples via ARC's generate_arc_refs, then run the
    same (p, h, soil, angles, DOYs) through our PyTorch decoder and check
    that the predicted reflectances agree with ARC's output.

    Tolerance is 5% relative — some discrepancy is expected due to the
    linear-interpolation approximation of the time-warp lookup.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "ARC"))
    from arc.arc_sample_generator import generate_arc_refs

    np.random.seed(7)
    doys_np = np.arange(150, 260, 8)   # ~14 acquisition dates
    angs_np = (
        np.full(len(doys_np), 30.0),
        np.full(len(doys_np),  5.0),
        np.full(len(doys_np), 120.0),
    )

    arc_refs, pheo_s, bio_s, orig_bios, soil_s = generate_arc_refs(
        doys_np, 150, 45, n_samples, angs_np, "maize"
    )
    # arc_refs: (10, T, N)  — S2 bands, DOYs, samples

    # Convert ARC outputs to tensors
    # pheo_s: (N, 4)  raw phenology params
    # bio_s:  (N, 7)  raw scaling params (divided by archetype max already)
    # soil_s: (N, 4)  raw soil params

    maize = np.load(_ARC_DATA / "US_001.npz")
    maxs  = np.nanmax(maize["meds"], axis=0)   # (7,)

    # p is bio_s * maxs  (back to physical scaling, so that p=1 --> peak archetype)
    p_t = torch.from_numpy(bio_s.astype(np.float32))      # (N, 7)

    # h: pheo_s columns are [growth_speed, start, senescence_speed, end_relative]
    # ARC: pheo_samples[:, 3] += pheo_samples[:, 1]  BEFORE the logistic call
    # Here double_logistic adds h[:,3] to h[:,1] internally — so pass raw pheo_s
    h_t = torch.from_numpy(pheo_s.astype(np.float32))     # (N, 4)

    doys_t = torch.from_numpy(doys_np.astype(np.float32)) # (T,)

    angles_t = torch.zeros(n_samples, len(doys_np), 3)
    angles_t[..., 0] = 30.0  # sza
    angles_t[..., 1] =  5.0  # vza
    angles_t[..., 2] = 120.0 # raa

    soil_t = torch.from_numpy(soil_s.astype(np.float32))  # (N, 4)

    # Run decoder
    decoder = ARCDecoder(crop_type="maize")
    decoder.eval()
    with torch.no_grad():
        r_hat = decoder(p_t, h_t, doys_t, angles_t, soil_t)  # (N, T, 10)

    # ARC reference:  arc_refs (10, T, N) --> (N, T, 10)
    r_arc = torch.from_numpy(arc_refs.transpose(2, 1, 0).astype(np.float32))

    abs_err = (r_hat - r_arc).abs()
    rel_err = abs_err / (r_arc.abs() + 1e-6)

    print(f"\nDecoder verification (n={n_samples}, T={len(doys_np)} dates):")
    print(f"  Mean abs error : {abs_err.mean():.5f}")
    print(f"  Max  abs error : {abs_err.max():.5f}")
    print(f"  Mean rel error : {rel_err.mean():.4f}")
    print(f"  Max  rel error : {rel_err.max():.4f}")
    print(f"  Reflectance range (ARC): [{r_arc.min():.4f}, {r_arc.max():.4f}]")

    passed = rel_err.mean().item() < rtol
    status = "Passed." if passed else "Fail: mean rel error exceeds tolerance."
    print(f"  Result: {status}")
    return passed


def verify_gradients():
    """Confirm that gradients flow through the full decoder."""
    decoder = ARCDecoder(crop_type="maize")

    B, T = 4, 8
    p = torch.ones(B, 7,    requires_grad=True)
    # h[:,3] must be the ABSOLUTE end DOY (> start).
    h = torch.tensor([[0.15, 140.0, 0.10, 240.0]] * B, requires_grad=True)
    doys   = torch.arange(150, 150 + T * 8, 8, dtype=torch.float32)
    angles = torch.zeros(B, T, 3)
    angles[..., 0] = 30.0; angles[..., 1] = 5.0; angles[..., 2] = 120.0
    soil   = torch.tensor([[0.3, 15.0, 30.0, 20.0]] * B)

    r = decoder(p, h, doys, angles, soil)
    loss = r.sum()
    loss.backward()

    p_grad_ok = p.grad is not None and p.grad.abs().sum() > 0
    h_grad_ok = h.grad is not None and h.grad.abs().sum() > 0

    print(f"\nGradient flow check:")
    print(f"  ∂L/∂p non-zero: {'Pass' if p_grad_ok else 'Fail'}")
    print(f"  ∂L/∂h non-zero: {'Pass' if h_grad_ok else 'Fail'}")
    print(f"  ∂L/∂p sample:   {p.grad[0].tolist()}")
    print(f"  ∂L/∂h sample:   {h.grad[0].tolist()}")

    return p_grad_ok and h_grad_ok


if __name__ == "__main__":
    print("Verifying differentiable ARC archetype decoder")
    v1 = verify_decoder()
    v2 = verify_gradients()
    if v1 and v2:
        print("\nPassed..")
    else:
        print("\nSome checks failed..")
