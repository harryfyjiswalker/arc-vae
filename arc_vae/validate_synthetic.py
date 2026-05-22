"""
Step 6: Synthetic Validation Experiments
=========================================
Tests the trained ARC-VAE encoder on synthetic data with known ground truth.

Three regimes:
  A. Identifiability  — clean data, recover known (p, h)
  B. Cloud stress     — increasing cloud fraction, measure accuracy degradation
  C. Calibration      — are posterior intervals calibrated?

Usage:
  python validate_synthetic.py \
    --checkpoint /path/to/checkpoint_best.pt \
    --out_dir    /path/to/output/dir
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "ARC"))
sys.path.insert(0, str(Path(__file__).parent))

from training import ARCVAE
from synthetic_dataloader import (
    GermanyMaizeDataset, _apply_cloud_masking, MAX_OBS, N_BANDS
)

PARAM_NAMES = ["p_N", "p_Cab", "p_Cm", "p_Cw", "p_LAI", "p_ALA", "p_Cbrown",
               "h_growth", "h_start", "h_senes", "h_end"]
P_NAMES = PARAM_NAMES[:7]
H_NAMES = PARAM_NAMES[7:]

# Physical parameter names and units — matching ARC paper's validation
PHYS_NAMES = ["N", "Cab", "Cm", "Cw", "LAI", "ALA", "Cbrown"]
PHYS_UNITS = ["", "µg/cm²", "g/cm²", "g/cm²", "m²/m²", "°", ""]


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cpu") -> ARCVAE:
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Config is stored in config.json in the same directory as the checkpoint
    config_path = Path(checkpoint_path).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    model = ARCVAE(
        crop_type=config["crop_type"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        n_queries=config["n_queries"],
        d_ff=config["d_ff"],
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Regime A: Identifiability
# ---------------------------------------------------------------------------

def regime_a_identifiability(model, n_samples=2000, seed=1000, device="cpu"):
    """
    Generate synthetic data with known (p, h), run encoder,
    compare recovered posterior mean against ground truth.

    Returns per-parameter RMSE and R².
    """
    print("\n=== Regime A: Identifiability ===")

    ds = GermanyMaizeDataset(n_samples=n_samples, seed=seed)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    all_mu    = []
    all_p_true = []
    all_h_true = []

    with torch.no_grad():
        for batch in loader:
            s2_refl  = batch["s2_refl"] .to(device)
            angles   = batch["angles"]  .to(device)
            doys     = batch["doys"]    .to(device)
            obs_mask = batch["obs_mask"].to(device)

            mu, sigma = model.encoder(s2_refl, angles, doys, obs_mask)

            all_mu    .append(mu.cpu())
            all_p_true.append(batch["p_true"])
            all_h_true.append(batch["h_true"])

    mu_all     = torch.cat(all_mu,     dim=0).numpy()   # (N, 11)
    p_true_all = torch.cat(all_p_true, dim=0).numpy()   # (N, 7)
    h_true_all = torch.cat(all_h_true, dim=0).numpy()   # (N, 4)
    z_true_all = np.concatenate([p_true_all, h_true_all], axis=1)  # (N, 11)

    results = {}
    print(f"\n{'Parameter':12s}  {'RMSE':8s}  {'R²':8s}  {'Bias':8s}")
    print("-" * 48)

    for j, name in enumerate(PARAM_NAMES):
        pred  = mu_all[:, j]
        true  = z_true_all[:, j]
        rmse  = np.sqrt(np.mean((pred - true) ** 2))
        bias  = np.mean(pred - true)
        ss_res = np.sum((pred - true) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2    = 1 - ss_res / max(ss_tot, 1e-12)
        results[name] = {"rmse": float(rmse), "r2": float(r2), "bias": float(bias)}
        print(f"  {name:12s}  {rmse:8.4f}  {r2:8.4f}  {bias:+8.4f}")

    return results, mu_all, z_true_all


# ---------------------------------------------------------------------------
# Regime B: Cloud stress test
# ---------------------------------------------------------------------------

def regime_b_cloud_stress(model, n_samples=1000, seed=2000, device="cpu"):
    """
    Generate clean synthetic data, then apply increasing cloud fractions
    and measure how recovery accuracy degrades.

    Cloud fractions tested: 0%, 20%, 40%, 60%, 80%
    """
    print("\n=== Regime B: Cloud Stress Test ===")

    ds = GermanyMaizeDataset(n_samples=n_samples, seed=seed)

    # Store the clean data
    all_refl   = torch.from_numpy(ds._data["s2_refl"])    # (N, T, 10)
    all_angles = torch.from_numpy(ds._data["angles"])     # (N, T, 3)
    all_doys   = torch.from_numpy(ds._data["doys"])       # (N, T)
    all_mask   = torch.from_numpy(ds._data["obs_mask"])   # (N, T)
    p_true     = ds._data["p_true"]                       # (N, 7)
    h_true     = ds._data["h_true"]                       # (N, 4)
    z_true     = np.concatenate([p_true, h_true], axis=1) # (N, 11)

    cloud_fractions = [0.0, 0.2, 0.4, 0.6, 0.8]
    results = {}
    rng = np.random.default_rng(seed + 1)

    print(f"\n{'Cloud%':8s}  {'Mean_obs':8s}  {'LAI_RMSE':10s}  {'h_start_RMSE':12s}  {'Mean_R²':8s}")
    print("-" * 56)

    for cf in cloud_fractions:
        # Apply additional cloud masking on top of existing mask
        stressed_mask = all_mask.clone()

        if cf > 0:
            for i in range(len(ds)):
                valid_idx = torch.where(all_mask[i])[0].numpy()
                n_valid   = len(valid_idx)
                n_remove  = int(cf * n_valid)
                if n_remove > 0 and n_valid - n_remove >= 5:
                    remove_idx = rng.choice(valid_idx, size=n_remove, replace=False)
                    stressed_mask[i, remove_idx] = False

        # Inference in batches
        all_mu = []
        B_size = 256
        with torch.no_grad():
            for start in range(0, len(ds), B_size):
                end  = min(start + B_size, len(ds))
                refl = all_refl  [start:end].to(device)
                ang  = all_angles[start:end].to(device)
                doy  = all_doys  [start:end].to(device)
                msk  = stressed_mask[start:end].to(device)

                # Zero out newly masked positions
                refl = refl * msk.unsqueeze(-1).float()
                ang  = ang  * msk.unsqueeze(-1).float()

                mu, _ = model.encoder(refl, ang, doy, msk)
                all_mu.append(mu.cpu())

        mu_all   = torch.cat(all_mu, dim=0).numpy()
        mean_obs = stressed_mask.float().sum(dim=1).mean().item()

        # Per-parameter RMSE
        rmse_per = {}
        r2_per   = {}
        for j, name in enumerate(PARAM_NAMES):
            pred   = mu_all[:, j]
            true   = z_true[:, j]
            rmse   = np.sqrt(np.mean((pred - true) ** 2))
            ss_res = np.sum((pred - true) ** 2)
            ss_tot = np.sum((true - true.mean()) ** 2)
            r2     = 1 - ss_res / max(ss_tot, 1e-12)
            rmse_per[name] = float(rmse)
            r2_per  [name] = float(r2)

        lai_rmse     = rmse_per["p_LAI"]
        hstart_rmse  = rmse_per["h_start"]
        mean_r2      = float(np.mean(list(r2_per.values())))

        results[cf] = {
            "mean_obs":   mean_obs,
            "rmse":       rmse_per,
            "r2":         r2_per,
            "mean_r2":    mean_r2,
        }

        print(f"  {cf*100:5.0f}%  {mean_obs:8.1f}  {lai_rmse:10.4f}  {hstart_rmse:12.4f}  {mean_r2:8.4f}")

    return results


# ---------------------------------------------------------------------------
# Regime C: Uncertainty calibration
# ---------------------------------------------------------------------------

def regime_c_calibration(model, n_samples=2000, seed=3000,
                          n_mc=100, device="cpu"):
    """
    Check if posterior intervals are calibrated.

    For each parameter, draw n_mc samples from the posterior TN(μ,σ;lo,hi)
    and check if the true value falls within the nominal interval.

    Reports PICP (prediction interval coverage probability) at 50%, 90%, 95%.
    """
    print("\n=== Regime C: Uncertainty Calibration ===")
    from training import reparameterise

    ds = GermanyMaizeDataset(n_samples=n_samples, seed=seed)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    all_mu    = []
    all_sigma = []
    all_p_true = []
    all_h_true = []

    with torch.no_grad():
        for batch in loader:
            s2_refl  = batch["s2_refl"] .to(device)
            angles   = batch["angles"]  .to(device)
            doys     = batch["doys"]    .to(device)
            obs_mask = batch["obs_mask"].to(device)

            mu, sigma = model.encoder(s2_refl, angles, doys, obs_mask)
            all_mu   .append(mu.cpu())
            all_sigma.append(sigma.cpu())
            all_p_true.append(batch["p_true"])
            all_h_true.append(batch["h_true"])

    mu_all    = torch.cat(all_mu,    dim=0)   # (N, 11)
    sigma_all = torch.cat(all_sigma, dim=0)   # (N, 11)
    p_true    = torch.cat(all_p_true, dim=0)  # (N, 7)
    h_true    = torch.cat(all_h_true, dim=0)  # (N, 4)
    z_true    = torch.cat([p_true, h_true], dim=1)  # (N, 11)

    z_lo = model.encoder.z_lo.cpu()  # (11,)
    z_hi = model.encoder.z_hi.cpu()  # (11,)

    # Draw n_mc posterior samples per pixel
    # z_samples: (N, n_mc, 11)
    z_samples = torch.stack([
        reparameterise(mu_all, sigma_all, z_lo, z_hi)
        for _ in range(n_mc)
    ], dim=1)

    nominal_levels = [0.50, 0.90, 0.95]
    results = {}

    print(f"\n{'Parameter':12s}", end="")
    for lvl in nominal_levels:
        print(f"  PICP@{int(lvl*100)}%", end="")
    print(f"  Mean_σ")
    print("-" * 60)

    for j, name in enumerate(PARAM_NAMES):
        pred_samples = z_samples[:, :, j]   # (N, n_mc)
        true_vals    = z_true[:, j]          # (N,)
        mean_sigma   = sigma_all[:, j].mean().item()

        picps = {}
        for lvl in nominal_levels:
            alpha = (1 - lvl) / 2
            lo_q  = torch.quantile(pred_samples, alpha,     dim=1)
            hi_q  = torch.quantile(pred_samples, 1 - alpha, dim=1)
            covered = ((true_vals >= lo_q) & (true_vals <= hi_q)).float()
            picps[lvl] = covered.mean().item()

        results[name] = {
            "picps":      {str(lvl): picps[lvl] for lvl in nominal_levels},
            "mean_sigma": mean_sigma,
        }

        print(f"  {name:12s}", end="")
        for lvl in nominal_levels:
            picp = picps[lvl]
            flag = "✓" if abs(picp - lvl) < 0.08 else "✗"
            print(f"  {picp:.3f}{flag}", end="")
        print(f"  {mean_sigma:.4f}")

    return results


# ---------------------------------------------------------------------------
# Regime A2: Validate x_canopy(t) in physical units (ARC-comparable)
# ---------------------------------------------------------------------------

def regime_a2_xcano_physical(model, n_samples=2000, seed=1000, device="cpu"):
    """
    The ARC-comparable validation.

    ARC validates recovered biophysical parameter TIME SERIES x_canopy(t)
    against ground measurements. We do the same in synthetic space:
      - Generate known (p_true, h_true, orig_bios)
      - Run encoder to get (p̂, ĥ)
      - Compute x̂_canopy(t) = p̂_j * meds[τ(t;ĥ), j] at each observation DOY
      - Compare against orig_bios (the true biophysical trajectory)
      - Report RMSE in physical units

    This is directly comparable to ARC's reported performance:
      LAI RMSE=0.94 m²/m², Cab RMSE=6.59 µg/cm²,  CCw RMSE=0.03 g/cm²
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "ARC"))
    from arc.arc_sample_generator import generate_arc_refs
    import numpy as np

    print("\n=== Regime A2: x_canopy(t) validation in physical units ===")
    print("(Directly comparable to ARC paper Table/Fig 14)")

    # Load archetype medians for x_canopy reconstruction
    from archetype_decoder import ARCDecoder
    decoder_ref = ARCDecoder(crop_type="maize")
    meds = decoder_ref.meds.numpy()   # (365, 7)

    rng = np.random.default_rng(seed)
    doys_np = np.arange(130, 285, 8)   # ~20 obs covering full German maize season
    angs_np = (
        np.full(len(doys_np), 35.0),
        np.full(len(doys_np),  5.0),
        np.full(len(doys_np), 120.0),
    )

    all_x_true = []   # (N, T, 7)  true biophysical values
    all_x_pred = []   # (N, T, 7)  predicted biophysical values
    all_mu      = []

    batch_size = 256
    n_generated = 0

    while n_generated < n_samples:
        n_this = min(batch_size, n_samples - n_generated)
        try:
            arc_refs, pheo_s, bio_s, orig_bios, soil_s = generate_arc_refs(
                doys_np, 130, 145, n_this, angs_np, "maize"
            )
        except Exception as e:
            continue

        n_act = bio_s.shape[0]

        # orig_bios: (7, T, N) in internal ARC units — need to map to physical
        # ARC stores as integers * scale factors:
        scales = np.array([100., 100., 10000., 10000., 100., 100., 1000.])
        # x_physical = orig_bios[j, t, i] / scales[j]
        x_true_phys = orig_bios.transpose(2, 1, 0) / scales[None, None, :]  # (N, T, 7)

        # Build tensors for encoder
        T = len(doys_np)
        MAX_OBS = 50
        s2_refl  = np.zeros((n_act, MAX_OBS, 10), dtype=np.float32)
        ang_arr  = np.zeros((n_act, MAX_OBS, 3),  dtype=np.float32)
        doy_arr  = np.zeros((n_act, MAX_OBS),      dtype=np.float32)
        mask_arr = np.zeros((n_act, MAX_OBS),      dtype=bool)

        # arc_refs: (10, T, N) -> (N, T, 10)
        refs_np = arc_refs.transpose(2, 1, 0).astype(np.float32)
        s2_refl[:, :T, :]  = refs_np
        ang_arr[:, :T, 0]  = angs_np[0]
        ang_arr[:, :T, 1]  = angs_np[1]
        ang_arr[:, :T, 2]  = angs_np[2]
        doy_arr[:, :T]     = doys_np.astype(np.float32)
        mask_arr[:, :T]    = True

        s2_t   = torch.from_numpy(s2_refl).to(device)
        ang_t  = torch.from_numpy(ang_arr).to(device)
        doy_t  = torch.from_numpy(doy_arr).to(device)
        msk_t  = torch.from_numpy(mask_arr).to(device)

        with torch.no_grad():
            mu, _ = model.encoder(s2_t, ang_t, doy_t, msk_t)

        mu_np   = mu.cpu().numpy()     # (N, 11)
        p_hat   = mu_np[:, :7]         # (N, 7)  predicted scaling parameters
        h_hat   = mu_np[:, 7:]         # (N, 4)  predicted phenology

        # Compute x̂_canopy(t) = p̂_j * meds[τ(t;ĥ), j]
        # Use the decoder's archetype lookup at each observation DOY
        p_t_hat = torch.from_numpy(p_hat.astype(np.float32)).to(device)
        h_t_hat = torch.from_numpy(h_hat.astype(np.float32)).to(device)
        doys_t_ = torch.from_numpy(doys_np.astype(np.float32)).to(device)

        with torch.no_grad():
            # Compute normalised logistic and tau for predicted h
            # Use decoder's internal methods — move decoder to device
            decoder_ref_dev = decoder_ref.to(device)
            t_all = torch.arange(365, dtype=torch.float32, device=device)
            from archetype_decoder import double_logistic

            L_all = double_logistic(h_t_hat, t_all)   # (N, 365)
            Lmin  = L_all.amin(dim=1, keepdim=True)
            Lmax  = L_all.amax(dim=1, keepdim=True)
            L_norm = (L_all - Lmin) / (Lmax - Lmin + 1e-8)

            # Soft-flip (same as decoder)
            k1 = h_t_hat[:, 0:1]; t1 = h_t_hat[:, 1:2]
            k2 = h_t_hat[:, 2:3]; t2 = h_t_hat[:, 3:4]
            t_peak = (k2 * t1 + k1 * t2) / (k1 + k2 + 1e-8)
            t_grid = t_all.unsqueeze(0)
            soft_mask = torch.sigmoid(20.0 * (t_grid - t_peak))
            L_mono = (1.0 - soft_mask) * L_norm + soft_mask * (2.0 - L_norm)

            # Extract at obs DOYs and map to archetype DOY
            doy_idx = (doys_t_.long() - 1).clamp(0, 364)
            L_obs   = L_mono[:, doy_idx]   # (N, T)

            tau = decoder_ref_dev._interp1d(
                L_obs.reshape(-1),
                decoder_ref_dev.v_grid,
                decoder_ref_dev.doy_grid
            ).reshape(len(p_hat), T)   # (N, T)

            # Look up archetype and scale
            a = decoder_ref_dev._lookup_meds(tau)   # (N, T, 7)
            x_pred = p_t_hat.unsqueeze(1) * a       # (N, T, 7)
            x_pred_np = x_pred.cpu().numpy()         # in physical units

        all_x_true.append(x_true_phys[:n_act])
        all_x_pred.append(x_pred_np)
        n_generated += n_act

    x_true = np.concatenate(all_x_true, axis=0)[:n_samples]   # (N, T, 7)
    x_pred = np.concatenate(all_x_pred, axis=0)[:n_samples]   # (N, T, 7)

    # Flatten over N and T for metrics
    x_true_flat = x_true.reshape(-1, 7)
    x_pred_flat = x_pred.reshape(-1, 7)

    arc_targets = {
        "LAI":    (4, "m²/m²",  0.94),
        "Cab":    (1, "µg/cm²", 6.59),
        "Cw":     (3, "g/cm²",  0.03),
        "Cbrown": (6, "",       0.15),
    }

    print(f"\n{'Parameter':8s}  {'RMSE':8s}  {'Units':8s}  {'R²':6s}  {'ARC_RMSE':10s}  {'vs_ARC'}")
    print("-" * 65)

    results = {}
    for pname, (j, units, arc_rmse) in arc_targets.items():
        pred = x_pred_flat[:, j]
        true = x_true_flat[:, j]
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        ss_res = np.sum((pred - true) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        ratio = rmse / arc_rmse
        flag = "✓" if ratio < 2.0 else "✗"
        print(f"  {pname:8s}  {rmse:8.4f}  {units:8s}  {r2:6.4f}  "
              f"{arc_rmse:10.4f}  {ratio:.2f}x {flag}")
        results[pname] = {"rmse": float(rmse), "r2": float(r2),
                           "units": units, "arc_rmse": arc_rmse}

    print("\n  ARC targets: LAI RMSE=0.94 m²/m², Cab RMSE=6.59 µg/cm²,")
    print("               CCw RMSE=0.03 g/cm², Cbrown RMSE=0.15")
    print("  Note: ARC validated on REAL data; ours is synthetic (easier).")
    print("  Ratio < 1.0 = better than ARC; < 2.0 = within factor 2 of ARC.")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint_best.pt")
    parser.add_argument("--out_dir",    required=True,
                        help="Directory to save results")
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--n_samples",  type=int, default=2000)
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "cpu") if args.device == "auto" else args.device
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, device)
    print(f"Model loaded from epoch {torch.load(args.checkpoint, map_location='cpu')['epoch']}")

    # Run experiments
    res_a,  mu_a, z_true_a = regime_a_identifiability(
        model, n_samples=args.n_samples, device=device)

    res_a2 = regime_a2_xcano_physical(
        model, n_samples=args.n_samples, device=device)

    res_b = regime_b_cloud_stress(
        model, n_samples=args.n_samples // 2, device=device)

    res_c = regime_c_calibration(
        model, n_samples=args.n_samples, device=device)

    # Save results
    results = {
        "regime_a":  res_a,
        "regime_a2": res_a2,
        "regime_b":  {str(k): v for k, v in res_b.items()},
        "regime_c":  res_c,
    }
    out_path = os.path.join(args.out_dir, "synthetic_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n=== Summary ===")
    print("Regime A — Identifiability (clean data):")
    for name in ["p_LAI", "p_Cab", "h_start", "h_end"]:
        r = res_a[name]
        print(f"  {name:12s}: RMSE={r['rmse']:.4f}  R²={r['r2']:.4f}")

    print("\nRegime B — Cloud stress (LAI RMSE vs cloud fraction):")
    for cf, r in res_b.items():
        print(f"  {cf*100:.0f}% cloud: LAI_RMSE={r['rmse']['p_LAI']:.4f}  mean_obs={r['mean_obs']:.1f}")

    print("\nRegime C — Calibration (PICP should match nominal level):")
    for name in ["p_LAI", "h_start"]:
        picps = res_c[name]["picps"]
        print(f"  {name:12s}: PICP@50%={picps['0.5']:.3f}  PICP@90%={picps['0.9']:.3f}")


if __name__ == "__main__":
    main()
