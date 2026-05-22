"""
ARC-KNN Baseline Comparison
============================
Runs ARC's own KNN solver on the same synthetic test samples used to
evaluate the Transformer encoder, giving a fair head-to-head comparison.

This answers the key question: how much of the performance gap between
our encoder and ARC's published results is due to:
  (a) the approximation gap (encoder vs KNN) — measured here
  (b) the synthetic-to-real gap — requires real data validation

Usage:
  python compare_arc_knn.py \
    --checkpoint /path/to/checkpoint_best.pt \
    --out_dir    /path/to/output/dir \
    --n_samples  500 \
    --n_library  131072

Notes on n_library:
  ARC uses 2^21 = 2,097,152 Sobol samples in production.
  For comparison purposes 2^17 = 131,072 is sufficient and fast (~10 min).
  Use 2^19 = 524,288 for a closer approximation to ARC's production setting.
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

# Import directly from the module file to avoid arc/__init__.py
# which pulls in field_processor -> pynndescent -> numba chain
import importlib.util, pathlib
_arc_gen_path = pathlib.Path(__file__).parent.parent / "ARC" / "arc" / "arc_sample_generator.py"
_spec = importlib.util.spec_from_file_location("arc_sample_generator", _arc_gen_path)
_arc_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_arc_gen)
generate_arc_refs = _arc_gen.generate_arc_refs

# Import get_neighbours the same way
_knn_path = pathlib.Path(__file__).parent.parent / "ARC" / "arc" / "approximate_KNN_search.py"
_knn_spec = importlib.util.spec_from_file_location("approximate_KNN_search", _knn_path)
_knn_mod  = importlib.util.module_from_spec(_knn_spec)
_knn_spec.loader.exec_module(_knn_mod)
get_neighbours = _knn_mod.get_neighbours
from arc.approximate_KNN_search import get_neighbours
from training import ARCVAE
from synthetic_dataloader import GermanyMaizeDataset
from validate_synthetic import load_model as load_vae_model
from archetype_decoder import ARCDecoder


# ---------------------------------------------------------------------------
# ARC KNN inference on a set of observations
# ---------------------------------------------------------------------------

def arc_knn_predict(
    obs_refs:   np.ndarray,   # (T, 10)  observed S2 reflectance (bands last)
    obs_errs:   np.ndarray,   # (T, 10)  per-band uncertainty
    doys:       np.ndarray,   # (T,)     calendar DOYs
    angs:       tuple,        # (sza, vza, raa) each (T,)
    start_of_season:     int,
    growth_season_length: int,
    n_library:  int = 131072,
    k:          int = 100,
    crop_type:  str = "maize",
) -> dict:
    """
    Run ARC's KNN solver for one pixel's time series.

    Returns dict with:
      bio_mean  : (7,)  weighted mean of K-nearest biophysical params
      pheo_mean : (4,)  weighted mean of K-nearest phenology params
      bio_std   : (7,)  weighted std (ARC's uncertainty estimate)
    """
    # Generate Sobol library
    arc_refs_lib, pheo_lib, bio_lib, orig_bios_lib, soil_lib = generate_arc_refs(
        doys, start_of_season, growth_season_length,
        n_library, angs, crop_type
    )
    # arc_refs_lib: (10, T, N_lib)
    # obs_refs: (T, 10) → need (10, T, 1) for get_neighbours
    obs_refs_T = obs_refs.T[:, :, np.newaxis]  # (10, T, 1)
    obs_errs_T = obs_errs.T[:, :, np.newaxis]  # (10, T, 1)

    # Get K nearest neighbours
    nn_idx = get_neighbours(
        obs_refs_T, obs_errs_T, arc_refs_lib,
        doys, steps=10, k=k
    )  # (1, k)

    nn_idx = nn_idx[0]   # (k,)

    # Compute distances for weighting
    # Distance = weighted Euclidean between obs and library reflectances
    obs_flat = obs_refs_T[:, :, 0].flatten()        # (10*T,)
    lib_flat  = arc_refs_lib[:, :, nn_idx].reshape(10 * len(doys), k)  # (10*T, k)
    errs_flat = obs_errs_T[:, :, 0].flatten()       # (10*T,)
    weights_sq = 1.0 / (errs_flat[:, None] ** 2 + 1e-12)
    dists_sq = (weights_sq * (obs_flat[:, None] - lib_flat) ** 2).sum(axis=0)  # (k,)
    weights = dists_sq / (dists_sq.sum() + 1e-12)   # (k,)  inverse distance^2

    # Weighted mean and std of bio and pheo parameters
    bio_nn   = bio_lib [nn_idx]    # (k, 7)
    pheo_nn  = pheo_lib[nn_idx]    # (k, 4)

    bio_mean  = (weights[:, None] * bio_nn ).sum(axis=0)   # (7,)
    pheo_mean = (weights[:, None] * pheo_nn).sum(axis=0)   # (4,)
    bio_std   = np.sqrt(
        (weights[:, None] * (bio_nn  - bio_mean [None])**2).sum(axis=0))  # (7,)
    pheo_std  = np.sqrt(
        (weights[:, None] * (pheo_nn - pheo_mean[None])**2).sum(axis=0))  # (4,)

    return {
        "bio_mean":  bio_mean,
        "pheo_mean": pheo_mean,
        "bio_std":   bio_std,
        "pheo_std":  pheo_std,
    }


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(
    checkpoint_path: str,
    out_dir:         str,
    n_samples:       int = 200,
    n_library:       int = 131072,
    device:          str = "cpu",
):
    """
    Compare ARC-KNN and our Transformer encoder on the same synthetic data.

    For each test sample:
      1. Generate synthetic (X, p_true, h_true) via generate_arc_refs
      2. Run our encoder → (p̂_enc, ĥ_enc)
      3. Run ARC KNN on same X → (p̂_knn, ĥ_knn)
      4. Compare both against (p_true, h_true) and against x_canopy_true

    Reports per-parameter RMSE for both methods side by side.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load encoder
    print(f"Loading encoder from {checkpoint_path}")
    model = load_vae_model(checkpoint_path, device)
    decoder_ref = ARCDecoder(crop_type="maize").to(device)

    # ARC archetype metadata
    maize = np.load(
        Path(__file__).parent.parent / "ARC" / "arc" / "data" / "US_001.npz"
    )
    meds  = maize["meds"]    # (365, 7)
    scales = np.array([100., 100., 10000., 10000., 100., 100., 1000.])

    param_names = ["p_N","p_Cab","p_Cm","p_Cw","p_LAI","p_ALA","p_Cbrown",
                   "h_growth","h_start","h_senes","h_end"]
    # All 7 biophysical parameters reported in physical units.
    # ARC paper validated against MNI Germany ground measurements only for
    # LAI, Cab, Cw, Cbrown (no field measurements of N, Cm, ALA available).
    # We still report our own performance on N, Cm, ALA (no ARC benchmark).
    phys_names  = ["LAI",    "Cab",    "Cw",     "Cbrown", "N",  "Cm",     "ALA"]
    phys_idx    = [4,        1,        3,        6,        0,    2,        5    ]
    phys_units  = ["m²/m²",  "µg/cm²", "g/cm²",  "",       "",   "g/cm²",  "°"  ]
    arc_rmse    = [0.94,     6.59,     0.03,     0.15,     None, None,     None]

    # Fixed observation geometry for comparison (same as validation)
    doys_np = np.arange(130, 310, 8)   # ~23 observations, covers full German maize season
    T = len(doys_np)
    sza_arr = np.full(T, 35.0)
    vza_arr = np.full(T, 5.0)
    raa_arr = np.full(T, 120.0)
    angs    = (sza_arr, vza_arr, raa_arr)
    start_of_season      = 135
    growth_season_length = 145

    print(f"\nGenerating {n_samples} test samples...")
    np.random.seed(12345)
    arc_refs_test, pheo_s, bio_s, orig_bios, soil_s = generate_arc_refs(
        doys_np, start_of_season, growth_season_length,
        n_samples, angs, "maize"
    )
    # arc_refs_test: (10, T, N)
    # Note: generate_arc_refs rounds to Sobol sequence length, so actual N may differ
    n_samples = arc_refs_test.shape[2]   # use actual number generated
    print(f"  (generate_arc_refs returned {n_samples} samples)")

    # Add SIAC-level noise
    sigma_base = 0.005
    sigma_rel  = 0.020
    rng = np.random.default_rng(99)
    noise_sigma = sigma_base + sigma_rel * np.abs(arc_refs_test)
    obs_refs = arc_refs_test + rng.normal(0, noise_sigma)
    obs_refs = np.clip(obs_refs, 0, 1)   # (10, T, N)

    # True x_canopy in physical units: orig_bios (7, T, N) / scales
    x_true = orig_bios.transpose(2, 1, 0) / scales[None, None, :]  # (N, T, 7)

    print(f"Running encoder on {n_samples} test samples...")
    # Build encoder input tensors
    MAX_OBS = 50
    s2_refl = np.zeros((n_samples, MAX_OBS, 10), dtype=np.float32)
    ang_arr = np.zeros((n_samples, MAX_OBS, 3),  dtype=np.float32)
    doy_arr = np.zeros((n_samples, MAX_OBS),      dtype=np.float32)
    mask_arr= np.zeros((n_samples, MAX_OBS),      dtype=bool)
    sig_arr = np.zeros((n_samples, MAX_OBS, 10),  dtype=np.float32)

    refs_np = obs_refs.transpose(2, 1, 0).astype(np.float32)  # (N, T, 10)
    s2_refl[:, :T, :]  = refs_np
    ang_arr[:, :T, 0]  = sza_arr
    ang_arr[:, :T, 1]  = vza_arr
    ang_arr[:, :T, 2]  = raa_arr
    doy_arr[:, :T]     = doys_np.astype(np.float32)
    mask_arr[:, :T]    = True
    sig_arr[:, :T, :]  = (sigma_base + sigma_rel * refs_np).astype(np.float32)

    with torch.no_grad():
        mu_enc, sigma_enc = model.encoder(
            torch.from_numpy(s2_refl).to(device),
            torch.from_numpy(ang_arr).to(device),
            torch.from_numpy(doy_arr).to(device),
            torch.from_numpy(mask_arr).to(device),
        )
    mu_enc = mu_enc.cpu().numpy()   # (N, 11)

    # Encoder x_canopy predictions
    p_enc   = torch.from_numpy(mu_enc[:, :7].astype(np.float32)).to(device)
    h_enc   = torch.from_numpy(mu_enc[:, 7:].astype(np.float32)).to(device)
    doys_t  = torch.from_numpy(doys_np.astype(np.float32)).to(device)
    ang_t   = torch.zeros(n_samples, T, 3, device=device)
    ang_t[...,0]=35.; ang_t[...,1]=5.; ang_t[...,2]=120.
    soil_t  = torch.from_numpy(soil_s[:n_samples].astype(np.float32)).to(device)

    with torch.no_grad():
        from archetype_decoder import double_logistic
        t_all  = torch.arange(365, dtype=torch.float32, device=device)
        L_all  = double_logistic(h_enc, t_all)
        Lmin   = L_all.amin(1, keepdim=True)
        Lmax   = L_all.amax(1, keepdim=True)
        L_norm = (L_all - Lmin)/(Lmax - Lmin + 1e-8)
        k1=h_enc[:,0:1]; t1=h_enc[:,1:2]
        k2=h_enc[:,2:3]; t2=h_enc[:,3:4]
        t_pk   = (k2*t1 + k1*t2)/(k1+k2+1e-8)
        tg     = t_all.unsqueeze(0)
        sm     = torch.sigmoid(20.*(tg - t_pk))
        L_mono = (1-sm)*L_norm + sm*(2-L_norm)
        doy_idx= (doys_t.long()-1).clamp(0,364)
        L_obs  = L_mono[:, doy_idx]
        tau_enc= decoder_ref._interp1d(
            L_obs.reshape(-1), decoder_ref.v_grid, decoder_ref.doy_grid
        ).reshape(n_samples, T)
        a_enc  = decoder_ref._lookup_meds(tau_enc)
        x_enc  = (p_enc.unsqueeze(1) * a_enc).cpu().numpy()  # (N, T, 7)

    print(f"\nRunning ARC-KNN on {n_samples} test samples...")
    print(f"(Library size: {n_library:,} Sobol samples per pixel)")
    print(f"This will take approximately {n_samples * 1.5:.0f} seconds...\n")

    bio_knn_list  = []
    pheo_knn_list = []

    for i in range(n_samples):
        if i % 20 == 0:
            print(f"  KNN: {i}/{n_samples}")

        obs_i  = obs_refs[:, :, i].T   # (T, 10)
        errs_i = sig_arr[i, :T, :]     # (T, 10)

        result = arc_knn_predict(
            obs_i, errs_i, doys_np, angs,
            start_of_season, growth_season_length,
            n_library=n_library, k=100, crop_type="maize"
        )
        bio_knn_list .append(result["bio_mean"])
        pheo_knn_list.append(result["pheo_mean"])

    bio_knn  = np.stack(bio_knn_list)    # (N, 7)
    pheo_knn = np.stack(pheo_knn_list)   # (N, 4)

    # KNN x_canopy predictions
    p_knn   = torch.from_numpy(bio_knn .astype(np.float32)).to(device)
    h_knn   = torch.from_numpy(pheo_knn.astype(np.float32)).to(device)
    with torch.no_grad():
        L_all  = double_logistic(h_knn, t_all)
        Lmin   = L_all.amin(1, keepdim=True)
        Lmax   = L_all.amax(1, keepdim=True)
        L_norm = (L_all - Lmin)/(Lmax - Lmin + 1e-8)
        k1=h_knn[:,0:1]; t1=h_knn[:,1:2]
        k2=h_knn[:,2:3]; t2=h_knn[:,3:4]
        t_pk   = (k2*t1 + k1*t2)/(k1+k2+1e-8)
        sm     = torch.sigmoid(20.*(tg - t_pk))
        L_mono = (1-sm)*L_norm + sm*(2-L_norm)
        L_obs  = L_mono[:, doy_idx]
        tau_knn= decoder_ref._interp1d(
            L_obs.reshape(-1), decoder_ref.v_grid, decoder_ref.doy_grid
        ).reshape(n_samples, T)
        a_knn  = decoder_ref._lookup_meds(tau_knn)
        x_knn  = (p_knn.unsqueeze(1) * a_knn).cpu().numpy()  # (N, T, 7)

    # ---- Compute metrics ----
    z_true = np.concatenate([bio_s[:n_samples], pheo_s[:n_samples]], axis=1)  # (N,11)
    z_enc  = mu_enc
    z_knn  = np.concatenate([bio_knn, pheo_knn], axis=1)

    print("\n" + "="*70)
    print("RESULTS: ARC-KNN vs Transformer Encoder (same synthetic test set)")
    print("="*70)

    print(f"\n{'Parameter':12s}  {'KNN_RMSE':10s}  {'ENC_RMSE':10s}  {'KNN_R²':8s}  {'ENC_R²':8s}  {'Ratio'}")
    print("-"*65)

    results = {}
    for j, name in enumerate(param_names):
        knn_rmse = np.sqrt(np.mean((z_knn[:,j] - z_true[:,j])**2))
        enc_rmse = np.sqrt(np.mean((z_enc[:,j] - z_true[:,j])**2))
        def r2(pred, true):
            ss = np.sum((pred-true)**2)
            st = np.sum((true-true.mean())**2)
            return 1 - ss/max(st,1e-12)
        knn_r2 = r2(z_knn[:,j], z_true[:,j])
        enc_r2 = r2(z_enc[:,j], z_true[:,j])
        ratio  = enc_rmse / max(knn_rmse, 1e-9)
        results[name] = {
            "knn_rmse": float(knn_rmse), "enc_rmse": float(enc_rmse),
            "knn_r2":   float(knn_r2),   "enc_r2":   float(enc_r2),
        }
        flag = "✓" if ratio < 1.5 else ("~" if ratio < 2.0 else "✗")
        print(f"  {name:12s}  {knn_rmse:10.4f}  {enc_rmse:10.4f}  "
              f"{knn_r2:8.4f}  {enc_r2:8.4f}  {ratio:.2f}x {flag}")

    print(f"\n{'Parameter':8s}  {'KNN_RMSE':10s}  {'ENC_RMSE':10s}  {'Units':8s}  "
          f"{'KNN_R²':7s}  {'ENC_R²':7s}  {'ARC_pub':8s}")
    print("-"*70)
    print("Physical units (x_canopy):")

    x_true_flat = x_true.reshape(-1, 7)
    x_enc_flat  = x_enc .reshape(-1, 7)
    x_knn_flat  = x_knn .reshape(-1, 7)

    phys_results = {}
    for pname, j, units, arc_pub in zip(phys_names, phys_idx, phys_units, arc_rmse):
        knn_rmse = np.sqrt(np.mean((x_knn_flat[:,j] - x_true_flat[:,j])**2))
        enc_rmse = np.sqrt(np.mean((x_enc_flat[:,j] - x_true_flat[:,j])**2))
        knn_r2   = r2(x_knn_flat[:,j], x_true_flat[:,j])
        enc_r2   = r2(x_enc_flat[:,j], x_true_flat[:,j])
        phys_results[pname] = {
            "knn_rmse": float(knn_rmse), "enc_rmse": float(enc_rmse),
            "knn_r2":   float(knn_r2),   "enc_r2":   float(enc_r2),
            "arc_pub_rmse": arc_pub,
        }
        arc_pub_str = f"{arc_pub:8.4f}" if arc_pub is not None else "    n/a"
        print(f"  {pname:8s}  {knn_rmse:10.4f}  {enc_rmse:10.4f}  {units:8s}  "
              f"{knn_r2:7.4f}  {enc_r2:7.4f}  {arc_pub_str}")

    print(f"\n  Note: ARC_pub = ARC's published RMSE on REAL data (harder than synthetic).")
    print(f"        n/a = no published ARC benchmark (no field measurements at MNI Germany).")

    # Save results
    out = {"param_results": results, "phys_results": phys_results,
           "n_samples": n_samples, "n_library": n_library}
    with open(os.path.join(out_dir, "arc_knn_comparison.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_dir}/arc_knn_comparison.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--n_samples",  type=int, default=200,
                        help="Number of test pixels (200 takes ~30 min)")
    parser.add_argument("--n_library",  type=int, default=131072,
                        help="Sobol library size (131072=2^17, fast; 524288=2^19, better)")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    run_comparison(
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        n_library=args.n_library,
        device=args.device,
    )


if __name__ == "__main__":
    main()
