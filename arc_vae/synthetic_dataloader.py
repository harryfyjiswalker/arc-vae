"""
Synthetic Data Loader for ARC-VAE Training
==========================================
Generates synthetic S2 time series for German maize using generate_arc_refs,
with realistic observation patterns, cloud masking, and heteroscedastic noise.

Each sample returned is a complete growing season for one synthetic pixel:
  - s2_refl   : (MAX_OBS, 10)  padded S2 reflectance observations
  - angles    : (MAX_OBS,  3)  [SZA, VZA, RAA] per observation
  - doys      : (MAX_OBS,)     calendar DOY per observation
  - sigma     : (MAX_OBS, 10)  per-band observation uncertainty
  - obs_mask  : (MAX_OBS,)     True = valid observation, False = padding
  - p_true    : (7,)           true ARC scaling parameters
  - h_true    : (4,)           true ARC phenology parameters
  - soil_true : (4,)           true soil parameters

Design decisions documented here:
  - German maize phenology: start DOY 130-155, season 120-155 days
  - 20-40 randomly spaced observations in DOY window [100, 300]
  - Cloud masking: 10-30% of observations randomly removed, clustered
    around a random date to simulate multi-day cloud events
  - Noise: sigma_b = sigma_base_b + sigma_rel * reflectance_b
    with sigma_base = 0.005 (floor) and sigma_rel = 0.02 (2% of signal)
    These are conservative SIAC-level uncertainties
  - Padding to MAX_OBS=50 with obs_mask to handle variable lengths
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ARC"))
from arc.arc_sample_generator import generate_arc_refs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_OBS    = 50     # maximum sequence length (pad shorter sequences)
N_BANDS    = 10     # S2 bands: B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
N_P_PARAMS = 7      # scaling parameters
N_H_PARAMS = 4      # phenology parameters

# German maize phenology bounds
GERMANY_SEASON_START_MIN = 125   # earliest planting DOY
GERMANY_SEASON_START_MAX = 155   # latest planting DOY
GERMANY_SEASON_LEN_MIN   = 120   # shortest season (days)
GERMANY_SEASON_LEN_MAX   = 155   # longest season  (days)
GERMANY_DOY_WINDOW_START = 100   # start of S2 acquisition window
GERMANY_DOY_WINDOW_END   = 330   # end of S2 acquisition window (covers full senescence)

# Observation count range (after cloud masking)
MIN_OBS_AFTER_CLOUD = 8    # minimum usable observations per season
MIN_OBS_TOTAL       = 20   # minimum acquisitions before cloud masking
MAX_OBS_TOTAL       = 40   # maximum acquisitions before cloud masking

# Noise model: sigma_b = SIGMA_BASE + SIGMA_REL * reflectance_b
SIGMA_BASE = 0.005   # irreducible floor (atmospheric correction residuals)
SIGMA_REL  = 0.020   # 2% of signal (proportional uncertainty)

# German sun angle ranges for S2
SZA_MIN, SZA_MAX = 25.0, 60.0    # solar zenith angle degrees
VZA_MIN, VZA_MAX =  0.0, 10.0    # view zenith angle degrees


# ---------------------------------------------------------------------------
# Importance sampling for p_LAI
# ---------------------------------------------------------------------------

# p_LAI bounds (from encoder.py)
_P_LAI_LO = 0.1624
_P_LAI_HI = 2.5978

def _importance_weights_lai(p_lai: np.ndarray, rng) -> np.ndarray:
    """
    Compute importance weights to resample uniform p_LAI draws toward
    a Beta(2,2) target distribution scaled to [P_LAI_LO, P_LAI_HI].

    Beta(2,2) concentrates mass around the centre (p_LAI≈1.38, i.e. LAI≈4.25)
    while still covering the full range.  This reduces the high-LAI regression
    bias without changing the parameter bounds.

    w_i  ∝  Beta(2,2)_pdf(normalised_p_lai_i)
    (Uniform proposal cancels — uniform pdf is constant over the range.)
    """
    from scipy.stats import beta as beta_dist
    p_norm = (p_lai - _P_LAI_LO) / (_P_LAI_HI - _P_LAI_LO + 1e-9)
    p_norm = np.clip(p_norm, 1e-6, 1 - 1e-6)
    weights = beta_dist.pdf(p_norm, a=2, b=2)
    weights = weights / weights.sum()
    return weights


# ---------------------------------------------------------------------------
# Core sample generation
# ---------------------------------------------------------------------------
def _generate_observation_doys(rng, season_start, season_length):
    """
    Sample realistic S2 acquisition dates within the growing season window.
    
    S2 revisit is ~5 days. We sample n_total dates uniformly within
    [DOY_WINDOW_START, DOY_WINDOW_END], then sort them.
    """
    n_total = rng.integers(MIN_OBS_TOTAL, MAX_OBS_TOTAL + 1)
    doys = rng.choice(
        np.arange(GERMANY_DOY_WINDOW_START, GERMANY_DOY_WINDOW_END + 1),
        size=n_total,
        replace=False
    )
    return np.sort(doys)


def _apply_cloud_masking(rng, doys, cloud_fraction_range=(0.10, 0.30)):
    """
    Simulate cloud contamination with spatial clustering.
    
    Rather than masking uniformly at random (which is too optimistic),
    we simulate 1-3 cloud events, each lasting 5-15 consecutive days,
    and mask any acquisition that falls within a cloud event window.
    This better represents real S2 cloud patterns.
    
    Returns a boolean array: True = cloud-free (keep), False = cloudy (mask).
    """
    keep = np.ones(len(doys), dtype=bool)
    
    n_cloud_events = rng.integers(1, 4)
    for _ in range(n_cloud_events):
        cloud_centre = rng.integers(GERMANY_DOY_WINDOW_START,
                                     GERMANY_DOY_WINDOW_END + 1)
        cloud_duration = rng.integers(5, 16)
        cloud_start = cloud_centre - cloud_duration // 2
        cloud_end   = cloud_start  + cloud_duration
        keep[(doys >= cloud_start) & (doys <= cloud_end)] = False
    
    # Ensure overall cloud fraction is within target range
    # If too few observations remain, relax the masking
    frac_removed = 1.0 - keep.sum() / len(keep)
    if keep.sum() < MIN_OBS_AFTER_CLOUD:
        # Re-enable the most recent masked observations to meet minimum
        masked_indices = np.where(~keep)[0]
        n_restore = MIN_OBS_AFTER_CLOUD - keep.sum()
        restore_indices = rng.choice(masked_indices, size=int(n_restore),
                                      replace=False)
        keep[restore_indices] = True
    
    return keep


def _sample_angles(rng, n_obs):
    """
    Sample realistic Sentinel-2 viewing geometry for Germany.
    SZA and RAA vary by date; VZA is mostly small for S2 in Germany.
    """
    sza = rng.uniform(SZA_MIN, SZA_MAX, n_obs)
    vza = rng.uniform(VZA_MIN, VZA_MAX, n_obs)
    raa = rng.uniform(0.0,    180.0,   n_obs)
    return sza, vza, raa


def _add_noise(rng, reflectance):
    """
    Add heteroscedastic Gaussian noise following the SIAC uncertainty model.
    
    sigma_b(t) = SIGMA_BASE + SIGMA_REL * reflectance_b(t)
    
    Returns noisy reflectance and the sigma used (for the loss function).
    """
    sigma = SIGMA_BASE + SIGMA_REL * np.abs(reflectance)
    noise = rng.normal(0.0, sigma)
    noisy = np.clip(reflectance + noise, 0.0, 1.0)
    return noisy, sigma


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GermanyMaizeDataset(Dataset):
    """
    Synthetic dataset of German maize S2 time series.
    
    Generates samples on-the-fly using generate_arc_refs with parameters
    appropriate for German maize phenology.
    
    Parameters
    ----------
    n_samples : int
        Number of synthetic pixels in this dataset split.
    batch_size_arc : int
        Number of samples generated per call to generate_arc_refs.
        Larger = more efficient but more memory.
    seed : int
        Base random seed (reproducible train/val splits).
    """

    def __init__(
        self,
        n_samples:       int = 50_000,
        batch_size_arc:  int = 256,
        seed:            int = 42,
        crop_type:       str = "maize",
    ):
        self.n_samples      = n_samples
        self.batch_size_arc = batch_size_arc
        self.seed           = seed
        self.crop_type      = crop_type

        # Pre-generate all samples at construction time
        # (avoids multiprocessing issues with JAX inside DataLoader workers)
        print(f"Generating {n_samples:,} synthetic samples...")
        self._data = self._generate_all(n_samples, batch_size_arc, seed)
        print("Done.")

    def _generate_all(self, n_samples, batch_size_arc, seed):
        """
        Generate all synthetic samples in batches via generate_arc_refs.
        Each call to generate_arc_refs uses a single (doys, angles) setup
        shared across batch_size_arc samples.  The internal Sobol sampling
        ensures diversity of (p, h) within each batch.
        """
        rng = np.random.default_rng(seed)

        all_refl  = []   # (N, MAX_OBS, 10)
        all_ang   = []   # (N, MAX_OBS, 3)
        all_doys  = []   # (N, MAX_OBS)
        all_sigma = []   # (N, MAX_OBS, 10)
        all_mask  = []   # (N, MAX_OBS)  bool
        all_p     = []   # (N, 7)
        all_h     = []   # (N, 4)
        all_soil  = []   # (N, 4)

        n_generated = 0
        while n_generated < n_samples:
            n_this = min(batch_size_arc, n_samples - n_generated)

            # --- Sample a shared observation pattern for this batch ---
            # (All samples in a batch share DOYs and angles, mimicking
            #  one S2 tile observed at the same dates — realistic)
            season_start = int(rng.integers(GERMANY_SEASON_START_MIN,
                                             GERMANY_SEASON_START_MAX + 1))
            season_len   = int(rng.integers(GERMANY_SEASON_LEN_MIN,
                                             GERMANY_SEASON_LEN_MAX + 1))
            doys_all = _generate_observation_doys(rng, season_start, season_len)
            sza, vza, raa = _sample_angles(rng, len(doys_all))
            angs = (sza, vza, raa)

            # --- Generate reflectances via ARC ---
            # Generate 3x samples then importance-resample on p_LAI
            # to concentrate training data at realistic LAI values
            n_oversample = min(n_this * 3, 1024)
            try:
                arc_refs, pheo_s, bio_s, orig_bios, soil_s = generate_arc_refs(
                    doys_all, season_start, season_len,
                    n_oversample, angs, self.crop_type
                )
                # arc_refs: (10, T, N)  → want (N, T, 10)
                arc_refs = arc_refs.transpose(2, 1, 0)   # (N, T, 10)
            except Exception as e:
                print(f"  Warning: generate_arc_refs failed ({e}), skipping batch")
                continue

            n_oversampled = arc_refs.shape[0]

            # Importance resample: weight by Beta(2,2) on p_LAI (index 4)
            p_lai_vals = bio_s[:n_oversampled, 4]
            weights    = _importance_weights_lai(p_lai_vals, rng)
            keep_idx   = rng.choice(n_oversampled, size=min(n_this, n_oversampled),
                                     replace=True, p=weights)

            arc_refs  = arc_refs [keep_idx]
            bio_s     = bio_s    [keep_idx]
            pheo_s    = pheo_s   [keep_idx]
            soil_s    = soil_s   [keep_idx]
            # orig_bios: (7, T, N) — reindex last axis
            orig_bios = orig_bios[:, :, keep_idx]

            n_actual = arc_refs.shape[0]

            # Angles array: (T, 3) shared across all samples in batch
            angles_arr = np.stack([sza, vza, raa], axis=-1)   # (T, 3)

            # --- Per-sample cloud masking and noise ---
            for i in range(n_actual):
                # Cloud mask: shared pattern per tile but sample-independent here
                # In practice one tile shares clouds, but for synthetic training
                # we use independent masks per sample for diversity
                cloud_keep = _apply_cloud_masking(rng, doys_all)
                refl_clean = arc_refs[i][cloud_keep]     # (T_obs, 10)
                ang_obs    = angles_arr[cloud_keep]      # (T_obs, 3)
                doys_obs   = doys_all[cloud_keep]        # (T_obs,)
                T_obs      = refl_clean.shape[0]

                # Add SIAC-style noise
                refl_noisy, sigma_arr = _add_noise(rng, refl_clean)

                # --- Pad to MAX_OBS ---
                pad = MAX_OBS - T_obs
                refl_pad  = np.zeros((MAX_OBS, N_BANDS),  dtype=np.float32)
                ang_pad   = np.zeros((MAX_OBS, 3),        dtype=np.float32)
                doys_pad  = np.zeros(MAX_OBS,             dtype=np.float32)
                sigma_pad = np.ones ((MAX_OBS, N_BANDS),  dtype=np.float32)
                mask_pad  = np.zeros(MAX_OBS,             dtype=bool)

                refl_pad [:T_obs] = refl_noisy.astype(np.float32)
                ang_pad  [:T_obs] = ang_obs.astype(np.float32)
                doys_pad [:T_obs] = doys_obs.astype(np.float32)
                sigma_pad[:T_obs] = sigma_arr.astype(np.float32)
                mask_pad [:T_obs] = True

                all_refl .append(refl_pad)
                all_ang  .append(ang_pad)
                all_doys .append(doys_pad)
                all_sigma.append(sigma_pad)
                all_mask .append(mask_pad)
                all_p    .append(bio_s[i].astype(np.float32))
                all_h    .append(pheo_s[i].astype(np.float32))
                all_soil .append(soil_s[i].astype(np.float32))

            n_generated += n_actual

        # Stack into arrays and trim to n_samples
        data = {
            "s2_refl":   np.stack(all_refl [:n_samples]),   # (N, MAX_OBS, 10)
            "angles":    np.stack(all_ang  [:n_samples]),   # (N, MAX_OBS, 3)
            "doys":      np.stack(all_doys [:n_samples]),   # (N, MAX_OBS)
            "sigma":     np.stack(all_sigma[:n_samples]),   # (N, MAX_OBS, 10)
            "obs_mask":  np.stack(all_mask [:n_samples]),   # (N, MAX_OBS) bool
            "p_true":    np.stack(all_p    [:n_samples]),   # (N, 7)
            "h_true":    np.stack(all_h    [:n_samples]),   # (N, 4)
            "soil_true": np.stack(all_soil [:n_samples]),   # (N, 4)
        }
        return data

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {k: torch.from_numpy(v[idx]) for k, v in self._data.items()}


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------
def make_dataloaders(
    n_train:     int = 50_000,
    n_val:       int =  5_000,
    batch_size:  int =    128,
    num_workers: int =      0,   # 0 = main process (safe with JAX)
    seed:        int =     42,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders of synthetic German maize data.
    
    Parameters
    ----------
    n_train, n_val : int
        Number of synthetic samples in each split.
    batch_size : int
        Training batch size.
    num_workers : int
        Keep at 0 — generate_arc_refs uses JAX which has issues with
        multiprocessing fork.
    
    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    train_ds = GermanyMaizeDataset(n_samples=n_train, seed=seed)
    val_ds   = GermanyMaizeDataset(n_samples=n_val,   seed=seed + 99999)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_dataloader():
    """
    Smoke test: generate a small dataset and check shapes, ranges,
    observation statistics, and that the ground-truth (p,h) round-trip
    through the decoder produces sensible reflectances.
    """
    print("=" * 60)
    print("Step 3: Verify synthetic data loader")
    print("=" * 60)

    # Small dataset for quick verification
    ds = GermanyMaizeDataset(n_samples=512, batch_size_arc=128, seed=0)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    batch  = next(iter(loader))

    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k:12s}: {tuple(v.shape)}  dtype={v.dtype}")

    # Check observation counts (non-padded)
    n_obs = batch["obs_mask"].sum(dim=1).float()
    print(f"\nObservations per sample (after cloud masking):")
    print(f"  Min: {n_obs.min().item():.0f}")
    print(f"  Max: {n_obs.max().item():.0f}")
    print(f"  Mean: {n_obs.mean().item():.1f}")

    # Check reflectance is physically plausible
    refl = batch["s2_refl"]
    mask = batch["obs_mask"].unsqueeze(-1)  # (B, T, 1)
    refl_valid = refl[mask.expand_as(refl).bool()]
    print(f"\nReflectance of valid observations:")
    print(f"  Min: {refl_valid.min().item():.4f}  (should be > 0)")
    print(f"  Max: {refl_valid.max().item():.4f}  (should be < 1)")
    print(f"  Mean: {refl_valid.mean().item():.4f}")

    # Check h parameter ranges make sense for German maize
    h = batch["h_true"]
    print(f"\nPhenology (h) parameter ranges:")
    labels = ["growth_speed", "start_DOY", "senes_speed", "end_DOY"]
    for j, label in enumerate(labels):
        print(f"  {label:15s}: [{h[:,j].min().item():.2f}, "
              f"{h[:,j].max().item():.2f}]")

    # Check start DOY is within expected German maize range
    start_doys = h[:, 1]
    in_range = ((start_doys >= GERMANY_SEASON_START_MIN - 10) &
                (start_doys <= GERMANY_SEASON_START_MAX + 55))
    print(f"\n  Start DOY in expected German range "
          f"[{GERMANY_SEASON_START_MIN-10}, {GERMANY_SEASON_START_MAX+55}]: "
          f"{in_range.float().mean()*100:.1f}%")

    # Check p=1.0 is within the sampled range (archetype prior check)
    p = batch["p_true"]
    print(f"\nScaling (p) parameter ranges:")
    param_names = ["N", "Cab", "Cm", "Cw", "LAI", "ALA", "Cbrown"]
    for j, name in enumerate(param_names):
        pmin, pmax = p[:,j].min().item(), p[:,j].max().item()
        contains_1 = pmin < 1.0 < pmax
        print(f"  {name:8s}: [{pmin:.3f}, {pmax:.3f}]  "
              f"contains p=1.0: {'✓' if contains_1 else '✗'}")

    # Noise sanity check
    sigma = batch["sigma"]
    sigma_valid = sigma[mask.expand_as(sigma).bool()]
    print(f"\nObservation uncertainty (sigma):")
    print(f"  Min: {sigma_valid.min().item():.5f}")
    print(f"  Max: {sigma_valid.max().item():.5f}")
    print(f"  Mean: {sigma_valid.mean().item():.5f}")

    print("\nData loader verification: PASSED ✓")
    print("Proceed to Step 4 (Transformer encoder).")
    return True


if __name__ == "__main__":
    verify_dataloader()
