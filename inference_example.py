"""
ARC-VAE Field Inference
=======================
Mirrors the interface of arc.arc_field() — give it a GeoJSON file and
a date range, and it fetches Sentinel-2 data from AWS, runs the encoder
on every pixel in the field, and returns the biophysical parameter maps
and seasonal trajectories.

Quick start
-----------
    python inference_example.py --geojson my_field.geojson \
                                 --start 2018-04-01 \
                                 --end   2018-10-20

Without a GeoJSON, a synthetic single-pixel demo runs automatically:

    python inference_example.py

Requirements (beyond standard libraries)
-----------------------------------------
    pip install torch numpy matplotlib scipy
    pip install https://github.com/MarcYin/ARC/archive/refs/heads/main.zip
"""

import os, sys, argparse, json, warnings, subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────────
# BASE is the root of this repository (the folder containing this file).
# Works after a clean git clone, in Colab, or when run from any location.
BASE = os.path.dirname(os.path.abspath(__file__))

# arc_vae is a local package — add it to the Python path so its modules
# can be imported without installing it.
sys.path.insert(0, os.path.join(BASE, 'arc_vae'))

# ARC is an external dependency installed via pip (see requirements.txt).
# If it is not installed yet, install it automatically from GitHub.
# This means the script works straight after 'git clone' with no manual setup.
try:
    import eof   # eof is part of the ARC package — if this imports, ARC is ready
except ImportError:
    print("ARC not found — installing from GitHub (one-time, ~30 seconds)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "https://github.com/MarcYin/ARC/archive/refs/heads/main.zip"],
        check=True)
    import eof

from encoder import ARCVAEEncoder
from archetype_decoder import ARCDecoder, double_logistic

# ── Checkpoint location ───────────────────────────────────────────────────────
CKPT_PATH = os.path.join(
    BASE, 'outputs',
    'maize_d128_L4_e20+30_sup10.0_20260518_1637',
    'checkpoint_best.pt')

MAX_OBS = 50    # encoder padding length — do not change
N_BANDS = 10    # S2 bands used: B2 B3 B4 B5 B6 B7 B8 B8A B11 B12

# Mapping from latent vector index to physical meaning
PARAM_NAMES  = ['N', 'Cab', 'Cm', 'Cw', 'LAI', 'ALA', 'Cbrown',
                'k_growth', 'h_start', 'k_senes', 'h_end']
PARAM_UNITS  = ['—', 'µg/cm²', 'g/cm²', 'g/cm²', 'm²/m²',
                '°', '—', 'day⁻¹', 'DOY', 'day⁻¹', 'DOY']
LAI_IDX      = 4   # index 4 in the 7-parameter biophysical vector


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_encoder(ckpt_path: str, device: str = 'cpu') -> ARCVAEEncoder:
    """Load the trained encoder from a checkpoint file."""
    encoder = ARCVAEEncoder(d_model=128, n_layers=4, n_heads=4)
    ckpt    = torch.load(ckpt_path, map_location=device)
    state   = ckpt.get('model_state', ckpt)
    # The checkpoint stores the full ARCVAE; strip the 'encoder.' prefix
    enc_state = {k[len('encoder.'):]: v
                 for k, v in state.items() if k.startswith('encoder.')}
    encoder.load_state_dict(enc_state)
    encoder.eval()
    encoder.to(device)
    return encoder


def load_decoder(crop_type: str = 'maize') -> ARCDecoder:
    """Load the frozen archetype decoder (no weights to train)."""
    decoder = ARCDecoder(crop_type=crop_type)
    decoder.eval()
    return decoder


# ─────────────────────────────────────────────────────────────────────────────
# Encoder forward pass (batched)
# ─────────────────────────────────────────────────────────────────────────────

def encode_pixels(
    encoder:  ARCVAEEncoder,
    refl:     np.ndarray,   # (N_pixels, T, 10) reflectance
    angs:     np.ndarray,   # (N_pixels, T, 3)  sun/sensor angles
    doys:     np.ndarray,   # (T,) day-of-year — shared across all pixels
    device:   str = 'cpu',
    batch_sz: int = 256,
) -> tuple:
    """
    Run the encoder on a batch of pixels from the same field.

    All pixels share the same observation dates (same satellite overpass),
    so `doys` is a single (T,) array broadcast across all pixels. Each
    pixel may have missing data (NaN) on certain dates; these are replaced
    with the field mean and flagged in the mask so the encoder ignores them.

    Returns:
        mu    (N_pixels, 11) — posterior mean for each parameter
        sigma (N_pixels, 11) — posterior standard deviation (uncertainty)
    """
    N, T, _ = refl.shape
    assert T <= MAX_OBS, f"Too many observations ({T}); max is {MAX_OBS}"

    # Pre-allocate padded input arrays (zeros outside the valid window)
    rp = np.zeros((N, MAX_OBS, N_BANDS), np.float32)
    ap = np.zeros((N, MAX_OBS, 3),       np.float32)
    dp = np.zeros((N, MAX_OBS),          np.float32)
    mp = np.zeros((N, MAX_OBS),          bool)

    # Fill the first T slots; mark positions where all bands are finite
    rp[:, :T, :] = np.nan_to_num(refl, nan=0.0).astype(np.float32)
    ap[:, :T, :] = angs.astype(np.float32)
    dp[:, :T]    = doys.astype(np.float32)
    # A pixel-date is valid if all 10 bands are finite
    mp[:, :T]    = np.isfinite(refl).all(axis=-1)

    # Process in batches to avoid GPU memory overflow on large fields
    mu_list, sigma_list = [], []
    for start in range(0, N, batch_sz):
        end = min(start + batch_sz, N)
        with torch.no_grad():
            mu_b, sigma_b = encoder(
                torch.from_numpy(rp[start:end]).to(device),
                torch.from_numpy(ap[start:end]).to(device),
                torch.from_numpy(dp[start:end]).to(device),
                torch.from_numpy(mp[start:end]).to(device),
            )
        mu_list.append(mu_b.cpu().numpy())
        sigma_list.append(sigma_b.cpu().numpy())

    return np.concatenate(mu_list), np.concatenate(sigma_list)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def get_lai_trajectory(
    decoder:    ARCDecoder,
    mu:         np.ndarray,    # (N_pixels, 11) or (11,) for single pixel
    eval_doys:  np.ndarray,    # DOYs at which to evaluate the trajectory
    batch_sz:   int = 512,
) -> np.ndarray:
    """
    Reconstruct the LAI trajectory for each pixel over `eval_doys`.

    Uses the archetype double-logistic model to convert the predicted
    phenological parameters (h) and biophysical scaling (p_LAI) into
    physical LAI values at any requested day of year.

    Returns:
        (N_pixels, len(eval_doys)) array of LAI in m² m⁻²
    """
    if mu.ndim == 1:
        mu = mu[np.newaxis, :]   # single pixel → add batch dimension

    N  = mu.shape[0]
    T  = len(eval_doys)
    dt = torch.tensor(eval_doys, dtype=torch.float32)

    lai_out = np.zeros((N, T), dtype=np.float32)

    for start in range(0, N, batch_sz):
        end = min(start + batch_sz, N)
        p   = torch.from_numpy(mu[start:end, :7].astype(np.float32))
        h   = torch.from_numpy(mu[start:end, 7:].astype(np.float32))
        nb  = end - start

        with torch.no_grad():
            # Full double-logistic trajectory over all 365 days
            ta  = torch.arange(365, dtype=torch.float32)
            L   = double_logistic(h, ta)
            Lmn = L.amin(1, keepdim=True)
            Lmx = L.amax(1, keepdim=True)
            Ln  = (L - Lmn) / (Lmx - Lmn + 1e-8)

            # Force the curve to be monotone (up then down) to avoid
            # the symmetric ambiguity around the seasonal peak
            k1, t1 = h[:, 0:1], h[:, 1:2]
            k2, t2 = h[:, 2:3], h[:, 3:4]
            tp  = (k2 * t1 + k1 * t2) / (k1 + k2 + 1e-8)
            sm  = torch.sigmoid(20 * (ta.unsqueeze(0) - tp))
            Lm  = (1 - sm) * Ln + sm * (2 - Ln)

            # Sample at the requested evaluation DOYs
            idx = (dt.long() - 1).clamp(0, 364)
            Lo  = Lm[:, idx]

            # Convert trajectory position to archetype parameter values
            tau = ARCDecoder._interp1d(
                Lo.reshape(-1), decoder.v_grid, decoder.doy_grid
            ).reshape(nb, T)
            a   = decoder._lookup_meds(tau)   # (nb, T, 7)

            # Scale by p and clip to physical bounds
            x   = torch.clamp(p.unsqueeze(1) * a, decoder.lo, decoder.hi)

        lai_out[start:end] = x[:, :, LAI_IDX].numpy()

    return lai_out   # (N_pixels, T)


# ─────────────────────────────────────────────────────────────────────────────
# S2 data fetching and preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_preprocess(
    geojson_path: str,
    start_date:   str,
    end_date:     str,
    data_folder:  str,
    data_source:  str = 'aws',
) -> tuple:
    """
    Fetch Sentinel-2 data for a field and return cloud-free, deduplicated arrays.

    Returns:
        udoys   (T,)            — unique cloud-free day-of-year values
        refl_u  (N_pixels, T, 10) — reflectance per pixel per date
        angs_u  (T, 3)          — sun/sensor angles (field-level average)
        mask    (H, W)          — True where pixel is outside the field
    """
    try:
        import eof
    except ImportError:
        raise ImportError(
            "The eof package is required for real S2 data.\n"
            "Install with:\n"
            "  pip install https://github.com/MarcYin/ARC/"
            "archive/refs/heads/main.zip")

    print(f"Fetching S2 data ({start_date} → {end_date}) from {data_source}...")
    s2 = eof.get_s2_data(
        start_date=start_date,
        end_date=end_date,
        geojson_path=geojson_path,
        data_folder=data_folder,
        source=data_source)

    mask   = s2.mask                     # (H, W) bool — True = outside field
    n_pix  = int((~mask).sum())          # number of valid pixels
    doys   = s2.doys.astype(float)       # (T_all,) day-of-year per acquisition

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # s2.reflectance shape: (T_all, 10, N_pixels)
        refl_all = s2.reflectance[:, :, ~mask]   # (T_all, 10, n_pix)
        angs_all = s2.angles.T                   # (T_all, 3)

    # A date is "field-clear" if the spatial mean of all bands is finite
    refl_mean = np.nanmean(refl_all, axis=-1)    # (T_all, 10)
    cf_dates  = np.isfinite(refl_mean).all(axis=1) & (refl_mean >= 0).all(axis=1)

    refl_cf  = refl_all[cf_dates]   # (T_cf, 10, n_pix)
    angs_cf  = angs_all[cf_dates]
    doys_cf  = doys[cf_dates]

    # Deduplicate: average S2A and S2B overpasses on the same calendar day
    udoys = np.unique(doys_cf.astype(int))
    # refl_u shape: (n_pix, T, 10) — transposed for encoder
    refl_u = np.stack([
        np.nanmean(refl_cf[doys_cf.astype(int) == d], axis=0).T
        for d in udoys
    ], axis=1)   # (n_pix, T, 10)
    angs_u = np.array([
        angs_cf[doys_cf.astype(int) == d].mean(0)
        for d in udoys
    ])            # (T, 3)

    print(f"  {n_pix:,} valid pixels  |  "
          f"{len(udoys)} unique cloud-free dates  "
          f"(DOY {udoys.min()}–{udoys.max()})")

    return udoys, refl_u, angs_u, mask


# ─────────────────────────────────────────────────────────────────────────────
# Main field inference function  (mirrors arc.arc_field)
# ─────────────────────────────────────────────────────────────────────────────

def run_field(
    geojson_path:    str,
    start_date:      str,
    end_date:        str,
    crop_type:       str = 'maize',
    checkpoint_path: str = None,
    data_folder:     str = None,
    eval_doys:       np.ndarray = None,
    plot:            bool = True,
    data_source:     str = 'aws',
    device:          str = 'cpu',
) -> tuple:
    """
    Run ARC-VAE inference on a field defined by a GeoJSON polygon.

    Fetches Sentinel-2 data, runs the Transformer VAE encoder on every
    pixel in the field, and reconstructs the full seasonal LAI trajectory.

    Args:
        geojson_path:    path to a GeoJSON file defining the field boundary
        start_date:      first date to fetch S2 data, format 'YYYY-MM-DD'
        end_date:        last  date to fetch S2 data, format 'YYYY-MM-DD'
        crop_type:       archetype to use — currently only 'maize' is trained
        checkpoint_path: path to checkpoint_best.pt; defaults to the
                         pre-trained weights included in this repository
        data_folder:     folder to cache downloaded S2 data; defaults to
                         a subfolder inside the system temp directory
        eval_doys:       DOYs at which to reconstruct the LAI trajectory;
                         defaults to every 3rd day across the growing season
        plot:            if True, show LAI-over-time and LAI-map figures
        data_source:     'aws' (default, no credentials needed) or 'gee'
        device:          'cpu' or 'cuda'

    Returns:
        doys      (T,)              — unique cloud-free S2 acquisition DOYs
        mu        (N_pixels, 11)    — posterior mean for all 11 parameters
        sigma     (N_pixels, 11)    — posterior std (uncertainty) per parameter
        lai_traj  (N_pixels, T_eval)— reconstructed LAI trajectory
        mask      (H, W)            — True where pixel is outside the field
    """
    import tempfile

    if checkpoint_path is None:
        checkpoint_path = CKPT_PATH
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at:\n  {checkpoint_path}\n"
            "Download checkpoint_best.pt from the repository and place it "
            "at the path above, or pass checkpoint_path= explicitly.")

    if data_folder is None:
        data_folder = os.path.join(
            tempfile.gettempdir(),
            'arc_vae_s2_' + os.path.splitext(os.path.basename(geojson_path))[0])
    os.makedirs(data_folder, exist_ok=True)

    if eval_doys is None:
        eval_doys = np.arange(80, 310, 3)

    # Load models
    print(f"Loading encoder from {checkpoint_path}")
    encoder = load_encoder(checkpoint_path, device=device)
    decoder = load_decoder(crop_type=crop_type)
    print(f"Encoder: {sum(p.numel() for p in encoder.parameters()):,} params\n")

    # Fetch and preprocess S2 data
    doys, refl_u, angs_u, mask = fetch_and_preprocess(
        geojson_path, start_date, end_date, data_folder, data_source)

    # Broadcast angles to per-pixel (all pixels share same angles)
    N_pix = refl_u.shape[0]
    T     = len(doys)
    angs_broadcast = np.tile(angs_u[np.newaxis, :, :], (N_pix, 1, 1))  # (N, T, 3)

    # Run encoder on all pixels
    print(f"Running encoder on {N_pix:,} pixels × {T} dates...")
    mu, sigma = encode_pixels(encoder, refl_u, angs_broadcast,
                               doys.astype(float), device=device)

    # Reconstruct LAI trajectory at dense eval DOYs
    print("Reconstructing seasonal trajectories...")
    lai_traj = get_lai_trajectory(decoder, mu, eval_doys)

    print(f"\nDone. Summary:")
    print(f"  Median ĥ_end   = {np.median(mu[:, 10]):.0f} DOY")
    print(f"  Median ĥ_start = {np.median(mu[:,  8]):.0f} DOY")
    print(f"  Median peak LAI = {lai_traj.max(axis=1).mean():.2f} m² m⁻²")

    if plot:
        plot_lai_over_time(eval_doys, lai_traj)
        plot_lai_maps(doys, refl_u, mu, decoder, mask)

    return doys, mu, sigma, lai_traj, mask


# ─────────────────────────────────────────────────────────────────────────────
# Plotting  (mirrors ARC's plot functions)
# ─────────────────────────────────────────────────────────────────────────────

def plot_lai_over_time(eval_doys: np.ndarray, lai_traj: np.ndarray,
                       sample_step: int = 50):
    """
    Plot LAI trajectories for a random subsample of pixels over time.
    `sample_step` controls how many pixels are drawn (every Nth pixel).
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(eval_doys, lai_traj[::sample_step].T,
            '-', lw=1.0, alpha=0.4, color='#2166ac')
    # Bold median line
    ax.plot(eval_doys, np.median(lai_traj, axis=0),
            '-', lw=2.5, color='#1a1a1a', label='Field median')
    ax.set_ylabel('LAI (m² m⁻²)')
    ax.set_xlabel('Day of year')
    ax.set_title('Reconstructed LAI trajectories — all field pixels')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    plt.show()


def plot_lai_maps(doys: np.ndarray, refl_u: np.ndarray,
                  mu: np.ndarray, decoder: ARCDecoder,
                  mask: np.ndarray):
    """
    Plot a LAI map for each cloud-free S2 acquisition date.
    Layout mirrors ARC's plot_lai_maps function.
    """
    N_pix = refl_u.shape[0]
    T     = len(doys)

    # Get LAI at each acquisition DOY
    lai_at_doys = get_lai_trajectory(decoder, mu, doys)  # (N_pix, T)

    ncols = min(5, T)
    nrows = int(np.ceil(T / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4 * ncols, 3.5 * nrows))
    axs = np.array(axs).ravel()

    H, W = mask.shape
    for i in range(T):
        lai_map = np.full((H, W), np.nan)
        lai_map[~mask] = lai_at_doys[:, i]
        im = axs[i].imshow(lai_map, vmin=0, vmax=7, cmap='YlGn')
        plt.colorbar(im, ax=axs[i], shrink=0.8, label='LAI (m² m⁻²)')
        axs[i].set_title(f'DOY {doys[i]}', fontsize=9)
        axs[i].axis('off')

    for i in range(T, len(axs)):
        axs[i].axis('off')

    fig.suptitle('ARC-VAE LAI maps — field pixels', fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_parameter_map(mu: np.ndarray, mask: np.ndarray,
                       param_idx: int = 4, title: str = None):
    """
    Plot a spatial map of any of the 11 predicted parameters.

    Example — plot the predicted harvest DOY across the field:
        plot_parameter_map(mu, mask, param_idx=10, title='Predicted harvest DOY')
    """
    H, W = mask.shape
    param_map = np.full((H, W), np.nan)
    param_map[~mask] = mu[:, param_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(param_map, cmap='RdYlGn')
    plt.colorbar(im, ax=ax,
                 label=f'{PARAM_NAMES[param_idx]} ({PARAM_UNITS[param_idx]})')
    ax.set_title(title or PARAM_NAMES[param_idx])
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic demo  (no data required — runs instantly)
# ─────────────────────────────────────────────────────────────────────────────

def demo_synthetic(checkpoint_path: str, device: str = 'cpu'):
    """
    Sanity check: runs the encoder on a small set of synthetic maize pixels
    and plots the reconstructed LAI trajectory. No internet or S2 data needed.
    """
    print("\n── Synthetic demo (6 simulated pixels, maize) ──\n")

    encoder = load_encoder(checkpoint_path, device=device)
    decoder = load_decoder(crop_type='maize')

    # Six observations at typical German maize growing-season DOYs
    T       = 6
    obs_doys = np.array([167., 182., 197., 202., 212., 232.])
    # Approximate S2 reflectances for a healthy maize canopy at peak
    base_refl = np.array([0.04, 0.05, 0.04, 0.03, 0.08, 0.30,
                          0.37, 0.40, 0.42, 0.11])
    # Add small per-pixel variation to simulate spatial heterogeneity
    rng       = np.random.default_rng(42)
    N_demo    = 20
    refl_demo = np.tile(base_refl, (N_demo, T, 1)) + \
                rng.normal(0, 0.01, (N_demo, T, N_BANDS))
    angs_demo = np.tile([38., 5., 100.], (N_demo, T, 1))

    mu, sigma = encode_pixels(encoder, refl_demo, angs_demo,
                               obs_doys, device=device)

    print("Posterior parameter estimates (median across pixels):")
    print(f"  {'Parameter':<12}  {'Median':>10}  {'Std across pixels':>18}")
    print(f"  {'─'*44}")
    for i, (name, unit) in enumerate(zip(PARAM_NAMES, PARAM_UNITS)):
        print(f"  {name:<12}  {np.median(mu[:, i]):>10.4f}  "
              f"{'(' + unit + ')':>18}")

    h_end   = np.median(mu[:, 10])
    h_start = np.median(mu[:,  8])
    print(f"\n  ĥ_start (green-up) : {h_start:.0f} DOY")
    print(f"  ĥ_end   (harvest)  : {h_end:.0f}   DOY")

    eval_doys = np.arange(100, 310, 3)
    lai_traj  = get_lai_trajectory(decoder, mu, eval_doys)
    plot_lai_over_time(eval_doys, lai_traj, sample_step=1)


# ─────────────────────────────────────────────────────────────────────────────
# Command-line entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ARC-VAE field inference — give it a GeoJSON and '
                    'date range to retrieve biophysical parameters from S2.')
    parser.add_argument('--geojson',  default=None,
                        help='Path to GeoJSON file defining the field boundary')
    parser.add_argument('--start',    default='2018-04-01',
                        help='Start date YYYY-MM-DD (default: 2018-04-01)')
    parser.add_argument('--end',      default='2018-10-20',
                        help='End date   YYYY-MM-DD (default: 2018-10-20)')
    parser.add_argument('--crop',     default='maize',
                        help='Crop type — only maize is currently trained')
    parser.add_argument('--ckpt',     default=CKPT_PATH,
                        help='Path to checkpoint_best.pt')
    parser.add_argument('--outdir',   default=None,
                        help='Folder to cache downloaded S2 data')
    parser.add_argument('--device',   default='cpu',
                        help='torch device: cpu or cuda')
    parser.add_argument('--no-plot',  action='store_true',
                        help='Suppress all plots')
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"\nCheckpoint not found: {args.ckpt}")
        print("Place checkpoint_best.pt at the path above, or pass --ckpt.")
        sys.exit(1)

    if args.geojson is None:
        # No field provided — run the synthetic demo
        print("No --geojson provided. Running synthetic demo.")
        print("For real S2 data: python inference_example.py "
              "--geojson my_field.geojson --start 2022-04-01 --end 2022-10-01\n")
        demo_synthetic(args.ckpt, device=args.device)
    else:
        if not os.path.exists(args.geojson):
            print(f"GeoJSON not found: {args.geojson}")
            sys.exit(1)
        run_field(
            geojson_path    = args.geojson,
            start_date      = args.start,
            end_date        = args.end,
            crop_type       = args.crop,
            checkpoint_path = args.ckpt,
            data_folder     = args.outdir,
            plot            = not args.no_plot,
            device          = args.device,
        )
