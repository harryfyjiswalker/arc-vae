# GEOL0069 Final Project: Physics-Informed Deep Learning for Efficient Extraction of Biophysical Parameter Time Series

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zC4AWfp0Af7_LH2F_ZHQNeYrFT6Oe19A)

---

<details>
<summary><b>Table of Contents</b></summary>

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background](#2-background)
   - [Biophysical Parameters and Radiative Transfer Models](#21-arctic-leads)
   - [Current Approaches and Challenges](#22-sar-radar-altimeter)
   - [ARC-VAE](#23-clustering-algorithms)
3. [Data and Methods](#3-methods)
   - [Data & Preprocessing](#31-data--preprocessing)
4. [Discussion and Results](#4-discussion-and-results)
   - [Performance on Synthetic Data](#41-feature-space-analysis)
   - [Field Validation](#42-echo-waveform-analysis)
5. [Getting Started](#getting-started)
6. [Repository Structure](#repository-structure)
7. [References](#references)
8. [Contact](#contact)
9. [Acknowledgements](#acknowledgements)

</details>

## 1. Project Overview
---

## 2. Background

### 2.1 Biophysical Parameter Extraction

The regular and accurate elucidation of crop biophysical parameters is essential for crop trait and health monitoring, phenotyping, crop yield prediction, and precision agriculture more generally.[1][2] Leaf chlorophyll content ($C_{ab}$), for example, is an effective of indicator of stresses such as nitrogen deficiency.[3] Similarly, leaf area index (LAI) provides insight into ecological processes such as photosynthesis and evapotranspiration [2], while peak seasonal LAI exhibits a strong relationship with end-of-season yield.[4]

#### 2.1.1 Radiative Transfer Models and Challenges

<p align="center">
  <img src="/images/PROSAIL_flow1.png" width="50%" alt="PROSAIL flow">
  <br>
  <em>Figure 1: Flowchart depiction of the PROSAIL model, reproduced from Zérah (2024).[14]</em>

</p>

The increasing availability of remote sensing data has the potential to enable scalable mapping of these variables. However, due to the time-consuming and expensive nature of collection, field data on biophysical parameters is scarce, limiting the applicability of supervised learning methods.[12] Radiative transfer models (RTMs), which directly simulate the spectral and bidrectional reflectance of a crop canopy from its biophysical and biochemical properties, have therefore become foundational in this context (Verrelst et al., 2025), as their inversion - that is, reversing the problem to use known bidrectional reflectance data as input - can enable retrieval of these properties without the need for calibration with ground truth data.[1][5]

Among the most widely used RTMs is the PROSAIL model (Fig. 1), which couples the PROSPECT leaf optical properties and SAIL canopy bidrectional reflectance models.[6] PROSPECT, first developed by Jacquemoud and Baret (1990), simulates the reflectance and transmittance of a single leaf as a function of its biophysical properties.[7][8] Initially only employing three input parameters - leaf mesophyll (N), chlorophyll a and b concentration ($C_{ab}$), and leaf water content ($C_w$) - it has been expanded to incorporate additional variables including dry matter content ($C_m$), leaf mass per area (LMA), brown pigments ($C_{bp}$), total carotenoid content ($C_{cx}$, leaf anthocyanin content ($C_{anth}$, PROSPECT-D), and, most recently in PROSPECT-PRO, the subdivision of LMA into leaf protein content and carbon-based constituents (CBC).[9][10] SAIL then extends PROSPECT, simulating how light interacts with a full plant canopy, rather than a single leaf.[citation?]

However, the practical utility of PROSAIL and other RTMs in monitoring applications has traditionally faced a number of limitations. RTM parameter estimation from earth observation data is considered ill-posed: inversion typically treats each observation as an independent event, meaning that distinct combinations of biophysical parameters at certain time stamps produce identical spectral signals (equifinality).[11] Further, the computational intensity of inversion limits the scalability of RTM inversion-based models (Verrelst et al., 2025)[12][13], while difficulties in capturing the temporal evolution of crop trait dynamics over a growing season constrains their accuracy in downstream tasks such as yield prediction.[1]

#### 2.1.2 Archetypal Crop Trait Dynamics (ARC)

<p align="center">
  <img src="/images/ARC_model.jpg" width="50%" alt="ARC">
  <br>
  <em>Figure 2: Flowchart depiction of the ARC model, reproduced from Yin et al (2025).[11]</em>

</p>

The Archetypal Crop Trait Dynamics (ARC) model, recently developed by Yin _et al._ (2025) [11], addresses challenges of equfinality and phenological development by assuming that:
- biophysical parameters vary smoothly over the growing season; and,
- the smooth function that represents the variation of a biophysical parameter over a season  for a given crop in a given field can be approximated by a transformation (in terms of magnitude, p, and phenology, h) of the average function (the "archetype") for that crop over many instances.

In practice, ARC operates in two phases (Fig. 2). First, SIAC-corrected Sentinel-2 bidirectional refectance time series are obtained over a large region for pixels covering fields of a given crop. To these, artificial neural network (ANN) emulators of the (inverted) PROSAIL RTM are applied to obtain time series of seven biophysical parameters (Table 1) for each pixel, which are then matched to a common timeline using a double logistic model of LAI as a reference point. For each parameter, the median time series over all pixels is then obtained, which constitutes that parameter's "archetype" i.e. its typical seasonal development.

These archetypes are then used to constrain parameter retrieval when the model is applied to a new pixel. Eleven scaling parameters are defined, comprising seven magnitude parameters (h, one for each archetype) and four timing parameters that describe the start and end of the growing season alongside behaviour during green-up and senescence. _In essence, the model then seeks to obtain the scaling parameters that "squish" or "stretch" the archetype curves along the y-axis (magnitude) and/ or x-axis (phenology) to transform them into curves matching the observed variations in the target pixel._ This is achieved by applying a Monte Carlo search to generate many (p,h) parameter combinations from uniform bounds. The archetype model is then applied to each (p,h) to obtain the resulting biophysical parameter time series, which are subsequently run back through the forward PROSAIL model to estimate the candidate Sentinel-2 reflectances resulting from each (p,h) combination. A KNN search is then used to obtain the K = 100 nearest neighbours to the true reflectance of the target pixel; the mean, weighted by distance to the target, of the (p,h) combinations responsible for these 100 nearest reflectances is then taken to produce the final estimate of the biophysical parameter time series at that pixel.

This approach demonstrates impressive performance in validation. Using archetypes obtained from maize pixels in Northeast China, Yin _et al._ apply the model to winter wheat, winter triticale, and maize fields in the Munich North ISar (MNI) test site in Germany, achieving $R^2$ of 0.92 and RMSE of 0.46 $m^2/m^2$ for maize LAI, alongside $R^2 > 0.69$ for LAI, $C_ab$, $CC_w$, and $C_{brown}$ when results were aggregated across all crop types. However, in the context of scalable monitoring, the approach faces limitations in terms of the computational intensity of the retrieval step.

#### 2.1.3 Variational Autoencoders and PROSAIL-VAE

<p align="center">
  <img src="/images/Zerah et al.png" width="50%" alt="PROSAIL-VAE">
  <br>
  <em>Figure 3: Architecture diagram of the PROSAIL-VAE model, reproduced from Zérah et al. (2024).[14]</em>

</p>

Simultaneously, recent physics-informed deep learning architectures have exhibited success in accurate single-date radiative transfer model inversion. Alongside the popular SNAP biophysical processor developed by the European Space Agency [15], Zerah _et al._ (2024) introduced PROSAIL-VAE, which embeds a differentiable PROSAIL decoder into a variational autoencoder. The authors train the model on Sentinel-2 imagery from fourteen western European tiles without requiring ground-truth biophysical labels, achieving probabilistic inversion of all PROSAIL parameters with LAI accuracy rivalling SNAP.[14] Mensah _et al._ (2025) then demonstrated that an equivalent Transformer-VAE trained exclusively on PROSAIL-simulated data achieves comparable LAI retrieval accuracy (RMSE 0.99 vs 1.16 for PROSAIL-VAE across all test sites), substantially reducing the data requirements of the hybrid approach.[13] However, these approaches involve single-date inversion, limiting their utility in downstream tasks such as yield prediction that benefit from richer phenological data. Further, the models of both Zerah _et al._ and Mensah _et al._ employ independent uniform distributions within physiological bounds as the prior, which constrains parameters to plausible ranges but encodes no crop- or region-specific information.

### 2.2. ARC-VAE

For this project, we therefore investigate whether integrating ARC's archetype and decoder into these VAE-based frameworks can combine the complementary strengths of both approaches. We consider that, with respect to ARC, the VAE framework would allow amortisation of the cost of inference through a learned encoder. ARC's Monte Carlo solver requires  $N_{MC} = 2 \times 10^6$ PROSAIL forward model evaluations per pixel at inference time, which is paid independently for each pixel, year, and region. In contrast, a trained VAE encoder would enable mapping of the full S2 time series directly to the posterior distribution over $(p,h)$ in a single forward pass, with the computational cost of training paid once on simulations sampled from the archetype prior distribution. 

Simultaneously, with respect to the VAE framework, the ARC archetype provides a structured prior over $(p,h)$ -- the posterior distribution of scaling and phenological parameters derived from Sentinel-2 observations -- which naturally replaces the crop-agnostic uniform distributions of Zerah _et al._ and Mensah _et al._ with a physically-grounded, temporally-resolved description of expected canopy dynamics specific to a given crop. Further, the ARC decoder enables extraction of the development trajectory of biophysical parameters over the full growing season rather than solely single-date inversion.

## 3. Remote Sensing Technique and Model Architecture

### 3.1 Sentinel-2

### 3.2 ARC-VAE Architecture

The ARC-VAE encoder maps a variable-length time series of cloud-free Sentinel-2 surface reflectance observations to a posterior distribution over the 11 ARC parameters **z** = (**p**,**h**), where **p** $\in$ $\mathbb{R}^7$ are biophysical scaling parameters and **h** $\in$ $\mathbb{R}^4$ are phenological timing parameters. At inference, the posterior mean $**\mu**$ is used as a point estimate, allowing for mapping from the observed time series to a full seasonal biophysical parameter trajectory in a single forward pass. The data generation and architecture is described in detail in Sections 3.1.1-3.1.6, and depicted in Figure 4 below.

<p align="center">
  <img src="/images/Zerah et al.png" width="50%" alt="PROSAIL-VAE">
  <br>
  <em>Figure 4: Architecture diagram of the ARC-VAE model.</em>

</p>

#### 3.2.1 Synthetic Training Data Generation

#### 3.2.2 Input Representation

Each observation at time $t$ consists of 10-band surface reflectance **$r_t$** $\in$ $\mathbb{R}^{10}$, viewing geometry **$a_t$** $\in$ $\mathbb{R}^3$ (solar zenith, view zenith, relative azimuth), and calendar day-of-year $DOY_t$. To support batching, sequences of variable length are padded to a fixed maximum length $T_{max}$. Padded positions are filled with zeroes and excluded from subsequent computations via an observation mask. 

Reflectance and angular variables are normalised using fixed band-wise constants and concatenated into a 13-dimensional vector, which is projected to a $d_{model} = 128$ embedding via a linear layer to form token $\mathbf{u_t}$. To encode seasonality, we construct a DOY positional encoding consisting of four sinusoidal annual harmonics combined with a standard sinusoidal positional encoding. The resulting 136-dimensional representation is projected to 128 dimensions and added to each token embedding.

#### 3.2.3 Transformer Encoder

The encoder consists of four pre-norm Transformer layers. Each layer applies multi-head self-attention with four heads and key dimension $d_k = 32$, followed by a position-wise feed-forward network with hidden dimension $d_{\text{ff}} = 256$ and ReLU activation. Residual connections are used throughout. Self-attention is computed only over valid (non-padded) positions via the observation mask. The output is a contextualised sequence representation $\mathbf{H} \in R^{Tx128}$, where each token aggregates information across all valid observations in the time series.

Instead of global pooling, we introduce parameter-specific cross-attention, where we define eleven learnable query vectors $\mathbf{Q_j} \in R^{128}$, each corresponding to one ARC parameter. These queries attend over $\mathbf{H}$ to produce parameter-specific context vectors $\mathbf{C_j} \in R^{128}$. The motivation here is to allow each parameter to selectively weight the observations most informative for its estimation. For example, the query for $h_{end}$ can attend to late-season observations constraining senescence timing, while the query for $p_{LAI} attends to peak-canopy observations. 

Each context vector is then mapped to a posterior mean and log-standard deviation via dedicated prediction heads, with outputs constrained to physiological bounds:

```math
\mu_j
=
z_{\mathrm{lo},j}
+
\sigma
\left(
\mathbf{w}_j^\mu \cdot \mathbf{C}_j + b_j^\mu
\right)
\cdot
\left(
z_{\mathrm{hi},j} - z_{\mathrm{lo},j}
\right)
```

The full encoder output defines the posterior

```math
q(\mathbf{z} \mid \mathbf{x})
=
\prod_{j=1}^{11}
\mathcal{TN}
\left(
\mu_j,\,
\sigma_j;\,
z_{\mathrm{lo},j},\,
z_{\mathrm{hi},j}
\right)
```

as 11 independent truncated normal distributions. Samples are drawn via the inverse-CDF reparameterisation trick to maintain differentiability.

#### 3.2.4 Decoder

The ARC decoder is identical to the deterministic ARC forward model and is kept fixed during training. Given sampled parameters $(\mathbf{p}, \mathbf{h})$, observation DOYs, viewing geometry, and soil parameters, the decoder proceeds as follows:
   1. A double-logistic phenological curve is evaluated over a 365-day calendar year
   2. The curve is inverted using a pre-computed lookup table to obtain archetype time $\tau \in [0, 365]$ for each observation date
   3. A canonical seasonal trajectory $\mathbf{a}(\tau) \in \mathbb{R}^7$ is obtained via linear interpolation from a stored 365-day reference table
   4. Canopy state variables are computed via the archetype model as $\hat{x}_{\text{canopy}}(t) =p \cdot a\\left(\mathcal{T}^{-1}(t_0, h \rightarrow h_0)\right)$
   5. The resulting canopy state and geometry are passed through a frozen PROSAIL neural network emulator to predict reflectance: $\hat{\mathbf{r}} \in \mathbb{R}^{T \times 10}$.

#### 3.2.5 Training objective

The model is trained using a composite objective:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{rec}}
+
\beta \mathcal{L}_{\mathrm{KL}}
+
\lambda_{\mathrm{sup}} \mathcal{L}_{\mathrm{sup}}
```

The reconstruction loss $\mathcal{L}_{\mathrm{rec}}$ is a heteroscedastic mean squared error over reflectance predictions. The KL divergence term is defined as

```math
\mathcal{L}_{\mathrm{KL}}
=
\mathrm{KL}
\left[
\mathcal{TN}(\mu, \sigma)
\,\|\, 
\mathcal{TN}(\mu_{\mathrm{prior}}, \sigma_{\mathrm{prior}})
\right]
```

and is estimated via Monte Carlo sampling. This penalises deviation of the posterior mean from the known $\mathbf{p},\mathbf{h}) used to generate each synthetic sample.

The supervised regularization term enforces consistency with known latent parameters:

```math
\mathcal{L}_{\mathrm{sup}}
=
\frac{1}{11}
\sum_j
\left(
\frac{
\mu_j - z_{\mathrm{true},j}
}{
z_{\mathrm{hi},j} - z_{\mathrm{lo},j}
}
\right)^2
```

#### 3.2.6 Training Protocol

We train the model on 200,000 synthetic Sentinel-2 reflectance time series generated by the ARC archetype forward model. Each sample is produced by first drawing biophysical scaling parameters $\mathbf{p}$ and phenological timing parameters $\mathbf{h}$ from their prior distributions, then applying the frozen ARC decoder - here, we use the 'maize' archetype - to produce simultaed surface reflectances at a randomly selected set of observation dates. Observation dates are drawn from the maize growing season and padded to a maximum sequence length of $T_{max}$, with a binary mask simulating variable cloud-free acquisition schedules typical of real Sentinel-2 time series. Since the ground-truth parameters $\mathbf{p}$ and $\mathbf{h}$ are known - being the inputs to the forward model - the supervised auxiliary loss $L_{\text{sup}}$ requires no external labels and training is entirely self-supervised.

Training proceeds over two stages on a Tesla T4 GPU (Google Colab). This comprises 50 epochs in total; combined data generation and training takes 262 minutes with a batch size of 128. In the first stage (20 epochs), $\beta = 0$, enabling the encoder to learn good reflectance reconstruction before any KL regularisation is applied. This warm up is used to avoid early posterior collapse, where the encoder defaults to the prior before the reconstruction signal has been established. In the second stage (30 epochs), $\beta$ is linearly annealed from 0 to 0.3 to gradually regularise the posterior towards the prior distribution. $\beta = 0.3$ is chosen to be sufficiently small to preserve reconstruction quality, while sufficient to maintain non-degenerate posterior variances and provide meaningful uncertainty estimates.

The supervised loss is applied throughout training with weight $\lambda_{\text{sup}} = 10.0$. This penalises deviation of the posterior mean $\mathbf{mu}$ from the known true parameters $\mathbf{z}$ used to generate each synthetic sample, normalised by the parameter range so that each of the 11 parameters contributes equally. We find this term to be essential in prevention of degenerate solutions; without it, the encoder appears to exploit the weak sensitivity of near-infrared reflectance to leaf area above a certain LAI, finding parameter combinations that reproduce the observed spectra accurately but with incorrect seasonal timing and amplitude.

The encoder contains 619,414 trainable parameters, while the decoder is fully deterministic and contains no learnable parameters.

#### 3.6.7

At inference, ...

[Discuss sentinel-2 stuff]

### 4. Results and Comparison to ARC

#### 4.1 Performance on Synthetic Test Data

We perform initial comparative analysis of ARC-VAE and ARC on synthetic test data (32 samples), investigating reconstruction of both the biophysical parameter time series and latent **(p,h)** variables. The total inference time and inference time per pixel are reported in Table 2 below, showing an acceleration of approximately $7 x 10^4$ of ARC VAE compared to ARC at inference.

<div align="center">

| Method | Total Time (s) | Time per Pixel (s) |
|--------|----------------|---------------------|
| Encoder (ARC-VAE) | 0.04 | 0.00136 |
| ARC-KNN | 3164.00 | 98.90 |

</div>

##### 4.1.1 Biophysical Parameter Time Series and Latent Variable Reconstruction

In terms of biophysical parameter reconstruction, our results (Table 2) suggest that ARC-VAE can rival ARC on synthetic data. ARC-VAE exhibits superior $R^2$ and RMSE for five of seven biophysical parameters; notable gains are observed for nitrogen and water content elucidation (N and $C_w$, respectively), while performance on LAI, $C_m$, and ALA is very similar between the two models. By contrast, ARC slightly outperforms ARC-VAE on $C_{ab}$ reconstruction, and shows significantly superior performance for $C_{\text{brown}}$.

<p align="center">
  <img src="/images/BioParamRecon.png" width="30%" alt="PROSAIL flow">
  <br>
  <em>Table 2. Biophysical parameter reconstruction performance.</em>

</p>

However, plotting the median time series for each parameter (Figure 4) suggests that these metrics do not capture the whole picture. ARC-VAE appears more severely overstimate LAI, N, and $C_m$ in the peak season than ARC. Similarly, the model underestimates ALA during this period, compared to a more accurate reconstruction by ARC. Errors also emerge in the green-up and senescence period, with ARC-VAE suffering from positive bias in the early and late season for $C_{ab}$, $N$, and $C_m$. 

<p align="center">
  <img src="/images/synthetic_comparison.png" width="80%" alt="PROSAIL flow">
  <br>
  <em>Figure 4. Comparison of ARC-VAE and ARC time series reconstructions.</em>

</p>

Some insight into these failure modes can be gleaned from reconstruction accuracy of the **(p,h)** latent variables (Table 3). ARC-VAE outperforms ARC across all magnitude parameters **$p$**, while the opposite occurs for the phenological timing parameters **$h$**. Most striking is the VAE model's complete failure to capture the parameters defining green-up and senescence ($h_{\text{growth}}$ and $h_{\text{senes}}$, respectively), which may explain its difficulties in faithfully capturing trajectories early and late in the season. 


<p align="center">
  <img src="/images/LatentVariableRecon.png" width="30%" alt="PROSAIL flow">
  <br>
  <em>Table 3. Latent variable reconstruction performance.</em>

</p>

##### 4.1.3 Performance as a function of number of observations

The poor performance of ARC-VAE on $h_{\text{growth}}$ and $h_{\text{senes}}$ suggests that the model may struggle when the full set of seasonal observations are not available to constrain start and end dates, potentially severely limiting its utility where cloud cover is high. We therefore compare the LAI reconstruction $R^2$ and RMSE of ARC-VAE compared to ARC as a function of the number of observations (Figure 5).

#### 4.2 Performance on MNI Field Data

We validate model performance on real world data using the Munich-North-Isar dataset, which includes LAI measurements for various fields for 2017 and 2018. NB specify dates.

<p align="center">
  <img src="/images/Field Validation Results.png" width="30%" alt="PROSAIL flow">
  <br>
  <em>Table 3. Latent variable reconstruction performance.</em>

</p>

## 5. Video Summary


## 6. Environmental Cost Analysis

We consider that the main sources of energy usage and emissions in this project result from: data generation, training, and inference; generative AI usage; and Sentinel-2 data extraction. 

### 6.1 Computational Energy Cost and Carbon Footprint

#### 6.1.1 Synthetic Data Generation, Training, and Inference

| Phase | Duration | Energy (Wh) | Carbon (g CO₂eq) |
|---|---:|---:|---:|
| Synthetic data generation | 18m 34s | 15.5 | 7.36 |
| ARC-VAE training | 4h 03m 24s | 202.8 | 96.35 |
| ARC-VAE inference | 0.04 s | 0.0006 | 0.0003 |
| ARC-KNN retrieval | 52m 44s | 43.95 | 20.88 |
| **Total project** | **5h 14m 42s** | **262.3** | **124.59** |

Experiments were conducted on an NVIDIA Tesla T4 GPU within Google Colab.

#### 6.1.2 Generative AI Usage


#### 6.1.3 Validation Data Acquisition

The Sentinel-2 imagery used in the Munich-North-Isar dataset is produced by the European Space Agency Copernicus programme using the Sentinel-2 mission, which involves substantial upfront environment expenditure associated with satellite manufacturing, launch, and long-term mission operations. Similarly, agricultural field campaign sites such as Munich-North-Isar require manufacture of specialised sensing instruments, repeated technician travel, and long-term site maintenance. However, both of these costs are amortised across large numbers of observations, users, and downstream applications and are thus considered negligible with respect to this project. As a result, for model validation, the primary costs are the extraction of the relevant Sentinel-2 scenes from the Amazon Web Services API and inference of the ARC-VAE and ARC models on these scenes.

... Quantify this

### Environmental Benefits

---

## Getting Started

### Prerequisites

- A Google account with access to [Google Colab](https://colab.research.google.com/)
- A [Google Drive](https://drive.google.com/) folder containing the Sentinel-3 `.SEN3` data file

### Installation

The notebook runs in Google Colab. Install non-standard packages at the start of each session:

```python
!pip install netCDF4
!pip install rasterio
!pip install basemap
!pip install cartopy
```

Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Data

The base reference notebook, developed by Dr Michel Tsamados is available at:  
https://drive.google.com/file/d/1HDSLjsWhLIDF-qbRj6sbGVd9t1LB7890/view?usp=drive_link

The Sentinel-3 data is available via the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/). The specific file used is:

```
S3B_SR_2_LAN_SI_20190301T231304_20190301T233006_20230405T162425_1021_022_301______LN3_R_NT_005.SEN3
```

> **Note:** The data file is **not included** in this repository due to size. Download it from the Copernicus Data Space and update the file path in the notebook.

---

## Repository Structure

```
GEOL0069-Week4/
 ├── Unit_2_Unsupervised_Learning_Methods.ipynb          # Main assignment notebook
 ├── Chapter1_Unsupervised_Learning_Methods_Michel.ipynb  # Base reference notebook
 ├── figures/
 │   ├── feature_space_scatter.png        # Fig 1 - PP vs SSD feature space
 │   ├── sample_waveforms_sea_ice.png     # Fig 2a - Individual sea-ice echoes
 │   ├── sample_waveforms_leads.png       # Fig 2b - Individual lead echoes
 │   ├── mean_std_waveforms.png           # Fig 3 - Mean +/- std (unaligned)
 │   ├── aligned_mean_std_waveforms.png   # Fig 4 - Mean +/- std (FFT-aligned)
 │   └── confusion_matrix.png             # Fig 5 - Confusion matrix heatmap
 └── README.md
```

---

## References

[1] Ishaq, R.A.F., Zhou, G., Tian, C., Tan, Y., Jing, G., Jiang, H. and Obaid-ur-Rehman (2024) 'A Systematic Review of Radiative Transfer Models for Crop Yield Prediction and Crop Traits Retrieval', Remote Sensing, 16(1), p. 121. doi: 10.3390/rs16010121.

[2] Sehgal, V.K., Chakraborty, D. and Sahoo, R.N., 2016. Inversion of radiative transfer model for retrieval of wheat biophysical parameters from broadband reflectance measurements. Information Processing in Agriculture, 3(2), pp.107–118. Available at: https://doi.org/10.1016/j.inpa.2016.04.001.

[3] Xie, Q., Dash, J., Huete, A., Jiang, A., Yin, G., Ding, Y., Peng, D., Hall, C.C., Brown, L., Shi, Y., Ye, H., Dong, Y. and Huang, W. (2019) 'Retrieval of crop biophysical parameters from Sentinel-2 remote sensing imagery', International Journal of Applied Earth Observation and Geoinformation, 80, pp. 187-195. doi: 10.1016/j.jag.2019.04.019.

[4] Lewis, P.E., Yin, F., Gomez-Dans, J.L., Weiß, T. and Adam, E., 2024. Crop Yield Mapping with ARC using only Optical Remote Sensing. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, X-3-2024, pp. 199–206. Available at: https://doi.org/10.5194/isprs-annals-X-3-2024-199-2024.

[5] Sibiya, B.S., Odindi, J., Mutanga, O., Cho, M.A. and Masemola, C., 2025. The utility of radiative transfer models (RTM) on remotely sensed data in retrieving biophysical and biochemical properties of terrestrial biomes: A systematic review. Advances in Space Research, 75(10), pp. 7424–7444. Available at: https://doi.org/10.1016/j.asr.2025.02.052.

[6] S. Jacquemoud, W. Verhoef, F. Baret, C. Bacour, P. J. Zarco-Tejada, G. P. Asner,
C. François, and S. L. Ustin, “PROSPECT + SAIL models: A review of use for vegetation
characterization,” Remote Sensing of Environment, vol. 113, pp. S56–S66, Sept. 2009.

[7] S. Jacquemoud and F. Baret, “PROSPECT: A model of leaf optical properties spectra,”
Remote Sensing of Environment, vol. 34, no. 2, pp. 75–91, Nov. 1990.

[8] K. Berger, C. Atzberger, M. Danner, G. D’Urso, W. Mauser, F. Vuolo, and T. Hank, “Eval-
uation of the PROSAIL Model Capabilities for Future Hyperspectral Model Environments:
A Review Study,” Remote Sensing, vol. 10, no. 1, Jan. 2018.

[9] J.-B. Feret, C. François, G. P. Asner, A. A. Gitelson, R. E. Martin, L. P. R. Bidel, S. L.
Ustin, G. le Maire, and S. Jacquemoud, “PROSPECT-4 and 5: Advances in the leaf optical
properties model separating photosynthetic pigments,” Remote Sensing of Environment, vol.
112, no. 6, pp. 3030–3043, June 2008.

[10] J.-B. Feret, K. Berger, F. de Boissieu, and Z. MalenovskÃ½, “Prospect-pro: a leaf radia-
tive transfer model for estimation of leaf protein content and carbon-based constituents,” in
EGU General Assembly 2020, ser. EGU2020-5251, Online, 4–8 May 2020.

[11] Yin, F., Lewis, P.E., Gómez-Dans, J.L. and Weiß, T. (2025) 'Archetypal crop trait dynamics for enhanced retrieval of biophysical parameters from Sentinel-2 MSI', Remote Sensing of Environment, 318, 114510. Available at: https://doi.org/10.1016/j.rse.2024.114510.

[12] Zérah, Y., Valero, S. and Inglada, J. (2024) 'Physics-constrained deep learning for biophysical parameter retrieval from Sentinel-2 images: Inversion of the PROSAIL model', Remote Sensing of Environment, 312, 114309. Available at: https://doi.org/10.1016/j.rse.2024.114309.

[13] Mensah, P., Aderinto, P.V., Yusuf, I.S. and Pretorius, A. (2025) 'Physics informed Transformer-VAE for biophysical parameter estimation: PROSAIL model inversion in Sentinel-2 imagery', arXiv preprint arXiv:2511.10387. Available at: https://arxiv.org/abs/2511.10387.

[14] Zérah, Y., 2024. Biophysical parameter retrieval from Sentinel-2 images using physics-driven deep learning for PROSAIL inversion. Séries Temporelles, 27 October. Available at: https://www.cesbio.cnrs.fr/multitemp/biophysical-parameter-retrieval-from-sentinel-2-images-using-physics-driven-deep-learning-for-prosail-inversion/ [Accessed 18 May 2026].

[15] European Space Agency (n.d.) Biophysical Processor Overview. SNAP Online Help. Available at: https://step.esa.int/main/wp-content/help/versions/13.0.0/snap-toolboxes/eu.esa.opt.opttbx.biophysical/BiophysicalOpOverview.html (Accessed: 18 May 2026).


Verrelst, Jochem, Miguel Morata, José Luis García-Soria, Yilin Sun, Jianbo Qi, and Juan Pablo Rivera-Caicedo. 2025. "RTM Surrogate Modeling in Optical Remote Sensing: A Review of Emulation for Vegetation and Atmosphere Applications" Remote Sensing 17, no. 21: 3618. https://doi.org/10.3390/rs17213618



---

## Contact

**Harry Fyjis-Walker** – harryfyjiswalker@gmail.com 

Project Link: `https://github.com/harryfyjiswalker/GEOL0069-Week4`

---

## Acknowledgements

- This project is submitted as part of an assignment for **GEOL0069 – Artificial Intelligence for Earth Observation**, UCL Earth Sciences Department.
- Base notebook and course materials provided by **Dr Michel Tsamados**, UCL.
- Sentinel-3 data courtesy of the **European Space Agency (ESA)** / Copernicus programme.

<p align="right"><a href="#sea-ice--lead-classification-via-unsupervised-learning">Back to top</a></p>



