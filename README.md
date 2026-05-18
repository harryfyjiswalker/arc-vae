# Physics-Informed Deep Learning for Efficient Extraction of Biophysical Parameter Time Series
### GEOL0069 – Artificial Intelligence for Earth Observation | Final Project

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
  <em>Figure 2: Flowchart depiction of the ARC model, reproduced from Yin _et al_ (2025).[11]</em>

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
  <em>Figure 1: Architecture diagram of the PROSAIL-VAE model, reproduced from Zérah _et al._ (2024).[14]</em>

</p>

Simultaneously, recent physics-informed deep learning architectures have exhibited success in accurate single-date radiative transfer model inversion. Alongside the popular SNAP biophysical processor developed by the European Space Agency [15], Zerah _et al._ (2024) introduced PROSAIL-VAE, which embeds a differentiable PROSAIL decoder into a variational autoencoder. The authors train the model on Sentinel-2 imagery from fourteen western European tiles without requiring ground-truth biophysical labels, achieving probabilistic inversion of all PROSAIL parameters with LAI accuracy rivalling SNAP.[14] Mensah _et al._ (2025) then demonstrated that an equivalent Transformer-VAE trained exclusively on PROSAIL-simulated data achieves comparable LAI retrieval accuracy (RMSE 0.99 vs 1.16 for PROSAIL-VAE across all test sites), substantially reducing the data requirements of the hybrid approach.[13] 

### 2.2. ARC-VAE

Both employ independent uniform distributions within physiological bounds as the prior, which constrains parameters to plausible ranges but encodes no crop- or region-specific information.

with respect tothe integration of RTM

recent developments in physics-informed deep learning architectures have exhibited success in accurate single-date radiative transfer model inversion, while enabling significantly 





ARC AND MENSAH



## 3. Methods

## 3.1 Data & Preprocessing



## 4 Discussion and Results

### 4.1 Comparative Performance on Synthetic Data

### 4.2 Field Validation


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



