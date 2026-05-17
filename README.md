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

### 2.1 Biophysical Parameters and Radiative Transfer Models

The regular and accurate elucidation of crop biophysical parameters is essential for crop trait and health monitoring, phenotyping, crop yield prediction, and precision agriculture more generally (Ishaq et al., 2023; Seghal et al, 2016). Leaf chlorophyll content (C$_{ab}$), for example, is an effective of indicator of stresses such as nitrogen deficiency (Xie et al., 2019). Similarly, leaf area index (LAI) provides insight into ecological processes such as photosynthesis and evapotranspiration (Seghal et al. 2016), while peak seasonal LAI exhibits a strong relationship with end-of-season yield (Lewis et al, 2024).

The increasing availability of remote sensing data enables scalable mapping of these variables via inversion of radiative transfer models (RTMs), circumventing the need for time-consuming and expensive field studies. RTMs simulate the spectral and bidrectional reflectance of a crop canopy based on its biophysical and biochemical properties; inverting these models, using bidirectional reflectance data as input, hence enables retrieval of the crop properties (Ishaq et al., 2023; Sibiya et al., 2024).

Among the most widely used RTM is the PROSAIL model, which couples the PROSPECT leaf optical properties and SAIL canopy bidrectional reflectance models (Jacquemoud, 2009). PROSPECT, first developed by Jacquemoud and Baret (1990), simulates the reflectance and transmittance of a single leaf as a function of its biophysical properties (Jacquemoud and Baret, 1990; Berger et al., 2018). Initially only employing three input parameters -- leaf mesophyll (N), chlorophyll a and b concentration (C$_{ab}$), and leaf water content (C$_w$) -- it has been expanded to incorporate additional variables including dry matter content (C$_m$), leaf mass per area (LMA), brown pigments (C$_{bp}$), total carotenoid content (C$_{cx}$, leaf anthocyanin content (C$_{anth}$, PROSPECT-D), and, most recently in PROSPECT-PRO, the subdivision of LMA into leaf protein content and carbon-based constituents (CBC) (Feret et al., 2008; Feret et al., 2020). 

SAIL then extends PROSPECT, simulating how light interacts with a full plant canopy, rather
than a single leaf. It accounts for:
— Leaf optical properties (reflectance and transmittance from PROSPECT)
— Leaf angle distribution (how leaves are oriented in 3D)
— Leaf area index (LAI, total leaf area per ground area)
— Canopy structure and soil background
— Sun and sensor geometry: solar zenith angle, sensor/viewing angle, relative azimuth angle

Leaves are modelled as turbid medium layers: photons can reflect, transmit, or scatter between
leaves. SAIL uses a four-stream approximation:
- Direct sunlight down
- Direct sunlight reflected up
- Diffuse downward radiation
- Diffuse upwards radiation

SAIL then outputs canopy reflectance and transmittance, including soil contribution, thus acting as a bridge between fundamental biophysical properties and remote sensing observations.

### 2.2 Current Approaches and Challenges in Biophysical Parameter Extraction

#### 2.2.1 Radiative Transfer Models



#### 2.2.2 ARC


#### 2.2.3 Variational Autoencoders and PROSAIL-VAE


### 2.3 ARC-VAE



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

K. Berger, C. Atzberger, M. Danner, G. D’Urso, W. Mauser, F. Vuolo, and T. Hank, “Eval-
uation of the PROSAIL Model Capabilities for Future Hyperspectral Model Environments:
A Review Study,” Remote Sensing, vol. 10, no. 1, Jan. 2018.

J.-B. Feret, C. François, G. P. Asner, A. A. Gitelson, R. E. Martin, L. P. R. Bidel, S. L.
Ustin, G. le Maire, and S. Jacquemoud, “PROSPECT-4 and 5: Advances in the leaf optical
properties model separating photosynthetic pigments,” Remote Sensing of Environment, vol.
112, no. 6, pp. 3030–3043, June 2008.

J.-B. Feret, K. Berger, F. de Boissieu, and Z. MalenovskÃ½, “Prospect-pro: a leaf radia-
tive transfer model for estimation of leaf protein content and carbon-based constituents,” in
EGU General Assembly 2020, ser. EGU2020-5251, Online, 4–8 May 2020.

Ishaq, R.A.F., Zhou, G., Tian, C., Tan, Y., Jing, G., Jiang, H. and Obaid-ur-Rehman (2024) 'A Systematic Review of Radiative Transfer Models for Crop Yield Prediction and Crop Traits Retrieval', Remote Sensing, 16(1), p. 121. doi: 10.3390/rs16010121.

S. Jacquemoud, W. Verhoef, F. Baret, C. Bacour, P. J. Zarco-Tejada, G. P. Asner,
C. François, and S. L. Ustin, “PROSPECT + SAIL models: A review of use for vegetation
characterization,” Remote Sensing of Environment, vol. 113, pp. S56–S66, Sept. 2009.

S. Jacquemoud and F. Baret, “PROSPECT: A model of leaf optical properties spectra,”
Remote Sensing of Environment, vol. 34, no. 2, pp. 75–91, Nov. 1990.

Lewis, P.E., Yin, F., Gomez-Dans, J.L., Weiß, T. and Adam, E., 2024. Crop Yield Mapping with ARC using only Optical Remote Sensing. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, X-3-2024, pp. 199–206. Available at: https://doi.org/10.5194/isprs-annals-X-3-2024-199-2024.

Sehgal, V.K., Chakraborty, D. and Sahoo, R.N., 2016. Inversion of radiative transfer model for retrieval of wheat biophysical parameters from broadband reflectance measurements. Information Processing in Agriculture, 3(2), pp.107–118. Available at: https://doi.org/10.1016/j.inpa.2016.04.001.

Sibiya, B.S., Odindi, J., Mutanga, O., Cho, M.A. and Masemola, C., 2025. The utility of radiative transfer models (RTM) on remotely sensed data in retrieving biophysical and biochemical properties of terrestrial biomes: A systematic review. Advances in Space Research, 75(10), pp. 7424–7444. Available at: https://doi.org/10.1016/j.asr.2025.02.052.

Xie, Q., Dash, J., Huete, A., Jiang, A., Yin, G., Ding, Y., Peng, D., Hall, C.C., Brown, L., Shi, Y., Ye, H., Dong, Y. and Huang, W. (2019) 'Retrieval of crop biophysical parameters from Sentinel-2 remote sensing imagery', International Journal of Applied Earth Observation and Geoinformation, 80, pp. 187-195. doi: 10.1016/j.jag.2019.04.019.



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



