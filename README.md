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

The regular and accurate elucidation of crop biophysical parameters is essential for crop trait and health monitoring, phenotyping, and crop yield prediction (Ishaq et al., 2023; Seghal et al, 2016). Leaf chlorophyll content (C$_{ab}$), for example, is an effective of indicator of stresses such as nitrogen deficiency (Xie et al., 2019). Similarly, leaf area index (LAI) provides insight into ecological processes such as photosynthesis and evapotranspiration (Seghal et al. 2016), while peak seasonal LAI exhibits a strong relationship with end-of-season crop yield (Lewis et al, 2024).



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

Ishaq, R.A.F., Zhou, G., Tian, C., Tan, Y., Jing, G., Jiang, H. and Obaid-ur-Rehman (2024) 'A Systematic Review of Radiative Transfer Models for Crop Yield Prediction and Crop Traits Retrieval', Remote Sensing, 16(1), p. 121. doi: 10.3390/rs16010121.

Lewis, P.E., Yin, F., Gomez-Dans, J.L., Weiß, T. and Adam, E., 2024. Crop Yield Mapping with ARC using only Optical Remote Sensing. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, X-3-2024, pp. 199–206. Available at: https://doi.org/10.5194/isprs-annals-X-3-2024-199-2024.

Sehgal, V.K., Chakraborty, D. and Sahoo, R.N., 2016. Inversion of radiative transfer model for retrieval of wheat biophysical parameters from broadband reflectance measurements. Information Processing in Agriculture, 3(2), pp.107–118. Available at: https://doi.org/10.1016/j.inpa.2016.04.001.

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



