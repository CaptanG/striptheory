# Strip Theory Seakeeping Analysis Toolkit

This repository implements a Python-based strip theory solver for the **seakeeping analysis of a Wigley hull**.  
The toolkit calculates hydrodynamic coefficients (added mass, damping), ship responses, and visualizes hull geometry, using input sectional data and hydrodynamic curves.

---

## 📖 Overview

This project provides a modular codebase for simulating ship motions in waves using **strip theory**. It supports:

- Extraction and processing of hull sectional data  
- Calculation of sectional added mass, damping, and βₙ coefficients  
- RAO (Response Amplitude Operator) plot generation for heave motion  
- 3D visualization of hull geometry and motion results  
- Integration of experimental or theoretical hydrodynamic curves  

The design is extensible, enabling adaptation to different hulls and hydrodynamic datasets.

---

## 📂 File Structure

| File                     | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `main.py`                | Main analysis script: computes RAO, plots response, visualizes hull         |
| `sectionFunction.py`     | Core computational functions: geometry, hydrostatics, plotting              |
| `calculateBeta_n.py`     | Calculates βₙ nondimensional hull coefficients                              |
| `sectionAddedMass.py`    | Added mass and amplitude ratio processing using Lewis curves                |
| `convertLewis.py`        | Utilities for hydrodynamic curve interpolation                              |
| `wig_seakeeping.txt`     | Sectional coordinate file for Wigley hull                                   |
| `addedMass_0.7.txt`      | Added mass curves (Lewis-form, tabulated)                                   |
| `amplitudeRatio_0.7.txt` | Amplitude ratio curves (Lewis-form, tabulated)                              |
| `lewisCurvesGraph1.txt`  | Lewis-form hydrodynamic curves                                              |
| `lwis-curve.xlsx`        | Tabular Lewis curves data                                                   |

---

## ✨ Features

- **Strip theory-based analysis:** Calculates added mass, damping coefficients, and ship motions section-wise.  
- **Flexible hydrodynamics:** Reads tabular Lewis-form coefficients (added mass/amplitude ratio).  
- **Section visualization:** Plots hull shape in 3D with easy customization.  
- **RAO computation:** Produces heave response amplitude operator and phase plots versus tuning parameter.  
- **Extensible modules:** Easily adjust files or scripts for other hulls or curves.  

---

## 🚀 Getting Started

### Requirements
- Python 3.x  
- `numpy`  
- `scipy`  
- `matplotlib`  

### Usage
1. Clone the repository and navigate into the folder:
   ```bash
   git clone <your-repo-url>
   cd strip-theory-seakeeping
