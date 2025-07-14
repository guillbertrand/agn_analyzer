# AGNAnalyzer

A Python tool to estimate the mass of supermassive black holes in Active Galactic Nuclei (AGN) from 1D spectra.

---

## 📌 Status

**⛏️ Work in Progress (WIP)**  
This project is under active development, mainly aimed at astronomy enthusiasts interested in spectral analysis. 

---

## 🚀 Features

- **Redshift estimation** from emission lines (Hβ, [OIII], Hα)
- Multi-Gaussian **emission line fitting**
- Extraction of flux at **5100 Å** and extinction correction
- Calculation of **FWHM** of broad Hβ and luminosity estimation
- Monte Carlo simulation to estimate **uncertainties**
- **Black hole mass** estimation using scaling relations (Vestergaard or Feng)
- Automatic generation of **PNG figures**
- Fully configurable via a JSON file

---

## 📦 Dependencies

Make sure you have the following libraries installed:

```bash
numpy
matplotlib
scipy
astropy
specutils
extinction
```

You can install them via pip:

```bash
pip install numpy matplotlib scipy astropy specutils extinction
```

---

## 🧪 Example Usage

Example configuration JSON (customize for your own data):

```json
{
  "object_name": "NGC_5548",
  "spectrum_path": "spectrum.fits",
  "spectrum_abs_path": "spectrum_absolute.fits",
  "initial_guess": [
    {"name": "Hbeta", "rest": 4861, "mean": 4875, "stddev": 6},
    {"name": "[OIII]5007", "rest": 5007, "mean": 5020, "stddev": 6}
  ],
  "fit_o_4959": true,
  "R": 1000,
  "mag_V": 15.2,
  "mag_err": 0.1,
  "EBV": 0.03,
  "subtitle_1": "Spectral fit around Hβ",
  "subtitle_2": "Broad and narrow components",
  "monte_carlo_runs": 200
}
```

Run the analysis:

```python
from agn_analyzer import AGNAnalyzer

analyzer = AGNAnalyzer("config.json")
analyzer.analyze()
```

A `.png` figure will be automatically generated with the fitting result.

---

## 📊 Methodology

The code follows standard methods from the literature:

- Gaussian fitting of **Hβ and [OIII]**
- Redshift determination from the emission line centers
- Measurement of **FWHM** (in km/s) and mass estimation via:
  - **Vestergaard & Peterson (2006)** relation
  - **Feng et al. (2014)** relation

Monochromatic luminosity at **5100 Å** is extracted and corrected for galactic extinction using the **Cardelli, Clayton & Mathis (1989)** extinction law.

