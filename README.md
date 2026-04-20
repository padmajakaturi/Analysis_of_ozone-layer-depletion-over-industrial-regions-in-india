# Analysis of Ozone Layer Depletion over Industrial Regions in India

This repository contains the complete project on ozone layer depletion analysis in industrial regions of India using machine learning. The work includes data processing, ML model training, and visualization of the results to understand how industrial emissions affect ozone concentration.

---

## 🧠 Project Overview

The **ozone layer** is a region of the Earth’s stratosphere that absorbs harmful ultraviolet (UV) radiation from the sun. Human activities release ozone-depleting substances (like CFCs and NOx), which reduce ozone concentrations, especially over industrial regions. Effective analysis can help in forecasting future ozone levels and support environmental policy. :contentReference[oaicite:0]{index=0}

This project focuses on:

- Collecting ozone concentration and relevant environmental data.
- Preprocessing and preparing the dataset.
- Applying machine learning models to forecast ozone depletion trends.
- Visualizing results in easy-to-understand plots and dashboards.

---

## 🗂 Repository Structure

---

## 🛠️ Tech Stack

| Category | Tools/Technologies |
|----------|--------------------|
| Language | **Python** |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn, Statsmodels, Keras/TensorFlow |
| Visualization | Matplotlib, Seaborn, Plotly |
| Presentation | PowerPoint, Project Report Docs |

---

## 💾 Dataset

Your main dataset is:

- `india_ozone_dataset_2022_2025.csv` – Contains ozone concentration and environmental parameters for industrial regions.

> 📌 If the dataset is large, consider keeping it separate or linking an external source (e.g., HF dataset link) instead of uploading large CSV files in GitHub.

---

## 🚀 How to Run (Local Setup)

1. **Clone the repository**

```bash
git clone https://github.com/padmajakaturi/Analysis_of_ozone-layer-depletion-over-industrial-regions-in-india.git
cd Analysis_of_ozone-layer-depletion-over-industrial-regions-in-india
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python main_analysis.py
