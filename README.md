# 🧬 Breaking the Lead Optimization Bottleneck in Drug Discovery  
### Artificial Intelligence–Driven Novel Antiviral Generation  

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-red)
![RDKit](https://img.shields.io/badge/RDKit-Chemistry-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-orange)
![Capstone](https://img.shields.io/badge/USD-Capstone_Project-blueviolet)
![Team](https://img.shields.io/badge/Team-8-black)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Overview

This project builds an end-to-end generative AI pipeline for antiviral drug discovery. It focuses on accelerating the transition from a **known active compound (“hit”)** to a **viable candidate (“lead”)** by generating and filtering novel molecular structures.

The system uses SMILES-based molecular representations, deep learning, and rule-based filtering to propose new antiviral candidates that satisfy drug-like constraints.

This work was developed as part of the University of San Diego MS in Applied Data Science capstone.

## 🎯 Objective

The goal of this project is to:

- Generate **novel antiviral molecules** using generative AI  
- Ensure outputs are **chemically valid and drug-like**  
- Provide a **decision-support tool** for early-stage drug discovery  

Rather than replacing laboratory validation, this system acts as a **computational screening layer** to prioritize candidates.

## 🧠 Key Features

- SMILES-based molecular generation  
- Baseline mutation model vs LSTM comparison  
- RDKit validation and standardization  
- Lipinski Rule of Five filtering  
- Similarity scoring against seed molecules  
- Streamlit web application for interaction  

## 🏗️ Project Pipeline

The system follows a structured pipeline:

1. **Data Collection**
   - J05 antiviral compounds from ChEMBL

2. **Data Cleaning & Validation**
   - RDKit sanitization  
   - Deduplication  
   - Canonical SMILES conversion  

3. **Feature Engineering**
   - Molecular descriptors (MW, logP, HBD, HBA)  
   - Lipinski compliance flag  

4. **Modeling**
   - Baseline: mutation-based generator  
   - Primary: character-level LSTM  

5. **Generation**
   - Sample candidate molecules  
   - Validate with RDKit  

6. **Filtering & Ranking**
   - Lipinski rules  
   - Similarity scoring  
   - Structural constraints  

7. **Deployment**
   - Streamlit web app interface  

## 📊 Results Summary

- **Baseline Model**
  - Higher validity rate  
  - Low novelty  

- **LSTM Model**
  - Lower validity  
  - 100% novelty among valid outputs  
  - All valid outputs passed Lipinski rules  

**Key takeaway:**  
There is a tradeoff between **quantity (baseline)** and **quality + novelty (LSTM)**.
## 🌐 Web Application

🔗 **Live App:**  
https://ads-599-capstone-team-8-ga4zqwdzwasig2jxp33hgr.streamlit.app/

### What the app does:

- Accepts a SMILES string (or seed molecule)  
- Generates new molecular candidates  
- Displays:
  - Molecular structures  
  - SMILES output  
  - Drug-like properties  
  - Lipinski pass/fail  
  - Similarity score  

The app serves as a **visual and interactive layer** for exploring model outputs.

## 🧰 Tech Stack

### Core Modeling & ML
- Python 3.11  
- PyTorch (LSTM neural network)  
- Scikit-learn  

### Chemistry & Data
- RDKit  
- ChEMBL Web Resource Client  
- Pandas / NumPy  

### Visualization
- Matplotlib  
- Seaborn  
- Plotly  

### Application Layer
- Streamlit  

### Development Environment
- JupyterLab  
- Git / GitHub  

## 📁 Repository Structure

```bash
ADS-599-CAPSTONE-Team-8/
├── app.py
├── model_utils.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── outputs/
│   ├── artifacts/
│   └── results/
├── requirements.txt
└── README.md
```
---

### Description

- **app.py** – Streamlit web application  
- **model_utils.py** – Model training and generation logic  

- **data/**
  - **raw/** – Original datasets (ChEMBL, etc.)  
  - **processed/** – Cleaned and model-ready data  

- **notebooks/** – EDA, preprocessing, and modeling notebooks  

- **outputs/**
  - **artifacts/** – Saved models (.pt files, metadata)  
  - **results/** – Generated molecules and evaluation outputs  

- **requirements.txt** – Python dependencies  
- **README.md** – Project documentation  

## ⚙️ How to Run

### 1. Clone the repo
```
git clone https://github.com/KatieKimberling/ADS-599-CAPSTONE-Team-8.git  
cd ADS-599-CAPSTONE-Team-8  
```

### 2. Create environment
```
conda create -n capstone_env python=3.11  
conda activate capstone_env  
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Run the app
```
steam_mvp.py run app.py
```

## 🧪 How to Use

1. Input a SMILES string or choose a seed molecule  
2. Generate candidates  
3. Review:
   - Validity  
   - Drug-likeness  
   - Structural similarity  
4. Identify promising molecules for further analysis  

## ⚠️ Limitations

- Small dataset (~87 molecules) limits generalization  
- LSTM struggles with complex molecular structures  
- Heavy reliance on Lipinski rules may exclude valid antivirals  
- Not a replacement for experimental validation  

## 🚀 Future Improvements

- Expand dataset (antibiotics, cancer drugs, larger ChEMBL subsets)  
- Use transformer-based models  
- Add ADMET prediction models  
- Improve ranking and scoring system  
- Optimize generation speed in the app  

## 🤝 Authors

- Linden Conrad-Marut  
- Katherine Kimberling  
- Jordan Torres  

University of San Diego  
MS Applied Data Science  

## 📌 Acknowledgment

AI tools were used to support development, planning, and drafting. All outputs were reviewed and validated by the authors, who take full responsibility for the final content.

## 🔑 Final Notes

This project demonstrates how generative AI can be integrated into a practical pipeline for molecular discovery, with a clear path toward real-world research support.

