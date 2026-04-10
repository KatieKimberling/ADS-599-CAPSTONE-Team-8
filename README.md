# **Breaking the “Lead Optimization” Bottleneck in Drug Development for Emerging Infectious and Drug-Resistant Diseases:**
## **Artificial Intelligence-Driven Novel Antiviral Drug Discovery**

## A data science capstone project using generative AI in a pipeline to create novel BSA molecules that (a) target emerging infectious or drug-resistant diseases while (b)  passing membrane solubility and human toxicity guidelines in order to (c) drastically reduce the amount of time traditionally required in drug development.

## University of San Diego Applied Data Science Masters Program
### Spring 2026 (B), Professor Ebrahim Tarshizi

#### **Authors:**
- Linden Conrad-Marut
- Katherine "Katie" Kimberling
- Jordan Torres

## **Project Overview**

This project implements a production-style data science pipeline that integrates:

- Harvesting J05 broad spectrum antiviral SMILES strings from ChEMBL
- Standardizing and tokenizing text-based descriptions of chemical compounds (SMILES)
- Feature engineering (Lipinski's Rule of Five)
- Incorporating generative artificial intelligence into modeling
- Development of a user-facing, web-based artifact

The system demonstrates how the exorbitant computing power of Generative AI can be harnessed to speed the time in antiviral drug discovery from a "hit" (a chemical compound that targets a virus) to a "lead" (compound will also be a safe vaccine or medication for humans).

Link to Web Artifact: [Web Artifact](https://ads-599-capstone-team-8-ga4zqwdzwasig2jxp33hgr.streamlit.app/)

## System Architecture

1. Data Ingestion & Cleaning  
2. Feature Engineering  
3. Statistical Analysis & Visualization (EDA)
4. Modeling (naive baseline and LSTM)
5. Hyperparameter Tuning
6. Model Evaluation 
7. Generation of Web Artifact

## Technology Stack

- Python 3.x
- RDKit
- PyTorch LSTM
- ChEMBL
- Pandas / NumPy
- Matplotlib / Seaborn

## Web Artifact



## Ethical and Practical Considerations

- blah blah blah

## How to Run This Project

1.  Clone the repository:
```
git clone (https://github.com/KatieKimberling/ADS-599-CAPSTONE-Team-8.git)
cd ADS-599-CAPSTONE-Team-8
```

2.  Create & Activate Environment:  
We recommend a fresh Conda environment to avoid dependency conflicts.
```
conda create -n capstone_eq python=3.11
conda activate capstone_eq
```

Install required packages:
```
pip install -r requirements.txt
```

3.  Run the BLAH BLAH


4.  Generate Required Artifacts (One-Time Setup):  


5. Launch the Streamlit App:  
From the project root type: 
```
streamlit run app.py
```
The app will automatically open in your browser.

## Using the Application





