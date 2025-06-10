# Glucoma-ml

# BERT-Based Glaucoma Diagnosis from Clinical Text Data

This repository contains a BERT-based model for diagnosing glaucoma using structured clinical data converted into natural language format. It leverages the PapilaDB dataset for training and evaluation.

## 🧠 Project Overview

The project transforms tabular ophthalmic clinical data into textual descriptions that can be processed using a BERT model (`bert-base-uncased`) for sequence classification. This approach enables NLP techniques to be applied in a medical diagnostic setting, particularly for classifying different glaucoma diagnoses.

## 📁 Dataset

The clinical data is extracted from the following Excel sheets:

- `patient_data_os.xlsx` (Left Eye)
- `patient_data_od.xlsx` (Right Eye)

Each record includes features such as:
- Age, Gender
- Dioptres, Astigmatism
- IOP (Pneumatic & Perkins)
- Axial length, Pachymetry
- VF_MD
- Diagnosis (used as classification label)

These datasets are combined, cleaned, and converted into natural language strings.

## 📊 Data Preprocessing

- Rows with missing diagnosis are dropped.
- Structured fields are converted into a textual summary per patient.
- The `diagnosis` column is encoded using `LabelEncoder`.
- Text is tokenized using BERT's tokenizer.

## 🤖 Model

We use the `bert-base-uncased` model from HuggingFace Transformers with a classification head. Key steps include:

- Tokenizing the text descriptions
- Fine-tuning BERT with the processed training data
- Evaluating performance on a test set

### Training Configuration
- Batch Size: 8
- Epochs: 3
- Learning Rate: 2e-5
- Evaluation: Per Epoch

## 🏁 Quick Start

### 1. Install Requirements

```bash
pip install pandas scikit-learn transformers datasets torch openpyxl
