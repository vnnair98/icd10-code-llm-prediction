# Enhancing Healthcare Efficiency: Automated Abdominal Pain ICD-10 Code Classification via Automatic/Deep Classifiers

## 📌 Overview
This project explores the use of machine learning and deep learning models to automate the assignment of ICD-10 codes for abdominal pain-related diagnoses. Conducted as part of a Master's course at the University of Pennsylvania, the goal is to reduce the manual burden on healthcare professionals while improving coding accuracy and efficiency.

## 🏥 Motivation
ICD-10 coding is a time-consuming task in clinical workflows. Automating this process, especially for common symptoms like abdominal pain, can:

Reduce administrative workload

Improve code accuracy

Streamline healthcare operations

Enable professionals to focus on more complex patient care

## 📊 Dataset
Source: MIMIC-IV, a publicly available medical dataset developed by MIT.

Size: ~9,000 clinical notes labeled with ICD-10 codes for abdominal pain cases.

Preprocessing: Text normalization, tokenization, and vectorization were applied prior to modeling.

## 🤖 Models Used
A diverse set of machine learning and deep learning models were implemented:

Baseline Models (via Scikit-learn):

- Logistic Regression

- Naive Bayes

- Linear Support Vector Machines (SVM)

- Advanced Models:

- XGBoost (Scikit-learn API)

- Convolutional Neural Network (PyTorch – built from scratch)

### CNN Architecture Highlights:
- Custom CNN in PyTorch

- Dropout layers for regularization

- MaxPooling layers for feature extraction

- Tuned using Optuna for hyperparameter optimization

## 🧪 Hyperparameter Tuning
Framework: Optuna

Method: Sequential Model-Based Optimization (SMBO)

Note: Only 10 trials were run due to compute constraints. Limited trials likely impacted tuning performance.

## 📈 Results
Accuracy: ~60% across most models

Key Observations:

Simpler models (e.g., Logistic Regression, SVM) outperformed the CNN.

Optuna-based hyperparameter tuning yielded negligible improvement, likely due to dataset size and low number of trials.

CNN underperformance attributed to limited data and high model complexity.

## 🚧 Limitations & Future Work
Data size: 9,000 samples is relatively small; performance is expected to improve with more data.

Compute resources: Limited RAM/time restricted full hyperparameter search.

Model tuning: Future work includes manual hyperparameter interaction analysis and expanding Optuna trials.

Generalization: Transfer learning and pre-trained language models like BERT may yield better results on clinical text.

## ✅ Conclusion
This project demonstrates the feasibility of using automated classifiers for ICD-10 code prediction in abdominal pain diagnoses. While baseline performance is moderate, this approach offers promise for reducing manual workload and improving coding consistency. With more data and fine-tuned models, these systems can become vital tools in healthcare delivery.
