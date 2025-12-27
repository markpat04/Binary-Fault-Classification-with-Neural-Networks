<div align="center">

# 🧠 Neural Network Fault Detection System
### Binary Classification for Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-Metrics-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-green)

[View Architecture](#-neural-network-architecture) • [Feature Importance](#-feature-importance-analysis)

</div>

---

## 📖 Overview

This project implements a **Deep Learning Classifier** designed to distinguish between normal operating conditions and fault states in industrial systems. 

Using synthetic data that mimics sensor readings, the model demonstrates how a Multi-Layer Perceptron (MLP) can learn to detect shifts in data distributions—a key technique in **Predictive Maintenance (PdM)** and **Quality Control**.

The pipeline includes end-to-end processing:
1.  **Synthetic Data Generation:** Simulating sensor drift.
2.  **Neural Network Training:** Using TensorFlow/Keras.
3.  **Comprehensive Evaluation:** Confusion matrices, ROC analysis, and Feature Importance.

---

## ⚡ Key Features

* **🏭 Synthetic Fault Simulation:** Generates distinct data distributions for "Normal" (0-10 range) and "Fault" (5-15 range) states.
* **🧠 Deep Learning Architecture:** A 4-layer dense neural network optimized for binary classification.
* **🔍 Explainable AI (XAI):** Implements **Permutation Importance** to quantify which features drive the model's decisions.
* **📊 Visualization Suite:**
    * Training/Validation Learning Curves.
    * Probability Distribution Histograms.
    * Confusion Matrix Heatmaps.

---

## 🏗 Neural Network Architecture

The model utilizes a Feed-Forward Neural Network (MLP) structure:

```
graph LR
    A[Input Layer<br/>(4 Features)] --> B[Dense Layer<br/>(32 Neurons, ReLU)]
    B --> C[Dense Layer<br/>(16 Neurons, ReLU)]
    C --> D[Dense Layer<br/>(8 Neurons, ReLU)]
    D --> E[Output Layer<br/>(Sigmoid)]
    E --> F[Fault Probability<br/>(0.0 - 1.0)]
```
