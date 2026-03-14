# Industrial Predictive Maintenance with Machine Learning

A machine learning system for predicting equipment failures in industrial processes using historical sensor data. Built with the [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020).

## Overview

Industrial equipment failures cause costly downtime and safety risks. This project develops and compares three classification models to predict failures before they occur, enabling proactive maintenance strategies.

## Dataset

- **Source:** AI4I 2020 Predictive Maintenance Dataset (Kaggle)
- **Size:** 10,000 data points, 14 features
- **Features:** Air temperature, process temperature, rotational speed, torque, tool wear, product quality type
- **Target:** Binary classification (failure / no failure)
- **Imbalance:** ~3.4% failure rate

## Models Compared

| Model | Description |
|-------|-------------|
| **Decision Tree** | Interpretable classifier with balanced class weights |
| **SVM** | Support Vector Machine for high-dimensional separation |
| **Neural Network (MLP)** | Multi-layer perceptron with two hidden layers (64, 32) |

## Key Steps

1. **Exploratory Data Analysis** — distribution analysis, correlation heatmap
2. **Preprocessing** — feature encoding, data leakage prevention, stratified split, feature scaling
3. **Training & Evaluation** — classification reports, confusion matrices
4. **Model Comparison** — accuracy, precision, recall, F1-score comparison

## How to Run
```bash
# Clone the repository
git clone https://github.com/yerayfdzp/predictive-maintenance-ml.git
cd predictive-maintenance-ml

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Download the dataset from Kaggle and place ai4i2020.csv in the root folder
# https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

# Run the notebook
jupyter notebook predictive_maintenance.ipynb
```

## Docker

Build and run the project using Docker:
```bash
docker build -t predictive-maintenance .
docker run -p 8888:8888 predictive-maintenance
```

Then open http://localhost:8888 in your browser to access the notebook.

## Tech Stack

- Python 3
- scikit-learn (Decision Tree, SVM, MLP)
- pandas & NumPy
- matplotlib & seaborn
- Docker
