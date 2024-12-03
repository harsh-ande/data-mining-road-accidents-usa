# Analyzing Factors Contributing to Accidents in the US
This project determines the important factors contributing to road accidents in the US.

## Dataset
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data
## Features
- Data preprocessing for missing values and derived features.
- Feature selection using Recursive Feature Elimination (RFE).
- Predictive modeling with Logistic Regression, Random Forest, and Decision Tree.
- Cross-validation for model robustness.
- Geospatial heatmaps for high-risk accident zones.

## Requirements
- Python 3.7 or later
- Jupyter Notebook

## How to Use
- Clone the Repository:
```
git clone https://github.com/harsh-ande/data-mining-road-accidents-usa
cd data-mining-road-accidents-usa
```
- Install Dependencies:
```
pip install -r requirements.txt
```
- Prepare the Dataset: Place the dataset file US_Accidents_March23.csv file in the root directory.

- Run the Notebook: Launch and execute step-by-step. (Alternatively run the .py file using the following command)
```
python3 ./script.py
```


## Outputs
- Model evaluation metrics (accuracy, precision, recall).
- Feature importance rankings.
- Heatmaps of accident-prone locations.
