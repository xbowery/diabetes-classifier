# Data Analysis Project - Diabetes Classifier

<b>Members</b>
- [Cheah King Yeh](https://www.linkedin.com/in/king-yeh-cheah/)
- [Yuen Kah May](https://www.linkedin.com/in/kah-may-yuen/)

To view the project live, please click on this link [here](https://share.streamlit.io/xbowery/diabetes-classifier/main/diabetes_predictor.py).

## Description
Our project aims to identify and understand the factors that causes diabetes (our dependent variable) and be able to develop a sound method to predict whether a person is suffering from diabetes based on the different parameters and factors given to us. We first start off with data collection from a given dataset which contains information about potential factors that causes diabetes including `age`, `glucose_concentration` and `blood_pressure`. 

We then continued our project by conducting some data pre-processing and data cleaning to deal with data outliers in order to ensure the outliers do not affect our data analysis. Following which, we used exploratory data analysis (EDA) techniques to find out any possible correlations between the factors and diabetes classification. We did so by doing a univariate analysis for each factor. We then did feature selection on the factors to only use factors that were highly correlated to diabetes classification factor. 

Finally, we tested these factors out by building 3 models, namely - **Logistic Regression model**, **K-Nearest Neighbours model** and **Random Forest Classifier model**. We tested our models against the sample data provided, and used the **Accuracy** and **F1-score** metrics to evaluate our model.

## Files Used
- [train.csv](https://github.com/xbowery/diabetes-predictor/blob/main/train.csv)
- [test.csv](https://github.com/xbowery/diabetes-predictor/blob/main/test.csv)
- [submission.csv](https://github.com/xbowery/diabetes-predictor/blob/main/submission.csv)

## Usage
1. Clone the repo <br>
`git clone ...` 
2. Open `Diabetes Predictor.ipynb` in your local jupyter notebook server. Do note that the following python packages need to be installed beforehand:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`

Original dataset can be found [here](https://www.kaggle.com/c/diabetes-classification/data).
