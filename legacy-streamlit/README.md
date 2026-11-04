# Experiment 6: Linear Regression Virtual Lab

An interactive Streamlit Machine Learning Virtual Lab for Linear Regression using
Scikit-learn's built-in Diabetes dataset.

## Run

```powershell
cd ".\Linear_Regression_Experiment"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`.

## Sections

The application contains exactly eight primary sections: Aim, Theory, Procedure,
Data Preparation, Simulation, Exercise, References, and Experiment History.
Simulation results are calculated from the selected features and current
configuration. Saved experiment history is stored locally in
`experiment_history.json`.

## Machine-learning workflow

The Diabetes dataset has 442 records, 10 numerical features, and a continuous
target. The app supports one-feature simple regression and multi-feature
regression, configurable train-test splits, optional StandardScaler preprocessing,
dynamic coefficients and equations, prediction inputs, actual-vs-predicted and
residual plots, MAE, MSE, RMSE, R², and coefficient analysis. The scaler is fitted
only on the training partition.
