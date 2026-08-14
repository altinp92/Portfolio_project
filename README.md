# Calories Burned Prediction

## Project Overview

This project analyzes gym workout data and uses machine learning to predict the number of calories burned during a workout.

The project includes data exploration, preprocessing, model training, model comparison, model interpretation, and a simple Python terminal application for making new predictions.

## Dataset

The dataset was obtained from Kaggle:

[Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset)

* Number of records: 973
* Target variable: `Calories_Burned`
* Input features: 14

The dataset contains personal and workout-related information such as age, gender, weight, height, heart rate, workout duration, workout type, body fat percentage, workout frequency, experience level, and BMI.

## Project Structure

```text
portfolio-project/
├── README.md
├── main.py
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── data/
│   └── raw/
│       └── gym_members_exercise_tracking.csv
└── notebooks/
    ├── 01_exploration.ipynb
    └── 02_modeling.ipynb
```

## Files

### `notebooks/01_exploration.ipynb`

This notebook focuses on understanding and exploring the dataset.

Main steps:

* Load and inspect the dataset
* Check data types
* Check for missing values
* Check for duplicate rows
* Explore numerical and categorical features
* Compare average calories burned across categorical groups
* Analyze correlations between numerical features and `Calories_Burned`
* Create visualizations to better understand the data

No missing values or duplicate rows were found, so no major cleaning changes were required.

The strongest positive correlation with `Calories_Burned` was found for session duration.

### `notebooks/02_modeling.ipynb`

This notebook contains the machine learning workflow.

Main steps:

* Define input features `X` and target `y`
* Identify numerical and categorical features
* Split the dataset into 80% training data and 20% test data
* Standardize numerical features using `StandardScaler`
* Encode categorical features using `OneHotEncoder`
* Apply preprocessing using `ColumnTransformer`
* Build machine learning pipelines
* Train Linear Regression
* Train Random Forest Regressor
* Evaluate both models using MAE, RMSE, and R²
* Compare model performance
* Analyze Linear Regression coefficients
* Compare actual and predicted values
* Analyze residuals

## Model Results

| Model             |   MAE |  RMSE |    R² |
| ----------------- | ----: | ----: | ----: |
| Linear Regression | 30.27 | 40.57 | 0.980 |
| Random Forest     | 36.17 | 48.18 | 0.972 |

Linear Regression performed slightly better than Random Forest because it achieved lower prediction errors and a higher R² score.

The best model explains approximately 98% of the variation in calories burned in the test data.

## `main.py`

`main.py` provides a simple terminal-based prediction program.

The user can enter information such as:

* Age
* Gender
* Weight
* Height
* Heart rate
* Workout duration
* Workout type
* Body fat percentage
* Water intake
* Workout frequency
* Experience level

BMI is calculated automatically from weight and height.

The entered information is prepared in the same format as the training data and is then used to predict calories burned.

Example output:

```text
Predicted Calories Burned: 982 kcal
```

## `data/raw/gym_members_exercise_tracking.csv`

This folder contains the original dataset used for the project.

The raw dataset is kept unchanged.

## `pyproject.toml` and `uv.lock`

These files contain the project configuration and Python dependencies managed with `uv`.

## Technologies Used

* Python
* pandas
* matplotlib
* scikit-learn
* Jupyter Notebook
* VS Code
* Git
* GitHub
* uv



## Limitations

* The model is based only on the available dataset.
* The dataset may not represent every type of gym member or workout.
* Correlation does not prove causation.
* The prediction should not be interpreted as a medical measurement.


## Conclusion

This project demonstrates a complete supervised machine learning regression workflow, from data exploration and preprocessing to model training, evaluation, comparison, and prediction.

Linear Regression achieved the best performance, with an R² score of approximately 0.98 and an average absolute prediction error of about 30 calories.
