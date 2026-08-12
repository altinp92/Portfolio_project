import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


df = pd.read_csv("data/raw/gym_members_exercise_tracking.csv")

X = df.drop(columns=["Calories_Burned"])
y = df["Calories_Burned"]


categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), numerical_columns),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)


linear_model = Pipeline(
    steps=[
        ("data_preparing", preprocessor),
        ("model", LinearRegression())
    ]
)


linear_model.fit(X, y)




print("Calories Burned Prediction")

age = int(input("Enter your age: "))
gender = input("Enter your gender (Male/Female): ")

weight = float(input("Enter your weight in kg: "))
height = float(input("Enter your height in meters: "))

while height < 1.0 or height > 2.5:
    print("Please enter height in meters, for example 1.59")
    height = float(input("Enter your height in meters: "))

max_bpm = int(input("Enter your maximum heart rate: "))
avg_bpm = int(input("Enter your average heart rate: "))
resting_bpm = int(input("Enter your resting heart rate: "))

session_duration = float(input("Enter workout duration in hours: "))
workout_type = input("Enter workout type (Cardio/HIIT/Strength/Yoga): ")

fat_percentage = float(input("Enter your body fat percentage: "))
water_intake = float(input("Enter your water intake in liters: "))

workout_frequency = int(input("Enter workout frequency in days per week: "))
experience_level = int(input("Enter experience level (1/2/3): "))

bmi = weight / (height ** 2)

new_workout = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Weight (kg)": [weight],
    "Height (m)": [height],
    "Max_BPM": [max_bpm],
    "Avg_BPM": [avg_bpm],
    "Resting_BPM": [resting_bpm],
    "Session_Duration (hours)": [session_duration],
    "Workout_Type": [workout_type],
    "Fat_Percentage": [fat_percentage],
    "Water_Intake (liters)": [water_intake],
    "Workout_Frequency (days/week)": [workout_frequency],
    "Experience_Level": [experience_level],
    "BMI": [bmi]
})

predicted_calories = linear_model.predict(new_workout)[0]
print("\nPredicted Calories Burned:", round(predicted_calories), "kcal")