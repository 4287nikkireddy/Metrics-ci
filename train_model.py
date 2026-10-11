
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create illustrative student-result data
records = []

for study_hours in range(1, 11):
    for attendance in range(35, 100, 5):
        result = int(study_hours >= 5 and attendance >= 65)

        records.append({
            "study_hours": study_hours,
            "attendance": attendance,
            "result": result
        })

df = pd.DataFrame(records)

X = df[["study_hours", "attendance"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save the trained model
joblib.dump(model, "student_result_model.pkl")

# Save evaluation metrics
metrics = {
    "accuracy": float(accuracy),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# Save the demonstration dataset
df.to_csv("student_results.csv", index=False)

print("Model accuracy:", round(accuracy, 4))
print("Model saved: student_result_model.pkl")
print("Metrics saved: metrics.json")
print("Dataset saved: student_results.csv")
