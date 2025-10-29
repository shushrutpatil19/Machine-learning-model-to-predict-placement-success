import pandas as pd #pandas:for working with tables(datasets)
import numpy as np #numpy:for math operation
from sklearn.model_selection import train_test_split #train_test_split:to split data into training and testing parts
from sklearn.preprocessing import LabelEncoder, StandardScaler #labelencoder: to convert text(like"yes"/"no")into numbers.#standardscaler:to scale your data so all features are on a similar range.
from sklearn.ensemble import RandomForestClassifier #randomforestclassifier:the machine learning model used to predict placement.
from sklearn.metrics import accuracy_score, classification_report #accuracy:to check how well the model performs.
import joblib #joblib:to save and load the trained moadel later.

# Load dataset
data = pd.read_csv("placement_data_v2.csv")

X = data.drop('Placed', axis=1)
y = data['Placed']

le = LabelEncoder()
for col in ['Internship_Activity', 'Internship_Type', 'Internship_Company']:
    X[col] = le.fit_transform(X[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "placement_model_v2.pkl")
joblib.dump(scaler, "scaler_v2.pkl")

def predict_placement(cgpa, comm, intern_act, intern_type, intern_company, extra):
    input_df = pd.DataFrame([{
        'CGPA': cgpa,
        'Communication_Skills': comm,
        'Internship_Activity': intern_act,
        'Internship_Type': intern_type,
        'Internship_Company': intern_company,
        'Extracurricular_Activities': extra
    }])

    for col in ['Internship_Activity', 'Internship_Type', 'Internship_Company']:
        input_df[col] = le.fit_transform(input_df[col])

    input_scaled = scaler.transform(input_df)
    result = model.predict(input_scaled)[0]
    return "Placed" if result == 1 else "Not Placed"

print("\nExample Prediction:")
print(predict_placement(8.8, 85, 'Yes', 'Technical', 'MNC', 75))
