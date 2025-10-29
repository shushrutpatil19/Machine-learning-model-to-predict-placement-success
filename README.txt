# Machine Learning Model to Predict Placement Success (v2)

This project predicts whether a student will be placed based on academic and activity-related features.

## 📊 Dataset Columns
- CGPA
- Communication_Skills
- Internship_Activity (Yes/No)
- Internship_Type (Technical/Non-Technical/None)
- Internship_Company (Startup/MNC/None)
- Extracurricular_Activities
- Placed (1=Yes, 0=No)

## 🚀 How to Run

1️⃣ Install required libraries:
```
pip install pandas numpy scikit-learn streamlit joblib
```

2️⃣ Train the model (optional, already trained):
```
python placement_model_v2.py
```

3️⃣ Launch the web app:
```
streamlit run app.py
```

Then open the displayed local URL (usually http://localhost:8501) in your browser.

Enjoy predicting placement success!
