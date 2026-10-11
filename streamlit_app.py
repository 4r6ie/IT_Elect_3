import streamlit as st
import pandas as pd
from joblib import load

model = load("student_pass_model.joblib")

st.title("Student Pass Predictor")

name = st.text_input("Student Name")
hours = st.number_input("Study Hours per Week", min_value=0, max_value=40, value=5)

if st.button("Predict"):
    prediction = model.predict(pd.DataFrame({"StudyHours": [hours]}))[0]

    st.success(f"{name}, your predicted result is: {prediction}")

    st.subheader("How this prediction was generated")
    st.write(
        "A K-Nearest Neighbors (KNN) model was trained on past student data "
        "containing study hours and pass/fail results, then saved to "
        "student_pass_model.joblib. Your study hours were compared to the "
        "closest training examples, and the majority class among those "
        "neighbors became the prediction."
    )

    hours_range = list(range(1, 15))
    preds = model.predict(pd.DataFrame({"StudyHours": hours_range}))
    numeric = [1 if p == "Pass" else 0 for p in preds]
    chart_data = pd.DataFrame({
        "Study Hours": hours_range,
        "Predicted Class (1=Pass, 0=Fail)": numeric,
    })
    st.subheader("Study Hours vs Predicted Class")
    st.line_chart(chart_data.set_index("Study Hours"))
