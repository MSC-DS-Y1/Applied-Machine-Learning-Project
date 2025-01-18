from prepare_data import DataPreparation
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

models = {
    "SVM": joblib.load("../models/fabian/svm_model.pkl"),
    # "Logistic Regression": joblib.load("../models/nasim/logistic_model.pkl"),
    # "Random Forest": joblib.load("../models/gianluca/random_forest_model.pkl"),
    # "Naïve Bayes": joblib.load("../models/zhehao/naive_bayes_model.pkl"),
    # "KNN": joblib.load("../models/jeantide/knn_model.pkl"),
}

def predict(
    marital_status, course, nationality, father_qualif, mother_occ, admission_grade,
    tuition_fees_up_to_date, scholarship, age_at_enrollment, units_1_credited,
    units_1_approved, units_2_enrolled, units_2_evaluations, units_2_approved,
    units_2_no_evaluations, unemployment_rate, inflation_rate, actual_output
) -> pd.DataFrame:
    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        "marital_status": [marital_status],
        "course": [course],
        "nationality": [nationality],
        "father_qualif": [father_qualif],
        "mother_occ": [mother_occ],
        "admission_grade": [admission_grade],
        "tuition_fees_up_to_date": [tuition_fees_up_to_date],
        "scholarship": [scholarship],
        "age_at_enrollment": [age_at_enrollment],
        "units_1_credited": [units_1_credited],
        "units_1_approved": [units_1_approved],
        "units_2_enrolled": [units_2_enrolled],
        "units_2_evaluations": [units_2_evaluations],
        "units_2_approved": [units_2_approved],
        "units_2_no_evaluations": [units_2_no_evaluations],
        "unemployment_rate": [unemployment_rate],
        "inflation_rate": [inflation_rate]
    })

    # Prepare the data
    data_prep = DataPreparation(input_data)
    df_pca = data_prep.apply_pca()

    predictions = pd.DataFrame({
        name: model.predict(df_pca) for name, model in models.items()
    }, columns=models.keys(), index=[0])

    return predictions

def plot_predictions(predictions: pd.DataFrame):
    """Generate a bar chart for model predictions."""
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(predictions.keys()), y=list(predictions.values[0]))
    plt.title("Model Predictions")
    plt.ylabel("Predicted Value")
    plt.xlabel("Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def load_options(file_path):
    marital_status_options = []
    course_options = []
    nationality_options = []
    qualifications_options = []
    occupations_options = []

    with open(file_path, 'r') as f:
        options = json.load(f)
        marital_status_options = [(option['name'], option['value']) for option in options['marital_status_options']]
        course_options = [(option['name'], option['value']) for option in options['course_options']]
        nationality_options = [(option['name'], option['value']) for option in options['nationality_options']]
        qualifications_options = [(option['name'], option['value']) for option in options['qualifications_options']]
        occupations_options = [(option['name'], option['value']) for option in options['occupations_options']]

        # Sort the options
        marital_status_options.sort(key=lambda x: x[0])
        course_options.sort(key=lambda x: x[0])
        nationality_options.sort(key=lambda x: x[0])
        qualifications_options.sort(key=lambda x: x[0])
        occupations_options.sort(key=lambda x: x[0])

    return marital_status_options, course_options, nationality_options, qualifications_options, occupations_options

def main():
    marital_status_options, course_options, nationality_options, qualifications_options, occupations_options = load_options("options.json")

    inputs = [
        gr.Dropdown(choices=marital_status_options, label="Marital Status"),
        gr.Dropdown(choices=course_options, label="Course"),
        gr.Dropdown(choices=nationality_options, label="Nationality"),
        gr.Dropdown(choices=qualifications_options, label="Father's Qualification"),
        gr.Dropdown(choices=occupations_options, label="Mother's Occupation"),
        gr.Number(label="Admission Grade"),
        gr.Radio(choices=[("Yes", 1), ("No", 0)], label="Tuition Fees Up-to-Date", type="index", value=0),
        gr.Radio(choices=[("Yes", 1), ("No", 0)], label="Scholarship Holder", type="index", value=0),
        gr.Slider(minimum=18, maximum=65, step=1, label="Age at Enrollment"),  # Age Slider
        gr.Number(label="Units 1 Credited"),
        gr.Number(label="Units 1 Approved"),
        gr.Number(label="Units 2 Enrolled"),
        gr.Number(label="Units 2 Evaluations"),
        gr.Number(label="Units 2 Approved"),
        gr.Number(label="Units 2 No Evaluations"),
        gr.Number(label="Unemployment Rate"),
        gr.Number(label="Inflation Rate"),
        gr.Dropdown(choices=["Dropout", "Enrolled", "Graduate"], label="Actual Output"),
    ]

    output = [
        gr.Dataframe(headers=list(models.keys()), datatype="number", col_count=len(models), row_count=1),
        gr.Plot(label="Prediction Bar Chart"),
    ]

    def predict_and_plot(*args):
        df_predictions = predict(*args)
        bar_chart = plot_predictions(df_predictions)
        return df_predictions, bar_chart

    gr.Interface(
        fn=predict_and_plot,
        inputs=inputs,
        outputs=output,
        title="Student Data Prediction",
        description="Enter student data to get a prediction based on the trained model. Compare results visually.",
        theme=gr.themes.Ocean(),
    ).launch()

if __name__ == "__main__":
    main()
