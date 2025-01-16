from prepare_data import DataPreparation
import gradio as gr
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load("../models/fabian/svm_model.pkl")

models = {
    "SVM": joblib.load("../models/fabian/svm_model.pkl"),
    # "Logistic Regression": joblib.load("../models/nasim/logistic_model.pkl"),
    # "Random Forest": joblib.load("../models/gianluca/random_forest_model.pkl"),
    # "Naïve Bayes": joblib.load("../models/zhehao/naive_bayes_model.pkl"),
    # "KNN": joblib.load("../models/jeantide/knn_model.pkl"),
}

# Function to handle input and return prediction
def predict(
    marital_status, course, nationality, father_qualif, mother_occ, admission_grade,
    tuition_fees_up_to_date, scholarship, age_at_enrollment, units_1_credited,
    units_1_approved, units_2_enrolled, units_2_evaluations, units_2_approved,
    units_2_no_evaluations, unemployment_rate, inflation_rate
):
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

    # Drop the target column before prediction
    predictions = {
        name: model.predict(df_pca)[0] for name, model in models.items()
    }
    return f"Predictions: {predictions}"

# Dropdown options for descriptive inputs
marital_status_options = [
  ("Single", 1),
  ("Married", 2),
  ("Widower", 3),
  ("Divorced", 4),
  ("Facto Union", 5),
  ("Legally Separated", 6),
]

course_options = [
  ("Biofuel Production Technologies", 33),
  ("Animation and Multimedia Design", 171),
  ("Social Service (Evening Attendance)", 8014),
  ("Agronomy", 9003),
  ("Communication Design", 9070),
  ("Veterinary Nursing", 9085),
  ("Informatics Engineering", 9119),
  ("Equinculture", 9130),
  ("Management", 9147),
  ("Social Service", 9238),
  ("Tourism", 9254),
  ("Nursing", 9500),
  ("Oral Hygiene", 9556),
  ("Advertising and Marketing Management", 9670),
  ("Journalism and Communication", 9773),
  ("Basic Education", 9853),
  ("Management (Evening Attendance)", 9991),
]

nationality_options = [
  ("Portuguese", 1),
  ("German", 2),
  ("Spanish", 6),
  ("Italian", 11),
  ("Dutch", 13),
  ("English", 14),
  ("Lithuanian", 17),
  ("Angolan", 21),
  ("Cape Verdean", 22),
  ("Guinean", 24),
  ("Mozambican", 25),
  ("Santomean", 26),
  ("Turkish", 32),
  ("Brazilian", 41),
  ("Romanian", 62),
  ("Moldova (Republic of)", 100),
  ("Mexican", 101),
  ("Ukrainian", 103),
  ("Russian", 105),
  ("Cuban", 108),
  ("Colombian", 109),
]

qualifications_options = [
  ("Secondary Education - 12th Year of Schooling or Eq.", 1),
  ("Higher Education - Bachelor's Degree", 2),
  ("Higher Education - Degree", 3),
  ("Higher Education - Master's", 4),
  ("Higher Education - Doctorate", 5),
  ("Frequency of Higher Education", 6),
  ("12th Year of Schooling - Not Completed", 9),
  ("11th Year of Schooling - Not Completed", 10),
  ("7th Year (Old)", 11),
  ("Other - 11th Year of Schooling", 12),
  ("2nd year complementary high school course", 13),
  ("10th Year of Schooling", 14),
  ("General commerce course", 18),
  ("Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.", 19),
  ("Complementary High School Course", 20),
  ("Technical-professional course", 22),
  ("Complementary High School Course - not concluded", 25),
  ("7th year of schooling", 26),
  ("2nd cycle of the general high school course", 27),
  ("9th Year of Schooling - Not Completed", 29),
  ("8th year of schooling", 30),
  ("General Course of Administration and Commerce", 31),
  ("Supplementary Accounting and Administration", 33),
  ("Unknown", 34),
  ("Can't read or write", 35),
  ("Can read without having a 4th year of schooling", 36),
  ("Basic education 1st cycle (4th/5th year) or equiv.", 37),
  ("Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.", 38),
  ("Technological specialization course", 39),
  ("Higher education - degree (1st cycle)", 40),
  ("Specialized higher studies course", 41),
  ("Professional higher technical course", 42),
  ("Higher Education - Master (2nd cycle)", 43),
  ("Higher Education - Doctorate (3rd cycle)", 44),
]

occupations_options = [
  ("Student", 0),
  ("Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers", 1),
  ("Specialists in Intellectual and Scientific Activities", 2),
  ("Intermediate Level Technicians and Professions", 3),
  ("Administrative staff", 4),
  ("Personal Services, Security and Safety Workers and Sellers", 5),
  ("Farmers and Skilled Workers in Agriculture, Fisheries and Forestry", 6),
  ("Skilled Workers in Industry, Construction and Craftsmen", 7),
  ("Installation and Machine Operators and Assembly Workers", 8),
  ("Unskilled Workers", 9),
  ("Armed Forces Professions", 10),
  ("Other Situation", 90),
  ("(blank)", 99),
  ("Health professionals", 122),
  ("Teachers", 123),
  ("Specialists in information and communication technologies (ICT)", 125),
  ("Intermediate level science and engineering technicians and professions", 131),
  ("Technicians and professionals, of intermediate level of health", 132),
  ("Intermediate level technicians from legal, social, sports, cultural and similar services", 134),
  ("Office workers, secretaries in general and data processing operators", 141),
  ("Data, accounting, statistical, financial services and registry-related operators", 143),
  ("Other administrative support staff", 144),
  ("Personal service workers", 151),
  ("Sellers", 152),
  ("Personal care workers and the like", 153),
  ("Skilled construction workers and the like, except electricians", 171),
  ("Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like", 173),
  ("Workers in food processing, woodworking, clothing and other industries and crafts", 175),
  ("Cleaning workers", 191),
  ("Unskilled workers in agriculture, animal production, fisheries and forestry", 192),
  ("Unskilled workers in extractive industry, construction, manufacturing and transport", 193),
  ("Meal preparation assistants", 194),
]

course_options.sort(key=lambda x: x[0])
nationality_options.sort(key=lambda x: x[0])
qualifications_options.sort(key=lambda x: x[0])
occupations_options.sort(key=lambda x: x[0])

# Create the Gradio Interface
inputs = [
    gr.Dropdown(choices=marital_status_options, label="Marital Status"),
    gr.Dropdown(choices=course_options, label="Course"),
    gr.Dropdown(choices=nationality_options, label="Nationality"),
    gr.Dropdown(choices=qualifications_options, label="Father's Qualification"),
    gr.Dropdown(choices=occupations_options, label="Mother's Occupation"),
    gr.Number(label="Admission Grade"),
    gr.Dropdown(choices=[("Yes", 1), ("No", 0)], label="Tuition Fees Up-to-Date", type="index"),
    gr.Dropdown(choices=[("Yes", 1), ("No", 0)], label="Scholarship Holder", type="index"),
    gr.Slider(minimum=18, maximum=65, step=1, label="Age at Enrollment"),  # Age Slider
    gr.Number(label="Units 1 Credited"),
    gr.Number(label="Units 1 Approved"),
    gr.Number(label="Units 2 Enrolled"),
    gr.Number(label="Units 2 Evaluations"),
    gr.Number(label="Units 2 Approved"),
    gr.Number(label="Units 2 No Evaluations"),
    gr.Number(label="Unemployment Rate"),
    gr.Number(label="Inflation Rate"),
]

output = gr.Textbox(label="Prediction Result")

gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=output,
    title="Student Data Prediction",
    description="Enter student data to get a prediction based on the trained model.",
).launch()