import pandas as pd
from sklearn.model_selection import GridSearchCV
import plotly
import plotly.express as px
import pickle

data = pd.read_csv("data/healthcare_stroke.csv")
X_test = data.drop("stroke", axis=1)

gs_model = pickle.load(open("data/stroke_algorithm_RFC3.pkl", "rb"))


def predict_stroke(patient):
    predict_patient = X_test[X_test['id'] == int(patient)]
    predict_patient.drop(['id', 'ever_married', 'Residence_type'], axis=1)
    prediction = gs_model.predict_proba(predict_patient)
    prediction[0][0] = round((prediction[0][0] * 100), 2)
    prediction[0][1] = round((prediction[0][1] * 100), 2)
    return prediction


def get_patient(patient):
    patient_data = X_test[X_test['id'] == int(patient)]
    patient_data['age'] = patient_data['age'].astype(int)
    patient_data['hypertension'] = patient_data['hypertension'].astype(bool)
    patient_data['heart_disease'] = patient_data['heart_disease'].astype(bool)
    patient_data['avg_glucose_level'] = patient_data['avg_glucose_level'].astype(int)
    return patient_data


def patients_on_floor():
    patients = X_test.sample(12)
    patients['age'] = patients['age'].astype(int)
    patients['hypertension'] = patients['hypertension'].astype(bool)
    patients['heart_disease'] = patients['heart_disease'].astype(bool)
    patients['avg_glucose_level'] = patients['avg_glucose_level'].astype(int)
    return patients


def getPlots():
    features = ["hypertension", "heart_disease", "bmi", "avg_glucose_level"]
    fig = px.scatter_matrix(
        data,
        dimensions=features,
        color="stroke"
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()
    return fig
