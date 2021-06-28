from flask import Flask, render_template, request
import os
from prediction import patients_on_floor, predict_stroke, get_patient, getPlots




app = Flask(__name__)

picFolder = os.path.join('static', 'dataImg')
app.config['UPLOAD_FOLDER'] = picFolder

@app.route("/", methods=['GET', 'POST'])
def home():
    dropdown_list = patients_on_floor()
    dropdown_select = request.form.get('selectPatient')
    if dropdown_select is not None:
        stroke_prediction = predict_stroke(dropdown_select) # Predict on patient
        patient = get_patient(dropdown_select) # Gets the patient to send to predict
        return render_template("prediction.html", patient=patient.to_numpy(), proba=stroke_prediction)
    else:
        return render_template("home.html", dropdown_list=dropdown_list.to_numpy())

@app.route("/data")
def data():
    imageList = os.listdir('static/dataImg')
    imageList = ['dataImg/' + image for image in imageList]
    return render_template("data.html", imageList=imageList)


if __name__ == "__main__":
    app.run()


