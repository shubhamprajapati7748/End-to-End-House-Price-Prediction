from flask import Flask,request,app,jsonify,render_template
from src.pipelines.Prediction_Pipleline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def predict_datapoints():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            crim = request.form.get("crim"),
            zn = request.form.get("zn"),
            indus = request.form.get("indus"),
            chas = request.form.get("chas"),
            nox = request.form.get("nox"),
            rm = request.form.get("rm"),
            age = request.form.get("age"),
            dis = request.form.get("dis"),
            rad = request.form.get("rad"),
            tax = request.form.get("tax"),
            ptratio = request.form.get("ptratio"),
            b = request.form.get("b"),
            lstat = request.form.get("lstat"),
        )

        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        return render_template('index.html', results=result[0])

## execution begin 
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)