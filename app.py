import pickle
from flask import Flask,request,app,jsonify,render_template
from src.pipelines.Prediction_Pipleline import CustomData, PredictPipeline
import numpy as np

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

# crim,zn,indus,chas
# 0.00632,18,2.31,0

# nox,rm,age,dis
# 0.538,6.575,65.2,4.09

# rad,tax,ptratio,b,lstat
# 1,296,15.3,396.9,4.98

# model = pickle.load(open('Artifacts/model.pkl','rb'))
# scaler = pickle.load(open('Artifacts/preprocessor.pkl','rb'))



# model_path = os.path.join("Artifacts", "model.pkl")
# preprocessor_path = os.path.join("Artifacts", "preprocessor.pkl")

# @app.route('/')
# def home():
#     return render_template('home.html')





# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data = request.json['data']
#     print(data)
#     a_data = (np.array(list(data.values())).reshape(1,-1))
#     new_data = scaler.transform(a_data)
#     output = model.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])

# @app.route('/predict',methods=['POST'])
# def predict():
#     data=[float(x) for x in request.form.values()]
#     print("data")
#     print(data)
#     final_input=scaler.transform(np.array(data).reshape(1,-1))
#     print(final_input)
#     output=model.predict(final_input)[0]
#     return render_template("home.html",prediction_text="The House price prediction is {}".format(output))


# @app.route("/", methods=["GET", "POST"])
# def predict_datapoints():
#     if request.method == "GET":
#         return render_template("index.html")
#     else:
#         # data=CustomData( 
#         #     carat=float(request.form.get('carat')),
#         #     cut = request.form.get('cut')
#         # )
#         # this is my final data
#         # final_data=data.get_data_as_dataframe()
#         # predict_pipeline=PredictPipeline()
#         # pred=predict_pipeline.predict(final_data)
#         # result=round(pred[0],2)
#         # return render_template("result.html",final_result=result)
#         return 

