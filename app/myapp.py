from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np

app = Flask(__name__)

# 피클 파일에서 훈련된 모델을 로드
with open('RFR_GYM_model_v2.pkl', 'rb') as f:
    model = pickle.load(f)

# 피클 파일에서 스케일러 객체 로드
with open('RFR_GYM_scaler_v2.pkl', 'rb') as f:
    scaler = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        "day_of_week": request.form['day_of_week'],
        "is_weekend": request.form['is_weekend'],
        "is_holiday": request.form['is_holiday'],
        "temperature": request.form['temperature'],
        "is_start_of_semester": request.form['is_start_of_semester'],
        "is_during_semester": request.form['is_during_semester'],
        "month": request.form['month'],
        "hour": request.form['hour'],
        "day": request.form['day'],
        "minute": request.form['minute']
    }

    # json으로 변환
    input_json = json.dumps(input_data)
    input_data_arr = np.array(list(input_data.values())).astype(float)
    test_predict = scaler.transform(input_data_arr.reshape(1, -1))

    # 예측 수행
    prediction = int(model.predict(test_predict))

    #웹 페이지 리턴
    return render_template('prediction.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)


    