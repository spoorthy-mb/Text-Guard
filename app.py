from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
from transformers import pipeline

model = joblib.load('logistic_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

app = Flask(__name__)

emotion_mapping = {
    'anger': 'anger',
    'joy': 'joy',
    'sadness': 'sadness',
    'fear': 'fear',
    'surprise': 'surprise',
    'neutral': 'neutral'
    
}

JOY_THRESHOLD = 0.85 
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        transcript = request.form['transcript']

        result = emotion_classifier(transcript)[0]
        emotion_scores = {emotion['label'].lower(): round(emotion['score'], 4) for emotion in result}

        features = np.array([emotion_scores.get(key, 0.0) for key in emotion_mapping]).reshape(1, -1)
        features_scaled = scaler.transform(features)

        model_prediction = model.predict(features_scaled)[0]
        churn_risk = model_prediction

        if emotion_scores.get('joy', 0) > JOY_THRESHOLD:
            churn_risk = '0'  

        return redirect(url_for('result', churn_risk=churn_risk, transcript=transcript, **emotion_scores))

    return render_template('index.html')

@app.route('/result')
def result():
    churn_risk = request.args.get('churn_risk')
    transcript = request.args.get('transcript')  
    emotion_scores = {key: request.args.get(key, 0) for key in emotion_mapping}
    return render_template('result.html', churn_risk=churn_risk, transcript=transcript, emotion_scores=emotion_scores)

if __name__ == '__main__':
    app.run(debug=True)