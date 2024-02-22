from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_mail = request.form['email']  # Getting the input email from the form
    
    # Convert text to feature vectors
    input_data_features = vectorizer.transform([input_mail])
    
    # Making prediction
    prediction = model.predict(input_data_features)[0]
    
    if prediction == 1:
        result = 'Ham mail'
    else:
        result = 'Spam mail'
        
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run()
