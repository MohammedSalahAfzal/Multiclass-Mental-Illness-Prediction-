from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import pickle
import os
import json
import re
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load ML Model + Tokenizer + Label Encoder
model = load_model("emotion_lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Mental illness → image mapping
image_map = {
    "Depression": "depression.jpg",
    "Anxiety": "anxiety.jpg",
    "Bipolar Disorder": "bipolar.jpg",
    "PTSD": "ptsd.jpg"
}

# Users stored in users.json
USER_FILE = 'users.json'
if not os.path.exists(USER_FILE):
    with open(USER_FILE, 'w') as f:
        json.dump([], f)

def load_users():
    with open(USER_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register')
def register_page():
    show_login = request.args.get('form') == 'login'
    return render_template("register.html", show_login=show_login)

@app.route('/register_user', methods=['POST'])
def register_user():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']

    users = load_users()
    if any(u['email'] == email for u in users):
        flash("User already exists. Please login.")
        return redirect(url_for('register_page', form='login'))

    users.append({'name': name, 'email': email, 'password': password})
    save_users(users)
    flash("Registration successful. Please login.")
    return redirect(url_for('register_page', form='login'))

@app.route('/login_user', methods=['POST'])
def login_user():
    email = request.form['email']
    password = request.form['password']

    users = load_users()
    for u in users:
        if u['email'] == email and u['password'] == password:
            session['username'] = u['name']
            return redirect(url_for('predict'))

    flash("Invalid login credentials.")
    return redirect(url_for('register_page', form='login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        flash("Please login first.")
        return redirect(url_for('register_page', form='login'))

    result = None
    image_file = None

    if request.method == 'POST':
        user_input = request.form['text']
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100)

        # Predict probabilities
        pred_probs = model.predict(padded)[0]
        max_prob = np.max(pred_probs)
        pred_index = np.argmax(pred_probs)

        # Confidence threshold
        if max_prob < 0.6:
            result = "Input not related to any known mental illness"
            image_file = "notfound.jpg"
        else:
            result = le.inverse_transform([pred_index])[0]
            image_file = image_map.get(result, "default.jpg")

    return render_template("predict.html", prediction=result, image_file=image_file, user=session['username'])


@app.route('/chart')
def chart():
    return render_template("chart.html")

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out successfully.")
    return redirect(url_for('register_page', form='login'))

if __name__ == '__main__':
    app.run(debug=True)
