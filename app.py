from flask import Flask, render_template, request, session
import joblib
import re
import string

app = Flask(__name__)
app.secret_key = "fake_news_secret_key"
@app.before_request
def initialize_session():
    if "fake" not in session:
        session["fake"] = 0
    if "real" not in session:
        session["real"] = 0

# ======================
# LOAD MODEL
# ======================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ======================
# TEXT CLEANING
# ======================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# ======================
# HOME PAGE
# ======================
@app.route("/")
def home():
    return render_template(
        "index.html",
        fake=session.get("fake", 0),
        real=session.get("real", 0)
    )

# ======================
# PREDICT ROUTE
# ======================
@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    cleaned_news = clean_text(news)

    vector_input = vectorizer.transform([cleaned_news])
    prediction = model.predict(vector_input)

    fake = session.get("fake", 0)
    real = session.get("real", 0)

    if prediction[0] == 1:
        result = "Real News ✅"
        real += 1
    else:
        result = "Fake News ❌"
        fake += 1

    session["fake"] = fake
    session["real"] = real

    return render_template(
        "index.html",
        prediction=result,
        fake=fake,
        real=real
    )

# ======================
# RESET COUNTER (FIX GRAPH ISSUE)
# ======================
@app.route("/reset")
def reset():
    session["fake"] = 0
    session["real"] = 0
    return "Reset Done ✅ Now go back to homepage"

# ======================
# RUN APP
# ======================
if __name__ == "__main__":
    app.run(debug=True)