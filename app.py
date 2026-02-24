from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load saved model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    risk = None

    if request.method == "POST":
        text = request.form["job_text"]

        vector = vectorizer.transform([text])
        result = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0][1] * 100

        if result == 1:
            prediction = "Fake Job 🚨"
        else:
            prediction = "Real Job ✅"

        risk = round(probability, 2)

    return render_template("index.html", prediction=prediction, risk=risk)

if __name__ == "__main__":
    app.run(debug=True)
