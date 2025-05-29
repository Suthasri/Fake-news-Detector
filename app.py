from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["news"]
    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)
    result = "REAL" if prediction[0] == 1 else "FAKE"
    return render_template("index.html", prediction_text=f"The news is: {result}")

if __name__ == "__main__":
    app.run(debug=True)