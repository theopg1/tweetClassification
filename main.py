from joblib import load
from flask import Flask, request, render_template

joblib = load("test.joblib")

classification = {0: "hate speech", 1: "offensive language", 2: "neither"}

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    classi = 3
    texte = ""
    if request.method == 'POST':
        texte = request.form.get('txt')
        prediction = joblib.predict([texte])
        classi = classification[prediction[0]]
    return render_template("home.html",txt=texte ,type=classi)

if __name__ == '__main__':
    app.run(debug=True)