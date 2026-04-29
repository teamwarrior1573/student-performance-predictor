from flask import Flask, render_template, request, session
import joblib
import pandas as pd

app = Flask(__name__)
app.secret_key = "atharv123"

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/step2", methods=["POST"])
def step2():
    session["name"] = request.form["name"]
    session["age"] = request.form["age"]
    return render_template("step2.html")


@app.route("/step3", methods=["POST"])
def step3():
    session.update(request.form)
    return render_template("step3.html")


@app.route("/step4", methods=["POST"])
def step4():
    session.update(request.form)
    return render_template("step4.html")


@app.route("/predict", methods=["POST"])
def predict():
    session.update(request.form)

    data = {}

    for col in columns:
        val = session.get(col, 0)

        try:
            data[col] = float(val)
        except:
            data[col] = 0

    df = pd.DataFrame([data])

    # Raw Prediction
    score = model.predict(df)[0]

    # Smart Calibration for strong academic inputs
    academic = (
        float(session.get("Hours_Studied", 0)) +
        float(session.get("Attendance", 0)) / 10 +
        float(session.get("Previous_Scores", 0)) / 10
    )

    if academic > 22:
        score += 8
    elif academic > 18:
        score += 5
    elif academic > 14:
        score += 3

    # Bonus for motivation + support
    support = (
        float(session.get("Motivation_Level", 0)) +
        float(session.get("Parental_Involvement", 0))
    )

    if support > 14:
        score += 3
    elif support > 10:
        score += 1.5

    # Final Limit
    score = max(0, min(score, 100))

    # Grade Logic
    if score >= 85:
        grade = "Excellent"
    elif score >= 70:
        grade = "Good"
    elif score >= 50:
        grade = "Average"
    else:
        grade = "Poor"

    return render_template(
        "result.html",
        name=session["name"],
        score=round(score, 2),
        grade=grade
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)