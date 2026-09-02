from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
import mysql.connector
import models

app = Flask(__name__)

FIELDS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
          "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# ---------------- MySQL config (change these to match your MySQL) ----------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",       # <-- your MySQL password
    "database": "heart_db",
}


def get_db():
    return mysql.connector.connect(**DB_CONFIG)


def init_db():
    # create database + table if they don't exist
    con = mysql.connector.connect(
        host=DB_CONFIG["host"], user=DB_CONFIG["user"], password=DB_CONFIG["password"])
    cur = con.cursor()
    cur.execute("CREATE DATABASE IF NOT EXISTS heart_db")
    cur.execute("USE heart_db")
    cur.execute("""CREATE TABLE IF NOT EXISTS history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100), age FLOAT, sex FLOAT, risk VARCHAR(20), probability FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    # add name column if an older table already existed without it
    try:
        cur.execute("ALTER TABLE history ADD COLUMN name VARCHAR(100) AFTER id")
    except mysql.connector.Error:
        pass
    con.commit()
    cur.close()
    con.close()


init_db()


# ---------------- Routes ----------------
@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "123":
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    con = get_db()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM history")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM history WHERE risk = 'High'")
    high = cur.fetchone()[0]
    cur.close()
    con.close()
    return render_template("dashboard.html", user="admin", total=total, high=high)


@app.route("/logout")
def logout():
    return redirect(url_for("login"))


@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")



@app.route("/predict", methods=["POST"])
def predict():
    try:
        name = request.form.get("name", "").strip() or "Unknown"
        data = [float(request.form[f]) for f in FIELDS]
        result = models.predict(data)

        # save to database
        con = get_db()
        cur = con.cursor()
        cur.execute("INSERT INTO history (name, age, sex, risk, probability) VALUES (%s, %s, %s, %s, %s)",
                    (name, data[0], data[1], result["risk"], result["avg_prob"]))
        con.commit()
        cur.close()
        con.close()

        return render_template("result.html", name=name, age=int(data[0]),
                               sex="Male" if data[1] == 1 else "Female",
                               date=datetime.now().strftime("%d %b %Y, %I:%M %p"), **result)
    except Exception as e:
        return f"Error: {str(e)}"


@app.route("/history")
def history():
    con = get_db()
    cur = con.cursor()
    cur.execute("SELECT id, name, age, sex, risk, probability, created_at FROM history ORDER BY id DESC")
    rows = cur.fetchall()
    cur.close()
    con.close()
    return render_template("history.html", rows=rows)


@app.route("/delete/<int:record_id>")
def delete(record_id):
    con = get_db()
    cur = con.cursor()
    cur.execute("DELETE FROM history WHERE id = %s", (record_id,))
    con.commit()
    cur.close()
    con.close()
    return redirect(url_for("history"))



if __name__ == "__main__":
    app.run(debug=True)
