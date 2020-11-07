from flask import Flask, render_template, url_for, request, flash, redirect
import sqlite3
from flask_bootstrap import Bootstrap
import pickle
import pandas as pd

pickle_in = open('rf_model.pkl', 'rb')
model = pickle.load(pickle_in)

app = Flask(__name__, static_url_path='/static')
Bootstrap(app)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/addrecord', methods=['POST', 'GET'])
def add_record():
    if request.method == 'POST':
        try:
            last_name = request.form['last_name']
            first_name = request.form['first_name']
            email = request.form['email']
            phone_number = request.form['phone_number']
            username = request.form['username']
            password = request.form['password']
            with sqlite3.connect("userData.db") as con:
                cur = con.cursor()
                cur.execute(
                    "INSERT into users (last_name, first_name, email, phone_number, username, password)VALUES(?,?,?,?,?,?)",
                    (last_name, first_name, email, phone_number, username, password))
                con.commit()
                msg = "Registration Complete."
        except:
            con.rollback()
            msg = "Unable to complete registration."
            return render_template("home.html", msg=msg)
        finally:
            return render_template("login.html", msg=msg)


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/getinfo', methods=['POST', 'GET'])
def getinfo():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['password']
            with sqlite3.connect("userData.db") as con:
                cur = con.cursor()
                cur.execute("SELECT * FROM users WHERE username = '%s' AND password = '%s'" % (username, password))
                if cur.fetchone() is not None:
                    return render_template("courseRegistration.html")
                else:
                    msg = "Login Failed."
                    return render_template("login.html", msg=msg)
        except:
            error = "Try again."
            return render_template("login.html", error=error)


@app.route('/course-reg')
def course_registration():
    return render_template('courseRegistration.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        last_name = request.form['last_name']
        first_name = request.form['first_name']
        gender = request.form['gender']
        age_group = request.form['age_group']
        disability = request.form['disability']
        highest_education = request.form['highest_education']
        phone_number = request.form['phone_number']
        email = request.form['email']
        street_address = request.form['street_address']
        street_address_line_2 = request.form['street_address_line_2']
        city = request.form['city']
        state = request.form['state']
        zip_code = request.form['zip_code']
        country = request.form['country']
        course = request.form['course']
        num_of_prev_attempts = request.form['num_of_prev_attempts']

        input_variables = pd.DataFrame(
            [[gender, highest_education, age_group, num_of_prev_attempts, disability]],
            columns=[gender, highest_education, age_group, num_of_prev_attempts, disability],
            dtype=int)
        result = model.predict(input_variables)[0]
        prediction = int(result)
        if prediction == 1:
            prediction = "Complete"
        else:
            prediction = "dropout"
        if gender == 1:
            gender = "Male"
        else:
            gender = "Female"
        if age_group == 1:
            age_group = "0-35"
        elif age_group == 2:
            age_group = "35-55"
        else:
            age_group = "55>="
        if disability == 2:
            disability = "Yes"
        else:
            disability = "No"
        if highest_education == 1:
            highest_education = "No Formal Education"
        elif highest_education == 2:
            highest_education = "Lower Than A Level (Primary School and SSCE)"
        elif highest_education == 3:
            highest_education = "A Level or Equivalent"
        elif highest_education == 4:
            highest_education = "HE Qualification"
        else:
            age_group = "Post Graduate Qualification"


        with sqlite3.connect("courseRegDB.db") as con:
            cur = con.cursor()
            cur.execute("INSERT into course_registration (last_name, first_name ,gender, age_group, disability,"
                        " highest_education, phone_number, email, street_address, street_address_line_2, city,"
                        " state,  zip_code, country, course, num_of_prev_attempts, prediction)"
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (last_name, first_name, gender, age_group, disability, highest_education, phone_number,
                         email, street_address, street_address_line_2, city, state, zip_code,
                         country, course, num_of_prev_attempts, prediction))
            con.commit()

    return render_template("response.html")


@app.route('/response-page')
def response_page():
    return render_template('response.html')


@app.route('/admin')
def admin():
    return render_template('admin.html')


@app.route('/adminlogin', methods=['POST'])
def adminlogin():
    if request.form['password'] == 'admin1234' and request.form['username'] == 'admin':
        return redirect('/dashboard')
    else:
        flash('wrong password!')


@app.route('/dashboard')
def dashboard():
    con = sqlite3.connect("userData.db")
    cur = con.cursor()
    cur.execute("SELECT * FROM users")
    rows = cur.fetchall()

    conn = sqlite3.connect("courseRegDB.db")
    curr = conn.cursor()
    curr.execute("SELECT * FROM course_registration")
    data = curr.fetchall()
    return render_template("dashboard.html", rows=rows, data=data)



@app.route('/logout')
def logout():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
