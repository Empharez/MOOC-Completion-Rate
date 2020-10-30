from flask import Flask, render_template,url_for, request, flash, redirect
import sqlite3
from flask_bootstrap import Bootstrap
import pickle
import pandas as pd

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
                cur.execute("INSERT into users (last_name, first_name, email, phone_number, username, password)VALUES(?,?,?,?,?,?)",
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

@app.route('/getinfo', methods=['POST','GET'])
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

if __name__ == '__main__':
    app.run(debug=True)