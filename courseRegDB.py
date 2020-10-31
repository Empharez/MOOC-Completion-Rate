import sqlite3 as sql

con = sql.connect('userData.db')
print("Database successfully open")

con.execute("CREATE TABLE student_registration (last_name TEXT, first_name TEXT,"
            "  email VARCHAR, phone_number VARCHAR, username VARCHAR, password VARCHAR)")
print("Table created successfully")
con.close()