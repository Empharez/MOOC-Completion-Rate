import sqlite3 as sql

con = sql.connect('courseRegDB.db')
print("Database successfully open")

con.execute("CREATE TABLE course_registration (last_name TEXT, first_name TEXT,"
            "  gender VARCHAR, age_group VARCHAR, disability VARCHAR, highest_education VARCHAR, "
            " phone_number INT, email VARCHAR, street_address VARCHAR, street_address_line_2 VARCHAR,"
            " city VARCHAR, state VARCHAR,  zip_code INT, country VARCHAR, course VARCHAR, num_of_prev_attempts INT, prediction INT)")
print("Table created successfully")
con.close()