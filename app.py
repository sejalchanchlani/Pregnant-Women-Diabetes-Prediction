import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pickle
from flask_mysqldb import MySQL,MySQLdb
import bcrypt

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'users'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/login',methods=["GET","POST"])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password'].encode('utf-8')

        curl = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        curl.execute("SELECT * FROM users WHERE email=%s",(email))
        user = curl.fetchone()
        curl.close()

        if len(user) > 0:
            if bcrypt.hashpw(password, user["password"].encode('utf-8')) == user["password"].encode('utf-8'):
                session['name'] = user['name']
                session['email'] = user['email']
                return render_template("lolo.html")
            else:
                return "Error password and email not match"
        else:
            return "Error user not found"
    else:
        return render_template("login.html")

@app.route('/logout', methods=["GET", "POST"])
def logout():
    session.clear()
    return render_template("index.html")

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == 'GET':
        return render_template("#register")
    else:
        name = request.form['name']
        email = request.form['email']
        mobile = request.form['mobile']
        state = request.form['state']
        city = request.form['city']
        password = request.form['password'].encode('utf-8')
        hash_password = bcrypt.hashpw(password, bcrypt.gensalt())
        repassword = request.form['repassword']

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (name, email, mobile, state, city, password, repassword) VALUES (%s,%s,%s,%s,%s,%s,%s)",(name,email,mobile,state,city,hash_password,repassword))
        mysql.connection.commit()
        session['name'] = request.form['name']
        session['email'] = request.form['email']
        return redirect(url_for('lolo'))


@app.route('/lolo')
def lolo():
    return render_template('lolo.html')

@app.route('/prediction')
def prediction():
    return render_template('pred.html')

@app.route('/fp')
def fp():
    return render_template('fp.html')

# prediction function 
def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1, 7) 
    #model = pickle.load(open("model.pkl", "rb")) 
    result = model.predict(to_predict) 
    return result[0] 

@app.route('/predict', methods = ['POST']) 
def predict(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        #to_predict_list = list(map(int, to_predict_list)) 
        result = ValuePredictor(to_predict_list)         
        if result== 1: 
            prediction_text ='Yes'
            return render_template("result.html", prediction_text=prediction_text)

        else: 
            prediction_text ='No'  
            return render_template("resultno.html", prediction_text=prediction_text)

        #return render_template("result.html", prediction_text=prediction_text)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
        app.secret_key = "^A%DJAJU^JJ123"
        app.run(debug=True,use_reloader=False)