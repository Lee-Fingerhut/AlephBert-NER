from flask import Flask, request, json
from new_code import look_and_see 
#pip install -U flask-cors
from flask_cors import CORS, cross_origin

app = Flask("asd")
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# pushing info into client with post and json input with the name key
@app.route('/new_code', methods=['POST'])
def check():
    data = json.loads(request.data.decode("utf-8"))
    look_and_see(data["name"])

    return "sucess"

# @app.route('E:/NER_Project-main/APP/server/textJSON.json', methods=['POST'])
# def check():
#     data = json.loads(request.data.decode("utf-8"))
#     look_and_see(data["name"])

    return "sucess"
app.run(host='0.0.0.0', port=105)
