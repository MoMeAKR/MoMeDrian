import json 
import os 
from flask import Flask, jsonify
from flask_cors import CORS
import mome_audio


app = Flask(__name__)
CORS(app)
@app.route('/mome_audio/get_content')
def get_content():
    
    # Update 
    mome_audio.translate_current_structure()

    # Loading 
    content = open(os.path.join(os.path.dirname(__file__), "my_test.txt")).read()
    data = {"strudel_data": content}
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)