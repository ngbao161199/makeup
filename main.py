import flask
from flask import Flask,render_template, request, flash, request, redirect, url_for, send_from_directory
import cv2
import requests
import numpy as np
import os
import time
import json
import string
import makeup_src
from makeup_src import lip_makeup, blush_makeup
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/image'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
INPUT_FILE_NAME = "input.png"
#OUTPUT_FILE_NAME = "tmp/output.jpg"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#app = flask.Flask(__name__)

''' API Run locally
@app.route('/makeup', methods=['GET','POST'])
def makeup():
    if request.method == "POST":
        # input form - hex code , choice (0: lips, 1: skin), image
        hex_code = request.form["hex"]
        choice = request.form["select"]
        binStrImg = request.files["input_image"].read()

        # process image
        npimg = np.fromstring(binStrImg, np.uint8)
        img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)

        # process hex code
        hex_code = (str)(hex_code)
        hex_code = hex_code.lstrip('#')
        hex_to_rbg = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

        # choice 
        choice = (int)(choice)
        
        
        start = time.time()
        # lips case
        if choice == 0:
            output_img_path = lip_makeup(img, hex_to_rbg)
        # skin case
        else:
            output_img_path = blush_makeup(img, hex_to_rbg)
        est = time.time() - start

        # response status
        response = {
            "out_file path": output_img_path,
			"time_run": est
        }
        
        return jsonify(response)
'''

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # input form - hex code , choice (0: lips, 1: skin), image
            hex_code = request.form["hex"]
            choice = request.form["select"]
            binStrImg = request.files["file"].read()

            filename = secure_filename(file.filename)
            input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], INPUT_FILE_NAME)

            

            # process image
            npimg = np.fromstring(binStrImg, np.uint8)
            img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
            cv2.imwrite(input_file_path, img)

            # process hex code
            hex_code = (str)(hex_code)
            hex_code = hex_code.lstrip('#')
            hex_to_rbg = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

            # choice 
            choice = (int)(choice)
            
            
            start = time.time()
            # lips case
            if choice == 0:
                output_img_path = lip_makeup(img, hex_to_rbg)
            # skin case
            else:
                output_img_path = blush_makeup(img, hex_to_rbg)
            
            est = time.time() - start
            
            return render_template("succeedscan.html")
    return render_template("home.html")

@app.route('/examples')
def examples():
    return render_template("examples.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=True)

#app.run(host="0.0.0.0", port=5000, debug=True)
