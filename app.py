from pathlib import Path
from flask import Flask, render_template, request, send_from_directory, redirect
from utils import is_file_allowed
# from werkzeug.utils import secure_filename

app = Flask(__name__)



# This will give us the PosixPath of the working directory
dir_path = Path('.')
# Using it we create a string path for the upload folder
upload_path = str(dir_path/'static'/'uploads')

# If we land on the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Deal with uploaded Style and Content Images
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method=='POST':
        content_image = request.files['content']
        style_image = request.files['style']
        content_filename = content_image.filename
        style_filename = style_image.filename
        if is_file_allowed(content_filename):
            content_image.save(
                upload_path + '/' + content_filename)
        if is_file_allowed(style_filename):
            style_image.save(
                upload_path + '/' + style_filename)
        else:
            print("Check the file extention you've uploaded")
            return render_template('index.html')
        return render_template(
            'upload.html', 
            content=content_image.filename,
            style=style_image.filename)

# Serve the uploaded files
@app.route('/upload/<filename>')
def serve_image(filename):
    return send_from_directory(
        directory='static/uploads',
        filename=filename)

# Route where style transfer results will be displayed
@app.route('/model', methods=['GET', 'POST'])
def style_transfer():
    if request.method=='POST':
        return render_template('model.html', output_image='output.png')