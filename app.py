from pathlib import Path
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

# This will give us the PosixPath of the working directory
dir_path = Path('.')
upload_path = str(dir_path/'static'/'uploads')

# If we land on the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Uploading Style and Content Images
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method=='POST':
        content_image = request.files['content']
        style_image = request.files['style']
        content_image.save(upload_path + '/' + content_image.filename)
        style_image.save(upload_path + '/' + style_image.filename)
        return render_template(
            'result.html', 
            content=content_image.filename,
            style=style_image.filename)

# Function to help display the uploaded files
@app.route('/upload/<filename>')
def serve_image(filename):
    return send_from_directory(
        directory='static/uploads',
        filename=filename)
