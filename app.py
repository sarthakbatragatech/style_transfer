from pathlib import Path
from flask import Flask, render_template, request

app = Flask(__name__)
dir_path = Path('.')
upload_path = dir_path/'static'
app.config['UPLOAD_FOLDER'] = str(upload_path)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method=='GET':
        return render_template('index.html')
    
    if request.method=='POST':
        content_image = request.files['content']
        style_image = request.files['style']
        content_image.save(str(upload_path/content_image.filename))
        style_image.save(str(upload_path/style_image.filename))
        return render_template(
            'result.html', 
            content=content_image.filename,
            style=style_image.filename)
