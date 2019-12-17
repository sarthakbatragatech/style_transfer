from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method=='GET':
        return render_template('index.html')
    
    if request.method=='POST':
        content_image = request.files['content']
        style_image = request.files['style']
        return render_template(
            'result.html', 
            content=content_image.filename,
            style=style_image.filename)
