# Function to check if uploaded file has an acceptable extention
def is_file_allowed(filename):
    if not "." in filename:
        return False
    
    suffix = filename.rsplit('.', 1)[1]

    if suffix.lower() in ['jpeg', 'jpg', 'png']:
        return True
    else:
        return False