"""Flask web app that presents an upload page and returns an image digit prediction"""

# Importing required libs
from flask import Flask, render_template, request
from model import preprocess_img, predict_result
from werkzeug.exceptions import BadRequestKeyError
from model import preprocess_img, predict_result

# Instantiating flask app
app = Flask(__name__)


# Home route
@app.route("/")
def main():
    """Renders the Flask app main page"""
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """Sends a user request through the image processing and modelling functions"""
    try:
        img = preprocess_img(request.files['file'].stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))

    except (BadRequestKeyError, UnidentifiedImageError):
        print(type(e))
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
