from flask import Flask, request, render_template
import os
import numpy as np
import cv2
from helpers import predict_image, process_and_save_image 

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('static', file.filename)
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            resized_image = cv2.resize(image, (256, 256))
            cv2.imwrite(file_path, resized_image)

            processed_image = process_and_save_image(file_path)
            processed_image_path = os.path.join('static', 'processed_' + file.filename)
            cv2.imwrite(processed_image_path, processed_image)

            predictions = predict_image(file_path)

            return render_template(
                'index.html', 
                prediction=predictions, 
                image_path=file_path, 
                processed_image_path=processed_image_path
            )
    return render_template('index.html', prediction=None, image_path=None, processed_image_path=None)


if __name__ == '__main__':
    app.run(debug=True)
