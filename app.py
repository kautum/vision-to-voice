from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import io
import base64
from werkzeug.utils import secure_filename
import os
from main import process_image  # assuming your image/audio processing logic is in main.py

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image = Image.open(filepath)

        elif 'webcam_image' in request.form and request.form['webcam_image'] != '':
            img_data = request.form['webcam_image'].split(',')[1]
            img_bytes = base64.b64decode(img_data)
            image = Image.open(io.BytesIO(img_bytes))
            filename = 'webcam_capture.png'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)

        else:
            return redirect(request.url)

        caption, translated_caption, audio_en, audio_translated = process_image(image)

        return render_template('result.html',
                               image_url=url_for('static', filename=f"uploads/{filename}"),
                               caption=caption,
                               translated_caption=translated_caption,
                               audio_en=url_for('static', filename=f"audio/{os.path.basename(audio_en)}"),
                               audio_translated=url_for('static', filename=f"audio/{os.path.basename(audio_translated)}"))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
