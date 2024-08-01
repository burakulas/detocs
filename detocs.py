"""
Author: Burak Ulas -  github.com/burakulas
2024, Konkoly Observatory, COMU
"""
from flask import Flask, render_template, request, Response, make_response, send_file, jsonify
from utils.utils import get_observation_ids, detect_objects, plot_png, get_fits_path, print_png_elements, above_conf_detection, period_from_tebs, input_extention, input_f, response_f, downl_f, downim_f, remove_tuples_with_common_last_elements

app = Flask(__name__)
#app.secret_key = 'tess_od'

@app.route('/')
def home():
    return render_template('index.html')

# Render results
@app.route('/results', methods=['GET','POST'])
def results():
    result_data = input_f()
    return render_template('index.html', result_data=result_data)

# Download images and annotations
@app.route('/index.html')
def index():
    return response_f()

@app.route('/downl', methods=['POST'])
def downl():
    return downl_f()

@app.route('/download_image', methods=['GET'])
def download_image():
    return downim_f()

if __name__ == '__main__':
    app.run(debug=True)
    