from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
import random
from datetime import datetime

from image_processing import apply_contour_manipulation, apply_object_addition

app = Flask(__name__)
CORS(app) 

UPLOAD_FOLDER = 'uploads' # Directory to save uploaded and processed images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload folder: {UPLOAD_FOLDER}")


objects_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'objects')
if not os.path.exists(objects_path):
    print(f"Warning: 'objects' folder not found at '{objects_path}'. Object addition will not work.")



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload-and-process', methods=['POST'])
def upload_and_process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            original_extension = file.filename.rsplit('.', 1)[1].lower()
            original_filename = f"original_{timestamp}.{original_extension}"
            modified_filename = f"modified_{timestamp}.{original_extension}" # Keep same extension for modified

            original_filepath = os.path.join(UPLOAD_FOLDER, original_filename)
            modified_filepath = os.path.join(UPLOAD_FOLDER, modified_filename)

            # Save original image
            file.save(original_filepath)
            print(f"Original image saved to: {original_filepath}")

            # Load image with OpenCV for processing
            original_img_array = cv2.imread(original_filepath)
            if original_img_array is None:
                return jsonify({'error': 'Could not read original image file (OpenCV failed to load)'}), 500

            modified_img_array = None
            differences = []

            # Randomly choose manipulation type
            manipulation_type = random.choice(['contour', 'add_object'])


            if manipulation_type == 'contour':
                print("Backend: Applying contour manipulation...")
                modified_img_array, differences = apply_contour_manipulation(original_img_array, num_of_changes=2)
                print(f"Backend: Contour differences generated: {len(differences)}")
            elif manipulation_type == 'add_object':
                print("Backend: Applying object addition manipulation...")
                modified_img_array, differences = apply_object_addition(original_img_array, num_objects=2, alpha=0.5, intended_width=30)
                print(f"Backend: Object addition differences generated: {len(differences)}")

            if modified_img_array is None:
                print("Backend: Image manipulation returned None, using original image.")
                modified_img_array = original_img_array.copy()
                differences = [] # No changes if manipulation failed

            # Save the modified image
            cv2.imwrite(modified_filepath, modified_img_array)
            print(f"Modified image saved to: {modified_filepath}")

            return jsonify({
                'originalImageUrl': f'/{UPLOAD_FOLDER}/{original_filename}',
                'modifiedImageUrl': f'/{UPLOAD_FOLDER}/{modified_filename}',
                'rawDifferencesForFrontendDemo': differences # Send the bounding box differences
            }), 200

        except Exception as e:
            print(f"Server error during processing: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            return jsonify({'error': f'Server processing failed: {e}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400

# Route to serve the uploaded/modified files
@app.route(f'/{UPLOAD_FOLDER}/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    if not os.path.exists(objects_path):
        print(f"Creating missing 'objects' directory at: {objects_path}")
        os.makedirs(objects_path)
    print(f"Backend running. Ensure '{UPLOAD_FOLDER}' and '{objects_path}' directories exist.")
    app.run(debug=True, port=5000)