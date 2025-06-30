from flask import Flask, request, jsonify, send_from_directory, session, make_response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
import numpy as np
import random
from datetime import datetime, timedelta
import traceback
import sys

from image_processing import apply_contour_manipulation, apply_object_addition

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'supersecretkey')
print(f"DEBUG: Flask SECRET_KEY loaded: {app.config['SECRET_KEY'][:10]}... (showing first 10 chars)")
if not app.config['SECRET_KEY'] or len(app.config['SECRET_KEY']) < 32:
    print("WARNING: SECRET_KEY is short or not set. Sessions may not work correctly!", file=sys.stderr)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://user:password@localhost:5432/spot_the_diff_db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_DOMAIN'] = None

ALLOWED_ORIGINS = os.environ.get(
    'CORS_ORIGINS',
    'https://localhost:5173,https://127.0.0.1:5173' # Local development origins
).split(',')

print(f"DEBUG: Flask-CORS configured with origins: {ALLOWED_ORIGINS}")

CORS(app, supports_credentials=True, origins=ALLOWED_ORIGINS)
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_app = app
login_manager.login_view = 'login'
login_manager.login_message = "Please log in to access this page."

#User class: better for authentication
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    #game statistics
    games_played = db.Column(db.Integer, default=0)
    games_won = db.Column(db.Integer, default=0)
    total_differences_found = db.Column(db.Integer, default=0) 

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'
    

@login_manager.unauthorized_handler
def unauthorized():
    is_ajax_request = request.accept_mimetypes.accept_json or 'application/json' in request.headers.get('Content-Type', '')

    if is_ajax_request:
        print("DEBUG: Unauthorized AJAX request detected. Returning 401.", file=sys.stderr)
        return jsonify({'error': 'Unauthorized: Please log in to access this resource.'}), 401
    print("DEBUG: Unauthorized non-AJAX request detected. Redirecting to login.", file=sys.stderr)
    return app.redirect(login_manager.login_view)

#user loader for flask-login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

UPLOAD_FOLDER = 'uploads' # Directory to save uploaded and processed images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload folder: {UPLOAD_FOLDER}")


objects_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'objects')
if not os.path.exists(objects_path):
    print(f"Warning: 'objects' folder not found at '{objects_path}'. Object addition will not work.")



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#Authentication routes
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400
    
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 409
    
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409
    
    new_user = User(username=username, email=email)
    new_user.set_password(password)
    
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'message': 'Registration successful! Please log in.'}), 201
    except Exception as e:
        db.session.rollback()
        print(f"Database error during registration: {e}")
        return jsonify({'error': 'An error occurred during registration.'}), 500
    

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        login_user(user) # log in the user
        session.permanent = True # make session permanent

        print(f"DEBUG: Login successful for user: {user.username}", file=sys.stderr)
        print(f"DEBUG: Session object before response: {session}", file=sys.stderr)
        print(f"DEBUG: Session keys before response: {list(session.keys())}", file=sys.stderr)
        if '_user_id' in session:
            print(f"DEBUG: _user_id in session: {session['_user_id']}", file=sys.stderr)
        else:
            print("DEBUG: _user_id NOT in session after login_user!", file=sys.stderr)


        response_data = {
            'message': 'Login successful!',
            'username': user.username,
            'games_played': user.games_played,
            'games_won': user.games_won,
            'total_differences_found': user.total_differences_found
        }

        resp = make_response(jsonify(response_data), 200)
        session_cookie_value = app.session_interface.get_signing_serializer(app).dumps(dict(session))

        resp.set_cookie(
            app.config['SESSION_COOKIE_NAME'],
            session_cookie_value,
            expires=datetime.now() + app.permanent_session_lifetime,
            httponly=True,
            secure=app.config['SESSION_COOKIE_SECURE'],
            samesite=app.config['SESSION_COOKIE_SAMESITE'],
            path=app.config.get('SESSION_COOKIE_PATH', '/')
        )

        print(f"DEBUG: Manually setting Set-Cookie header. Value starts with: {session_cookie_value[:20]}...", file=sys.stderr)
        print(f"DEBUG: Set-Cookie header on response object: {resp.headers.get('Set-Cookie')}", file=sys.stderr)
        print(f"DEBUG: Session cookie attributes: Secure={app.config['SESSION_COOKIE_SECURE']}, SameSite={app.config['SESSION_COOKIE_SAMESITE']}, Domain={app.config.get('SESSION_COOKIE_DOMAIN')}, Path={app.config.get('SESSION_COOKIE_PATH', '/')}", file=sys.stderr)

        return resp



        # return jsonify({
        #     'message': 'Login successful!',
        #     'username': user.username,
        #     'games_played': user.games_played,
        #     'games_won': user.games_won,
        #     'total_differences_found': user.total_differences_found
        # }), 200
    else:
        print("DEBUG: Login failed: Invalid credentials.", file=sys.stderr)
        return jsonify({'error': 'Invalid username or password'}), 401
    
@app.route('/logout', methods=['POST'])
@login_required # Requires user to be logged in to logout
def logout():
    print(f"DEBUG: User {current_user.username} logging out.", file=sys.stderr)
    logout_user()
    return jsonify({'message': 'Logged out successfully!'}), 200

@app.route('/user_profile', methods=['GET'])
@login_required # protect this route
def user_profile():
    print(f"DEBUG: Accessing user_profile for {current_user.username}.", file=sys.stderr)
    return jsonify({
        'username': current_user.username,
        'games_played': current_user.games_played,
        'games_won': current_user.games_won,
        'total_differences_found': current_user.total_differences_found
    }), 200

@app.route('/update_stats', methods=['POST'])
@login_required # Only logged-in users can update stats
def update_stats():
    print(f"DEBUG: Updating stats for user {current_user.username}.", file=sys.stderr)
    data = request.get_json()
    differences_found = data.get('differencesFound', 0)
    game_won = data.get('gameWon', False)

    try:
        current_user.games_played += 1
        current_user.total_differences_found += differences_found
        if game_won:
            current_user.games_won += 1
        
        db.session.commit()
        print("DEBUG: Stats updated successfully.", file=sys.stderr)
        return jsonify({'message': 'Stats updated successfully!'}), 200
    except Exception as e:
        db.session.rollback()
        print(f"Database error during stats update: {e}")
        print(f"Database error during stats update: {e}", file=sys.stderr)
        return jsonify({'error': 'Failed to update stats.'}), 500

@app.route('/upload-and-process', methods=['POST'])
@login_required 
def upload_and_process():
    print(f"DEBUG: Attempting upload for user {current_user.username}. Is authenticated: {current_user.is_authenticated}", file=sys.stderr)
    print(f"DEBUG: Request headers: {request.headers}", file=sys.stderr)
    print(f"DEBUG: Incoming Cookie header: {request.headers.get('Cookie')}", file=sys.stderr)

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
            return jsonify({'error': f'Server processing failed: {e}'}), 500

    else:
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400

# Route to serve the uploaded/modified files
@app.route(f'/{UPLOAD_FOLDER}/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database tables checked/created.")

    if not os.path.exists(objects_path):
        os.makedirs(objects_path)

    app.run(debug=True, port=5000)