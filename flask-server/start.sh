set -e

export PATH="/opt/render/project/src/flask-server/.venv/bin:$PATH"

echo "Running database table creation (db.create_all())..."

echo "Attempting database table creation (db.create_all())..."
python -c "
import sys
from app import app, db # Ensure 'app' and 'db' are imported from your app.py
with app.app_context():
    try:
        db.create_all()
        print('Database tables checked/created successfully via start.sh!', file=sys.stderr)
    except Exception as e:
        print(f'ERROR during db.create_all(): {e}', file=sys.stderr)
        # Propagate error, causing deployment to fail if db creation fails
        sys.exit(1)


# Start Gunicorn server after database setup is complete
echo "Starting Gunicorn server..."

exec gunicorn app:app --bind 0.0.0.0:$PORT