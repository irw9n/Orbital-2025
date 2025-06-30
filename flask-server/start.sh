set -e

export PATH="/opt/render/project/src/flask-server/.venv/bin:$PATH"

echo "Running database table creation (db.create_all())..."

echo "Attempting database table creation (db.create_all())..."
python -c "$(cat <<EOF_PYTHON
import sys
from app import app, db
with app.app_context():
    try:
        db.create_all()
        print('Database tables checked/created successfully via start.sh!', file=sys.stderr)
    except Exception as e:
        print(f'ERROR during db.create_all(): {e}', file=sys.stderr)
        sys.exit(1)
EOF_PYTHON
)"

# Start Gunicorn server after database setup is complete
echo "Starting Gunicorn server..."

exec gunicorn app:app --bind 0.0.0.0:$PORT