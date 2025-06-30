set -e

export PATH="/opt/render/project/src/flask-server/.venv/bin:$PATH"

echo "Running database table creation (db.create_all())..."

flask --app app shell <<EOF
from app import db
with app.app_context():
    db.create_all()
    print('Database tables checked/created successfully via start.sh!')
EOF

# Start Gunicorn server after database setup is complete
echo "Starting Gunicorn server..."

exec gunicorn app:app --bind 0.0.0.0:$PORT