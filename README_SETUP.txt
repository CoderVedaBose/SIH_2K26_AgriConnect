AGRICONNECT AUTH UPDATE

Replace your frontend files with index.html, script.js, style.css, auth.css, auth.js and the 3 auth HTML pages.
Replace main.py with the included main.py. Keep your backend folder with the model files if your project uses backend/ paths.

IMPORTANT: The included main.py expects:
backend/database.json
backend/demand_preprocessor.pkl
backend/demand_xgb_model.json
backend/Crop-Demand-Data.csv

Run:
python -m uvicorn main:app --reload --port 8000

Open index.html with Live Server.

Auth pages:
farmer-register.html
farmer-login.html
buyer-login.html

Note: This is a demo authentication system. Passwords are hashed with SHA-256, but a production system should use a password-hashing library such as bcrypt/argon2 and real session/JWT authentication.
