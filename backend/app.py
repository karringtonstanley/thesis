from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId
import bcrypt
import base64
import cv2
import numpy as np
import os 

load_dotenv()

app = Flask(__name__)
CORS(app)

mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME")

client = MongoClient(mongo_uri)
db = client[db_name]

@app.route("/")
def home ():
    return "Website is running"

@app.route("/test-db")
def test_db():
    users = list(db.users.find({}, {"name": 1, "email": 1}))
    for user in users:
        user["_id"] = str(user["_id"])
    return jsonify({
        "message": "MongoDB connected",
        "users": users
    })

@app.route("/login", methods=["POST"])
def login ():
    data = request.get_json()

    email = data.get("email")
    password = data.get("password")

    user = db.users.find_one({"email": email})

    if not user:
        return jsonify({"success": False, "message": "User not found"}), 404
    
    stored_password = user["password"]

    if not bcrypt.checkpw(password.encode("utf-8"), stored_password.encode("utf-8")):
        return jsonify ({"success": False, "message": "Invalid password"}), 401
    
    active_model = db.user_models.find_one({
        "user_id": user["_id"],
        "is_active": True
    })

    model_info = None
    if active_model:
        model_info = {
            "model_name": active_model["model_name"],
            "model_path": active_model["model_path"],
            "dataset_path": active_model["dataset_path"]
        }
    
    return jsonify({
        "success": True,
        "message": "Logged in",
        "user": {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
        },
        "active_model": model_info
    })

@app.route("/my-model/<user_id>", methods=["GET"])
def get_my_model(user_id):
    active_model = db.user_models.find_one({
        "user_id": ObjectId(user_id),
        "is_active": True
    })

    if not active_model:
        return jsonify({
            "success": False,
            "message": "No active model found for this user"
        }), 404

    return jsonify({
        "success": True,
        "active_model": {
            "model_name": active_model["model_name"],
            "model_path": active_model["model_path"],
            "dataset_path": active_model["dataset_path"]
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    image_data = data.get("image")
    user_id = data.get("user_id")

    if not image_data:
        return jsonify({
            "success": False,
            "message": "No image received"
        }), 400

    try:
        user = db.users.find_one({"_id": ObjectId(user_id)})

        if not user:
            return jsonify({
                "success": False,
                "message": "User not found"
            }), 404

        active_model = db.user_models.find_one({
            "user_id": user["_id"],
            "is_active": True
        })

        if not active_model:
            return jsonify({
                "success": False,
                "message": "No active model found for this user"
            }), 404

        model_path = active_model["model_path"]
        full_model_path = os.path.abspath(model_path)

        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "success": False,
                "message": "Could not decode image"
            }), 400

        return jsonify({
            "success": True,
            "message": "Image decoded and model record found",
            "model_name": active_model["model_name"],
            "model_path": model_path,
            "full_model_path": full_model_path,
            "model_exists": os.path.exists(full_model_path)
        })

    except Exception as error:
        return jsonify({
            "success": False,
            "message": f"Prediction setup error: {str(error)}"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)