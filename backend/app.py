# app.py

#connects to mongo

#user login mechanisms

#loads users model 

#runs ASL predictor








from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId
print("About to import ASLPredictor.")
from utils.asl_model import ASLPredictor
print("ASLPredictor imported.")

import bcrypt
import base64
import traceback 
import cv2
import numpy as np
import os 

load_dotenv() # loads mongodb

app = Flask(__name__)
CORS(app)

#read mongodb connection settings
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME")

#connect to mongoDB
client = MongoClient(mongo_uri)
db = client[db_name]

# cache so model doesnt need to be reloaded every time
predictor_cache = {}

@app.route("/")
def home ():
    return "Website is running" # backend is running 

# check for mongodb connection 
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

    email = data.get("email") #get email data form db
    password = data.get("password") #get passowrd data from db

    user = db.users.find_one({"email": email}) #look up user by email 

    if not user:
        return jsonify({"success": False, "message": "User not found"}), 404
    
    stored_password = user["password"]

    if not bcrypt.checkpw(password.encode("utf-8"), stored_password.encode("utf-8")): #passowrd in encrypted in mongodb
        return jsonify ({"success": False, "message": "Invalid password"}), 401
    
    active_model = db.user_models.find_one({ #gets users asl model 
        "user_id": user["_id"],
        "is_active": True
    })

    model_info = None #ASL model info 
    if active_model:
        model_info = {
            "model_name": active_model["model_name"],
            "model_path": active_model["model_path"],
            "dataset_path": active_model["dataset_path"]
        }
    
    return jsonify({ #log in success 
        "success": True,
        "message": "Logged in",
        "user": {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
        },
        "active_model": model_info
    })

# get ASL model for the specific user 
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

# get a base64 image from frontend 
#load ASL model and return a prediciton
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    image_data = data.get("image") #need image data
    user_id = data.get("user_id")# need the correct user 

    if not image_data: 
        return jsonify({
            "success": False,
            "message": "No image received"
        }), 400
    
    if not user_id:
        return jsonify({
            "success": False, 
            "message": "No user ID received"
        }), 400 

    try:
        user = db.users.find_one({"_id": ObjectId(user_id)}) #user exists? 

        if not user:
            return jsonify({
                "success": False,
                "message": "User not found"
            }), 404

        active_model = db.user_models.find_one({ # find the users model
            "user_id": user["_id"],
            "is_active": True
        })

        if not active_model:
            return jsonify({
                "success": False,
                "message": "No active model found for this user"
            }), 404

        model_path = active_model["model_path"]
        full_model_path = os.path.abspath(model_path) #the model path 

        if not os.path.exists(full_model_path): # make sure mode exists in the disk 
            return jsonify ({
                "success": False,
                "message": f"Model file not found at {full_model_path}"
            }), 404

        header, encoded = image_data.split(",", 1) # split data URL into meta data and base64 image content
        image_bytes = base64.b64decode(encoded) # turn base64 into computer langauge
        image_array = np.frombuffer(image_bytes, dtype=np.uint8) #convert computer language into a numpy array so opencv can read it 
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR) # decode image in a open cv frame 

        if frame is None:
            return jsonify({
                "success": False,
                "message": "Could not decode image"
            }), 400

        if full_model_path not in predictor_cache: # if model hasnt run before, start cache
            predictor_cache[full_model_path] = ASLPredictor(full_model_path)

        predictor = predictor_cache[full_model_path]
        result = predictor.predict_from_frame(frame) #run asl prediciton on image frame 

        return jsonify({ #predicion redults to frontend 
            "success": True,
            "message": "Prediction Successful",
            "prediction": result["label"],
            "confidence": round(result["confidence"] * 100, 2),
            "hand_detected": result["hand_detected"]        
        })

    except Exception as error:
        print("FULL PREDICTION TRACEBACK:") #debugging
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Prediction error: {str(error)}"
        }), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)