/// App.jsx

/// frontend shows everythin

///send image dtrings to backend for prediction



///

import { useEffect, useRef, useState } from "react";
import "./index.css";
import logo from "./assets/aslife-logo.jpg";

function App() {
  // login 
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  // messages 
  const [message, setMessage] = useState("");
  // user data
  const [user, setUser] = useState(null);
// model info 
  const [activeModel, setActiveModel] = useState(null);
  const [modelMessage, setModelMessage] = useState("");
  const [currentPage, setCurrentPage] = useState("login"); //pages 
  const [capturedImage, setCapturedImage] = useState(null); //image frames 
  const [backendCropImage, setBackendCropImage] = useState(null);
  const [predictionResult, setPredictionResult] = useState("");// prediciton
  const [isPredicting, setIsPredicting] = useState(false);
  const [isSendingFrame, setIsSendingFrame] = useState(false); // send frame to backend
  const [zoomLevel, setZoomLevel] = useState(1); //zooom
  const [boldText, setBoldText] = useState(false); //bold text
  const [highContrast, setHighContrast] = useState(false); // contrast
  const videoRef = useRef(null); //video elements
  const canvasRef = useRef(null);
  const predictionIntervalRef = useRef(null); //prediciton interval
  const isSendingFrameRef = useRef(false);
  

  const handleLogin = async (event) => {
    event.preventDefault();

    try { // login request
      const response = await fetch("http://127.0.0.1:5000/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email,
          password,
        }),
      });

      const data = await response.json();
      

      if (data.success) { // bring in the users info and model 
        setUser(data.user);
        setActiveModel(data.active_model);
        setMessage(`Welcome, ${data.user.name}!`); //welcome message 
        setCurrentPage("dashboard");
      } else {
        setMessage(data.message);
      }
    } catch (error) {
      console.error("Login error:", error); // unsuccessful login 
      setMessage("Could not connect to the server.");
    }
  };

  const handleLogout = () => { // logging out a user 
    if (predictionIntervalRef.current) {
      clearInterval(predictionIntervalRef.current);
      predictionIntervalRef.current = null;
    }
    setIsPredicting(false); //turn off webcam and clear everythin else when logged out 
    isSendingFrameRef.current = false;
    stopWebcam();
    setUser(null);
    setActiveModel(null);
    setEmail("");
    setPassword("");
    setMessage("");
    setModelMessage("");
    setCapturedImage(null);
    setBackendCropImage(null);
    setPredictionResult("");
    setCurrentPage("login");
  };

  const increaseZoom = () => { //zoom function
    setZoomLevel((prev) => Math.min(prev + 0.1, 1.4));
  };

  const decreaseZoom = () => { //zoom funciton
    setZoomLevel((prev) => Math.max(prev - 0.1, 0.9));
  };

  const resetZoom = () => {//reset zoom
    setZoomLevel(1);
  };

  const renderAccessibilityControls = () => ( // acessiblity buttons //
    <div className="accessibility-toolbar">
      <p className="accessibility-title">Accessibility Options</p>
      
      <div className="accessibility-btn-row">
        <button type="button" className="accessibility-btn" onClick={decreaseZoom}> 
          A- 
        </button>
        <button type="button" className="accessibility-btn" onClick={resetZoom}>
          A
        </button>
        <button type="button" className="accessibility-btn" onClick={increaseZoom}>
          A+
        </button>
      </div>

      <div className="accessibility-toggle-row">
        <button
          type="button"
          className={`accessibility-btn ${boldText ? "active-accessibility-btn" : ""}`}
          onClick={() => setBoldText((prev) => !prev)}
        >
          Bold Text 
        </button>

        <button
          type="button"
          className={`accessibility-btn ${highContrast ? "active-accessibility-btn" : ""}`}
          onClick={() => setHighContrast((prev) => !prev)}
        >
          High Contrast
        </button>
      </div>
    </div>
  );

  const handleStartCamera = () => {
    setCurrentPage("camera"); // starting camera
  };

  const handleBackToDashboard = () => { // stop camera when going to dash
    stopWebcam();
    if (predictionIntervalRef.current) {
      clearInterval(predictionIntervalRef.current);
      predictionIntervalRef.current = null;
    }
    setIsPredicting(false);
    isSendingFrameRef.current = false;
    setCapturedImage(null);
    setBackendCropImage(null);
    setPredictionResult("");
    setCurrentPage("dashboard");
  };

  const handleMyModel = async () => { // get the users model indo 
    try {
      const response = await fetch(`http://127.0.0.1:5000/my-model/${user.id}`);
      const data = await response.json();

      if (data.success) {
        setActiveModel(data.active_model);
        setModelMessage("Model information loaded successfully."); // if it finds, paste this 
      } else {
        setModelMessage(data.message);
      }
    } catch (error) {
      console.error("Model fetch error:", error); // if it cant find, paste this 
      setModelMessage("Could not load model information.");
    }
  };

  const startWebcam = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ // asks user for webcam access 
      video: true,
      audio: false,
    });

    if (videoRef.current) { // webcam stream
      videoRef.current.srcObject = stream;
    }
  } catch (error) {
    console.error("Webcam error:", error);
    setMessage("Unable to access the camera.");
  }
};

const sendCurrentFrameForPrediction = async () => { // sending frames to backend 
  if (!videoRef.current || !canvasRef.current || !user || isSendingFrameRef.current) {
    return;
  }

  isSendingFrameRef.current = true;
  setIsSendingFrame(true);

  try {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    if (!video.videoWidth || !video.videoHeight) { // get real frame dimesions 
      return;
    }
      // change canvas size to video frame size
    canvas.width = video.videoWidth; 
    canvas.height = video.videoHeight;

    context.drawImage(video, 0, 0, canvas.width, canvas.height); //draw image on hidde canvas 

    const imageDataUrl = canvas.toDataURL("image/jpeg"); //convert canvas into base64 image
    setCapturedImage(imageDataUrl);

    const response = await fetch("http://127.0.0.1:5000/predict", { //send base64 image to backend to predict 
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image: imageDataUrl,
        user_id: user.id,
      }),
    });

    const data = await response.json();
    console.log("Predict response:", data);

    if (data.success) {
      setPredictionResult( // show predicted sign and confidence 
        `Recognized Gesture: ${data.prediction} | Confidence: ${data.confidence}%`
      );
      setBackendCropImage(data.crop_preview || null);
    } else {
      setPredictionResult(data.message);
      setBackendCropImage(null);
    }
  } catch (error) {
    console.error("Prediction request error:", error);
    setPredictionResult("Could not send image to the server.");
    setBackendCropImage(null);
  } finally {
    isSendingFrameRef.current = false; //finish sending frame so it. can send another 
    setIsSendingFrame(false);
  }
};

const handleCaptureFrame = async () => { //send frame to backend 
  await sendCurrentFrameForPrediction();
};

const stopWebcam = () => { //stopping the webcam 
  if (videoRef.current && videoRef.current.srcObject) {
    const stream = videoRef.current.srcObject;
    const tracks = stream.getTracks();
    tracks.forEach((track) => track.stop());
    videoRef.current.srcObject = null;
  }
};

useEffect(() => {
  if (user && currentPage === "camera") { //start webcam
    startWebcam();
  }

  return () => {
    stopWebcam(); //stopping webcam
    if (predictionIntervalRef.current) {
      clearInterval(predictionIntervalRef.current);
      predictionIntervalRef.current = null;
    }
    isSendingFrameRef.current = false;
  };
}, [user, currentPage]);

  const startLivePrediction = () => { //start prediciton timer 
    if (predictionIntervalRef.current) return; //prevent multiple timers 

    setIsPredicting(true);
    sendCurrentFrameForPrediction(); //run prediciton

    predictionIntervalRef.current = setInterval(() => { //sends frame every 1.5 sec
      sendCurrentFrameForPrediction();
    }, 1500);
  };

  const stopLivePrediction = () => {// stop prediciton timer when prediction stops 
    if (predictionIntervalRef.current) {
      clearInterval(predictionIntervalRef.current);
      predictionIntervalRef.current = null;
    }
    setIsPredicting(false);
  };

  const shellClassName = `app-shell ${boldText ? "bold-mode" : ""} ${ // border for accessilibilty options
    highContrast ? "high-contrast-mode" : ""
  }`;

  if (user && currentPage === "camera") { //camera page , added logo 
    return (
      <div className={shellClassName} style={{ zoom: zoomLevel }}>
        <div className="camera-page-card">
          <div className="logo-wrap">
            <img src={logo} alt="ASLife logo" className="logo-image-small" /> 
          </div>
          
          {renderAccessibilityControls()} 
          <p className="subtitle">ASL Camera</p>

          <div className="instruction-box">
            Place your hand in front of the camera and hold your sign steady. 
            <br />
            <strong>Tip:</strong> For best results, use a plain, lighter background with good lighting and keep your full hand visible.
          </div>

          <div className="camera-placeholder">
            <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="camera-video"
           /> 
          </div>

          <canvas ref={canvasRef} style={{display: "none" }} />

          <div className="camera-controls-panel">
            <div className="camera-button-stack">
              <button className="dashboard-btn" onClick={handleCaptureFrame}>
                Capture Frame
              </button>

              {!isPredicting ? (
                <button className="dashboard-btn" onClick={startLivePrediction}>
                  Start Live Prediction
                </button>
              ) : (
                <button className="dashboard-btn logout-btn" onClick={stopLivePrediction}>
                  Stop Live Prediction
                </button>
              )}

              <button className="dashboard-btn" onClick={handleBackToDashboard}>
                Back to Dashboard
              </button>
            </div>
          </div>

          <div className="prediction-panel">
            <p className="prediction-panel-title">Recognition Status</p> 
            {isPredicting ? (
              <p className="live-status-text">Live prediction is running...</p>
            ) : (
              <p className="live-status-text">Live prediction is off.</p>
            )}

            {predictionResult ? (
              <p className="prediction-result-text">{predictionResult}</p>
            ) : (
              <p className="prediction-result-text">No prediction yet.</p>
            )}
          </div>

          {backendCropImage && (
                <div className="captured-preview-card">
                  <p className="model-info-title">Backend Crop Used for Prediction</p>
                  <img
                    src={backendCropImage}
                    alt="Backend crop used for prediction"
                    className="captured-preview-image"
                  />
                </div>
              )}
    </div>
  </div> 
  );
}
   

  if (user && currentPage === "dashboard") {
    return (
      <div className={shellClassName} style={{ zoom: zoomLevel }}>
        <div className="login-card dashboard-card">
          <div className="logo-wrap">
            <img src={logo} alt="ASLife logo" className="logo-image" />
          </div>
          {renderAccessibilityControls()}
<p className="dashboard-welcome">Welcome back, {user.name}.</p>

<p className="dashboard-subtext">
  Your personalized ASL assistant is ready.
</p>

<div className="instruction-box">
  Choose an option below to continue.
</div>

<div className="dashboard-grid">
  <button className="dashboard-btn" onClick={handleStartCamera}>
    Start Camera
  </button>

  <button className="dashboard-btn">
    My Profile
  </button>

  <button className="dashboard-btn" onClick={handleMyModel}>
    My Model
  </button>

  <button className="dashboard-btn logout-btn" onClick={handleLogout}>
    Log Out
  </button>
</div>

{modelMessage && <p className="status-message">{modelMessage}</p>}

{activeModel && (
  <div className="model-info-card">
    <p className="model-info-title">Current Model Information</p>
    <div className="model-info-row">
      <span className="model-label">Model Name:</span>
      <span>{activeModel.model_name}</span>
    </div>
    <div className="model-info-row">
      <span className="model-label">Dataset:</span>
      <span>{activeModel.dataset_path}</span>
    </div>
    <div className="model-info-row">
      <span className="model-label">Model File:</span>
      <span>{activeModel.model_path}</span>
    </div>
  </div>
)}

          
        </div>
      </div>
    );
  }

  return (
    <div className={shellClassName} style={{ zoom: zoomLevel }}>
      <div className="login-card">
        <div className="logo-wrap">
          <img src={logo} alt="ASLife logo" className="logo-image" />
        </div>

        {renderAccessibilityControls()}
        <p className="subtitle">Sign in to start communicating.</p>

        <div className="instruction-box">
          Please enter your email and password to continue.
        </div>

        <form className="login-form" onSubmit={handleLogin}>
          <label htmlFor="email" className="form-label">
            Email Address
          </label>
          <input
            id="email"
            type="email"
            placeholder="Enter your email"
            className="form-input"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />

          <label htmlFor="password" className="form-label">
            Password
          </label>
          <input
            id="password"
            type="password"
            placeholder="Enter your password"
            className="form-input"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          <button type="submit" className="primary-btn">
            Log In
          </button>
        </form>

        {message && <p className="status-message">{message}</p>}

        <div className="helper-box">
          <p>Need help?</p>
          <p>Please ask a caregiver, instructor, or assistant for support.</p>
        </div>
      </div>
    </div>
  );
}

export default App;