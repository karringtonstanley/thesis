import { useEffect, useRef, useState } from "react";
import "./index.css";
import logo from "./assets/aslife-logo.jpg";

function App() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [user, setUser] = useState(null);
  const [activeModel, setActiveModel] = useState(null);
  const [modelMessage, setModelMessage] = useState("");
  const [currentPage, setCurrentPage] = useState("login");
  const [capturedImage, setCapturedImage] = useState(null);
  const [predictionResult, setPredictionResult] = useState("");
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  

  const handleLogin = async (event) => {
    event.preventDefault();

    try {
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
      

      if (data.success) {
        setUser(data.user);
        setActiveModel(data.active_model);
        setMessage(`Welcome, ${data.user.name}!`);
        setCurrentPage("dashboard");
      } else {
        setMessage(data.message);
      }
    } catch (error) {
      console.error("Login error:", error);
      setMessage("Could not connect to the server.");
    }
  };

  const handleLogout = () => {
    setUser(null);
    setActiveModel(null);
    setEmail("");
    setPassword("");
    setMessage("");
    setModelMessage("");
    setCurrentPage("login");
  };

  const handleStartCamera = () => {
    setCurrentPage("camera");
  };

  const handleBackToDashboard = () => {
    stopWebcam();
    setCurrentPage("dashboard");
  };

  const handleMyModel = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:5000/my-model/${user.id}`);
      const data = await response.json();

      if (data.success) {
        setActiveModel(data.active_model);
        setModelMessage("Model information loaded successfully.");
      } else {
        setModelMessage(data.message);
      }
    } catch (error) {
      console.error("Model fetch error:", error);
      setModelMessage("Could not load model information.");
    }
  };

  const startWebcam = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
    }
  } catch (error) {
    console.error("Webcam error:", error);
    setMessage("Unable to access the camera.");
  }
};

const handleCaptureFrame = async () => {
  if (!videoRef.current || !canvasRef.current) return;

  const video = videoRef.current;
  const canvas = canvasRef.current;
  const context = canvas.getContext("2d");

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const imageDataUrl = canvas.toDataURL("image/jpeg");
  setCapturedImage(imageDataUrl);

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
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
      setPredictionResult(
        `Model check complete. Exists: ${data.model_exists ? "Yes" : "No"}` 
      );
    } else {
      setPredictionResult(data.message);
    }
  } catch (error) {
    console.error("Prediction request error:", error);
    setPredictionResult("Could not send image to the server.");
  }
};

const stopWebcam = () => {
  if (videoRef.current && videoRef.current.srcObject) {
    const stream = videoRef.current.srcObject;
    const tracks = stream.getTracks();
    tracks.forEach((track) => track.stop());
    videoRef.current.srcObject = null;
  }
};

useEffect(() => {
  if (user && currentPage === "camera") {
    startWebcam();
  }

  return () => {
    stopWebcam();
  };
}, [user, currentPage]);

  if (user && currentPage === "camera") {
    return (
      <div className="app-shell">
        <div className="camera-page-card">
          <div className="logo-wrap">
            <img src={logo} alt="ASLife logo" className="logo-image-small" />
          </div>

          <p className="subtitle">ASL Camera</p>

          <div className="instruction-box">
            Place your hand in front of the camera and hold your sign steady.
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

          <div className="dashboard-grid">
            <button className="dashboard-btn" onClick={handleCaptureFrame}>
              Capture Frame
            </button>

            <button className="dashboard-btn" onClick={handleBackToDashboard}>
              Back to Dashboard
            </button>
          </div>
          {predictionResult && (
            <p className="status-message">{predictionResult}</p>
          )}

          {capturedImage && (
            <div className="captured-preview-card">
              <p className="model-info-title">Captured Preview</p>
              <img
                src={capturedImage}
                alt="Captured ASL frame"
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
      <div className="app-shell">
        <div className="login-card dashboard-card">
          <div className="logo-wrap">
            <img src={logo} alt="ASLife logo" className="logo-image" />
          </div>
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
    <div className="app-shell">
      <div className="login-card">
        <div className="logo-wrap">
          <img src={logo} alt="ASLife logo" className="logo-image" />
        </div>

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