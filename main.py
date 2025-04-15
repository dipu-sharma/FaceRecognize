import asyncio
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image, ImageDraw

# Constants
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"
TRAINING_DIR = Path("training")
OUTPUT_DIR = Path("output")
VALIDATION_DIR = Path("validation")

# Create directories if they don't exist
for directory in [TRAINING_DIR, OUTPUT_DIR, VALIDATION_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

app = FastAPI()

class CameraManager:
    def __init__(self):
        self.active_cameras: Dict[str, cv2.VideoCapture] = {}
        self.known_encodings = self.load_encodings()

    def load_encodings(self) -> Dict:
        """Load face encodings from a pickle file."""
        try:
            with DEFAULT_ENCODINGS_PATH.open(mode="rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {"names": [], "encodings": []}
        except Exception as e:
            print(f"Error loading encodings: {e}")
            return {"names": [], "encodings": []}

    def add_camera(self, camera_id: str, rtsp_url: str) -> bool:
        """Add a new IP camera to the manager."""
        if camera_id in self.active_cameras:
            return False

        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return False

        self.active_cameras[camera_id] = cap
        return True

    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera from the manager."""
        if camera_id not in self.active_cameras:
            return False

        self.active_cameras[camera_id].release()
        del self.active_cameras[camera_id]
        return True

    def process_frame(self, frame: np.ndarray, model: str = "hog") -> bytes:
        """Process a single frame for face recognition."""
        # Convert from BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = frame[:, :, ::-1]
        
        # Find all faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame, model=model)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Convert to PIL Image for drawing
        pil_image = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_image)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = self._recognize_face(face_encoding) or "Unknown"
            self._display_face(draw, (top, right, bottom, left), name)

        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _recognize_face(self, unknown_encoding: List[float]) -> Optional[str]:
        """Match an unknown face encoding against known encodings."""
        if not self.known_encodings["encodings"]:
            return None

        boolean_matches = face_recognition.compare_faces(
            self.known_encodings["encodings"], 
            unknown_encoding,
            tolerance=0.6
        )
        votes = Counter(
            name
            for match, name in zip(boolean_matches, self.known_encodings["names"])
            if match
        )
        return votes.most_common(1)[0][0] if votes else None

    def _display_face(self, draw: ImageDraw.ImageDraw, bounding_box: Tuple, name: str) -> None:
        """Draw face bounding box and name label."""
        top, right, bottom, left = bounding_box
        # Draw bounding box
        draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR, width=2)
        
        # Draw label background
        text_left, text_top, text_right, text_bottom = draw.textbbox(
            (left, bottom), name
        )
        draw.rectangle(
            ((text_left, text_top), (text_right, text_bottom)),
            fill=BOUNDING_BOX_COLOR,
            outline=BOUNDING_BOX_COLOR,
        )
        
        # Draw text
        draw.text(
            (text_left, text_top),
            name,
            fill=TEXT_COLOR,
        )

    async def generate_frames(self, camera_id: str):
        """Generate frames from a camera with face recognition."""
        if camera_id not in self.active_cameras:
            raise HTTPException(status_code=404, detail="Camera not found")

        cap = self.active_cameras[camera_id]
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            await asyncio.sleep(0.033)  # ~30fps

camera_manager = CameraManager()

@app.on_event("shutdown")
async def shutdown_event():
    """Release all camera resources on shutdown."""
    for camera_id in list(camera_manager.active_cameras.keys()):
        camera_manager.remove_camera(camera_id)

@app.post("/cameras/{camera_id}")
async def add_camera(camera_id: str, rtsp_url: str):
    """Add a new IP camera to the system."""
    if camera_manager.add_camera(camera_id, rtsp_url):
        return {"message": f"Camera {camera_id} added successfully"}
    raise HTTPException(status_code=400, detail="Failed to add camera")

@app.delete("/cameras/{camera_id}")
async def remove_camera(camera_id: str):
    """Remove a camera from the system."""
    if camera_manager.remove_camera(camera_id):
        return {"message": f"Camera {camera_id} removed successfully"}
    raise HTTPException(status_code=404, detail="Camera not found")

@app.get("/cameras/{camera_id}/stream")
async def video_feed(camera_id: str):
    """Stream video feed with face recognition."""
    return StreamingResponse(
        camera_manager.generate_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/train")
async def train_model(model: str = "hog"):
    """Train the face recognition model."""
    names = []
    encodings = []

    for filepath in TRAINING_DIR.glob("*/*"):
        if not filepath.is_file():
            continue

        name = filepath.parent.name
        try:
            image = face_recognition.load_image_file(filepath)
            face_locations = face_recognition.face_locations(image, model=model)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for encoding in face_encodings:
                names.append(name)
                encodings.append(encoding)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    name_encodings = {"names": names, "encodings": encodings}
    try:
        with DEFAULT_ENCODINGS_PATH.open(mode="wb") as f:
            pickle.dump(name_encodings, f)
        
        # Reload encodings in the manager
        camera_manager.known_encodings = camera_manager.load_encodings()
        return {"message": f"Model trained with {len(names)} faces"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

# WebSocket endpoint for real-time communication
@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    try:
        if camera_id not in camera_manager.active_cameras:
            await websocket.close(code=1008, reason="Camera not found")
            return

        cap = camera_manager.active_cameras[camera_id]
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = camera_manager.process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.033)  # ~30fps
    except WebSocketDisconnect:
        print(f"Client disconnected from camera {camera_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# HTML page for testing
@app.get("/")
async def index():
    html_content = """
    <html>
        <head>
            <title>Face Recognition</title>
        </head>
        <body>
            <h1>IP Camera Streams</h1>
            <div>
                <h2>Camera 1</h2>
                <img src="/cameras/cam1/stream" width="640" height="480">
            </div>
            <div>
                <h2>Camera 2</h2>
                <img src="/cameras/cam2/stream" width="640" height="480">
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)