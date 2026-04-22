"""
FastAPI Backend for 3D Scene Reconstruction
Provides endpoints for image upload, reconstruction, and result serving.
"""
import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import (
    UPLOAD_DIR, RESULTS_DIR, CORS_ORIGINS,
    MAX_IMAGES, MIN_IMAGES, ALLOWED_EXTENSIONS, DEVICE
)
from reconstruction import ReconstructionPipeline

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App
app = FastAPI(
    title="3D Scene Reconstruction",
    description="Upload images and reconstruct 3D scenes using DUSt3R",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State
pipeline = ReconstructionPipeline(device=DEVICE)
sessions: Dict[str, dict] = {}
progress_connections: Dict[str, WebSocket] = {}


@app.on_event("startup")
async def startup():
    """Pre-load the model on startup."""
    logger.info("Starting up... loading model")
    pipeline.load_model()
    logger.info("Model ready")


@app.post("/api/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    """
    Upload multiple images for reconstruction.
    Returns a session_id used to track the reconstruction.
    """
    # Validate count
    if len(files) < MIN_IMAGES:
        raise HTTPException(400, f"At least {MIN_IMAGES} images required, got {len(files)}")
    if len(files) > MAX_IMAGES:
        raise HTTPException(400, f"At most {MAX_IMAGES} images allowed, got {len(files)}")

    # Create session
    session_id = str(uuid.uuid4())[:8]
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for f in files:
        # Validate extension
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            shutil.rmtree(session_dir, ignore_errors=True)
            raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

        # Save file
        file_path = session_dir / f.filename
        with open(file_path, "wb") as out:
            content = await f.read()
            out.write(content)
        saved_files.append(f.filename)

    sessions[session_id] = {
        "status": "uploaded",
        "n_images": len(saved_files),
        "files": saved_files,
        "image_dir": str(session_dir),
    }

    logger.info(f"Session {session_id}: uploaded {len(saved_files)} images")

    return {
        "session_id": session_id,
        "n_images": len(saved_files),
        "files": saved_files,
    }


@app.websocket("/ws/progress/{session_id}")
async def websocket_progress(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming reconstruction progress."""
    await websocket.accept()
    progress_connections[session_id] = websocket
    logger.info(f"WebSocket connected for session {session_id}")

    try:
        # Keep connection alive, wait for client messages or disconnect
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    finally:
        progress_connections.pop(session_id, None)


async def send_progress(session_id: str, stage: str, progress: int, message: str):
    """Send progress update via WebSocket if connected."""
    ws = progress_connections.get(session_id)
    if ws:
        try:
            await ws.send_text(json.dumps({
                "type": "progress",
                "stage": stage,
                "progress": progress,
                "message": message,
            }))
        except Exception:
            pass


@app.post("/api/reconstruct/{session_id}")
async def reconstruct(session_id: str, segment: bool = False):
    """
    Trigger 3D reconstruction for a session.

    Query params:
      segment (bool): If true, run Grounding DINO + SAM 2 segmentation
                      on each image before DUSt3R inference so only the
                      main foreground object is reconstructed.
    """
    if session_id not in sessions:
        raise HTTPException(404, f"Session {session_id} not found")

    session = sessions[session_id]
    if session["status"] == "processing":
        raise HTTPException(409, "Reconstruction already in progress")

    session["status"] = "processing"
    session["segment"] = segment
    image_dir = Path(session["image_dir"])
    output_dir = RESULTS_DIR / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()

    def progress_callback(stage, progress, message):
        """Sync callback that schedules async WebSocket send."""
        asyncio.run_coroutine_threadsafe(
            send_progress(session_id, stage, progress, message),
            loop
        )

    try:
        # Run reconstruction in thread pool to not block the event loop
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.reconstruct(image_dir, output_dir, progress_callback, segment=segment)
        )

        session["status"] = "done"
        session["result"] = result

        return {
            "status": "done",
            "n_points": result["n_points"],
            "n_cameras": result["n_cameras"],
        }

    except Exception as e:
        logger.exception(f"Reconstruction failed for {session_id}")
        session["status"] = "error"
        session["error"] = str(e)
        raise HTTPException(500, f"Reconstruction failed: {str(e)}")


@app.get("/api/result/{session_id}/pointcloud")
async def get_pointcloud(session_id: str):
    """Download the reconstructed point cloud as a PLY file."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[session_id]
    if session["status"] != "done":
        raise HTTPException(400, f"Reconstruction not complete. Status: {session['status']}")

    ply_path = Path(session["result"]["ply_path"])
    if not ply_path.exists():
        raise HTTPException(404, "Point cloud file not found")

    return FileResponse(
        ply_path,
        media_type="application/octet-stream",
        filename="reconstruction.ply",
    )


@app.get("/api/result/{session_id}/cameras")
async def get_cameras(session_id: str):
    """Get camera poses for the reconstruction."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[session_id]
    if session["status"] != "done":
        raise HTTPException(400, f"Reconstruction not complete. Status: {session['status']}")

    cameras_path = Path(session["result"]["cameras_path"])
    if not cameras_path.exists():
        raise HTTPException(404, "Cameras file not found")

    with open(cameras_path) as f:
        cameras = json.load(f)

    return JSONResponse(cameras)


@app.get("/api/result/{session_id}/pointcloud-json")
async def get_pointcloud_json(session_id: str):
    """
    Get point cloud data as JSON (positions + colors arrays).
    Parses binary PLY format. Suitable for direct Three.js consumption.
    """
    import struct

    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[session_id]
    if session["status"] != "done":
        raise HTTPException(400, f"Reconstruction not complete. Status: {session['status']}")

    ply_path = Path(session["result"]["ply_path"])
    if not ply_path.exists():
        raise HTTPException(404, "Point cloud file not found")

    # Parse binary PLY
    positions = []
    colors = []
    n_vertices = 0

    with open(ply_path, "rb") as f:
        # Read header (ASCII)
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            if line == "end_header":
                break

        # Read binary vertex data: 3 floats + 3 bytes per vertex
        vertex_size = struct.calcsize('<fffBBB')
        for _ in range(n_vertices):
            data = f.read(vertex_size)
            if len(data) < vertex_size:
                break
            x, y, z, r, g, b = struct.unpack('<fffBBB', data)
            positions.extend([x, y, z])
            colors.extend([r / 255.0, g / 255.0, b / 255.0])

    return JSONResponse({
        "positions": positions,
        "colors": colors,
        "n_points": len(positions) // 3,
    })


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session status and info."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    return sessions[session_id]


@app.get("/api/health")
async def health():
    """Health check."""
    from reconstruction import DUST3R_AVAILABLE
    return {
        "status": "ok",
        "dust3r_available": DUST3R_AVAILABLE,
        "device": DEVICE,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
