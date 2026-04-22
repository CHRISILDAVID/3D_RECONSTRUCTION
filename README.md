<div align="center">
  <h1>🌌 3D Scene Reconstruction Web App</h1>
  <p><strong>Transform 2D images into dense, interactive 3D point clouds in seconds using DUSt3R.</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
  [![Vite](https://img.shields.io/badge/Vite-React-646CFF?logo=vite)](https://vitejs.dev/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
  [![DUSt3R](https://img.shields.io/badge/DUSt3R-ViT_Large-orange)](https://github.com/naver/dust3r)
</div>

<br />

Easily generate vivid 3D scene reconstructions from multi-view images using the state-of-the-art **DUSt3R** (ViT-Large, 512px) model. This project features a robust **FastAPI backend** to handle image processing and 3D inference, and a sleek **React (Vite) + Three.js frontend** for lightning-fast model viewing right in your browser.

## ✨ Features

- **🚀 State-of-the-Art Reconstruction:** Utilizes DUSt3R for dense 3D point cloud generation without requiring camera poses.
- **📸 Multi-View Support:** Upload between 2–12 casually taken images of any scene. 
- **🌐 Interactive 3D Viewer:** Explore the colored 3D point cloud dynamically in your browser (powered by Three.js).
- **💾 Export Ready:** Save the output as a `.ply` file for use in Blender, MeshLab, or CloudCompare.
- **⚡ Fast Processing:** Inference is heavily optimized using PyTorch CUDA (processing typically takes 30-120 seconds).

## 🛠️ Architecture

* **Frontend:** React, Vite, Three.js (React Three Fiber for Point Cloud Visualization).
* **Backend:** FastAPI, Python, PyTorch, OpenCV.
* **Core Machine Learning:** [NAVER DUSt3R](https://github.com/naver/dust3r) (Dense and Unconstrained Stereo 3D Reconstruction).

---

## 🚀 Getting Started

### Prerequisites
* **Python** 3.10+
* **Node.js** 18+
* **NVIDIA GPU** (4 GB VRAM Minimum)
* **CUDA Driver** 11.8+
* **Git**

### 1. Clone the Repositories
Because the project depends on DUSt3R, make sure to clone it inside the backend.
```bash
git clone https://github.com/CHRISILDAVID/3d-reconstruction.git
cd 3d-reconstruction

# Clone DUSt3R (with submodules) into the backend
cd backend
git clone --recursive https://github.com/naver/dust3r.git
```

### 2. Backend Setup
Set up your Python virtual environment and install dependencies (make sure to install PyTorch with CUDA support first).
```bash
cd backend
python -m venv venv

# Activate venv (Windows)
venv\Scripts\activate

# IMPORTANT: Install CUDA-accelerated PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup
Install the React web client dependencies.
```bash
cd ../frontend
npm install
```

---

## 🎮 Usage

You need to run both the backend API and the frontend client simultaneously. Open two terminals.

**Terminal 1 (Backend):**
```bash
cd backend
venv\Scripts\activate
python main.py
```
*Note: On your very first run, it will automatically download the ~400MB model weights from HuggingFace.*

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

1. Open your browser and navigate to **[http://localhost:5173](http://localhost:5173)**.
2. Drag & drop 2–12 overlapping images of your scene.
3. Hit **Reconstruct** and wait for DUSt3R to process the images.
4. Tweak the point size & explore your gorgeous 3D point cloud interactively!

## 📂 Project Structure

```text
3d-reconstruction/
├── backend/
│   ├── main.py              # FastAPI server orchestrator
│   ├── reconstruction.py    # DUSt3R / OpenCV implementation
│   ├── dust3r/              # NAVER DUSt3R cloned repository
│   ├── results/             # Generated PLY point clouds
│   └── uploads/             # Ephemeral image upload storage
├── frontend/
│   ├── src/App.jsx          # Main React Application
│   ├── src/components/      # UI: Dropzone & Three.js Canvas
│   ├── vite.config.js       # Vite configurator
└── SETUP.md                 # Detailed initial setup instructions
```

## ⚠️ Troubleshooting

- **Out of Memory (OOM):** If your GPU throws a CUDA Out of Memory error, try reducing the number of uploaded images to 3-5 images.
- **CPU Fallback:** If inference is extremely slow and the backend logs `DUSt3R model ready on cpu`, PyTorch missed the CUDA libraries. Reinstall PyTorch using the `--index-url https://download.pytorch.org/whl/cu121` flag.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 
Feel free to check [issues page](https://github.com/CHRISILDAVID/3d-reconstruction/issues) if you want to contribute.

## 📜 License

This application is provided under the MIT License. 
*Note:* The DUSt3R model itself operates under its independent license (CC-BY-NC 4.0), meaning commercial use of the raw model requires explicit permission from Naver Corp. Please consult `backend/dust3r/LICENSE` for more detailed information.
