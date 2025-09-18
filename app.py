from typing import Optional, List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import pandas as pd
import uvicorn
import io
import os

app = FastAPI(title="Rotating Machine Fault Classifier", description="Upload or stream files to classify faults.")

# ---- Paths ----
MODEL_PATH   = "best_model.keras"
STATIC_DIR   = "static"
FFT_DIR      = os.path.join(STATIC_DIR, "fft")         # normal.npy, imbalance.npy, ...
SAMPLES_DIR  = os.path.join(STATIC_DIR, "samples")     # normal_sample_0.npy, etc.

# ---- Load model ----
model = tf.keras.models.load_model(MODEL_PATH)

# ---- Templates & static ----
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---- Class maps ----
CLASS_ID_TO_NAME = {0: "normal", 1: "imbalance", 2: "misalignment", 3: "bearing"}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_ID_TO_NAME.items()}
CHANNEL_NAMES     = ["IV", "IH", "OV", "OH"]  # 0..3


# ================= Helpers =================
def _read_array_upload(file: UploadFile) -> np.ndarray:
    """Accepts .csv or .npy with shape (4,1013) or (1013,4). Returns (1013,4)."""
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file, header=None)
            arr = df.to_numpy()
        elif file.filename.endswith(".npy"):
            arr = np.load(io.BytesIO(file.file.read()))
        else:
            raise ValueError("Unsupported file. Upload a .csv or .npy file.")
        if arr.shape == (4, 1013):
            arr = arr.T
        elif arr.shape != (1013, 4):
            raise ValueError(f"Expected shape (4,1013) or (1013,4), got {arr.shape}")
        return arr
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _read_array_path(path: str) -> np.ndarray:
    """Reads .csv or .npy from disk; returns (1013,4)."""
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Sample not found: {os.path.basename(path)}")
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path, header=None)
            arr = df.to_numpy()
        elif path.endswith(".npy"):
            arr = np.load(path)
        else:
            raise ValueError("Only .csv or .npy files are supported in samples.")
        if arr.shape == (4, 1013):
            arr = arr.T
        elif arr.shape != (1013, 4):
            raise ValueError(f"{os.path.basename(path)} has shape {arr.shape}, expected (4,1013) or (1013,4)")
        return arr
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _predict(arr_1013x4: np.ndarray):
    """Returns (predicted_class:int, confidence:float)."""
    x = np.expand_dims(arr_1013x4, axis=0)  # (1,1013,4)
    preds = model.predict(x, verbose=0)
    return int(np.argmax(preds, axis=1)[0]), float(np.max(preds))


def _pick_channel(arr_1013x4: np.ndarray, ch_idx: int) -> np.ndarray:
    """Returns a 1D signal for channel index (0..3)."""
    if ch_idx not in (0, 1, 2, 3):
        ch_idx = 0
    return arr_1013x4[:, ch_idx]


def _infer_condition_from_filename(filename: str) -> Optional[str]:
    """
    Infer condition from filename: supports names like normal_sample_0.npy, etc.
    Returns one of: normal / imbalance / misalignment / bearing / None
    """
    if not filename:
        return None
    base = os.path.basename(filename).lower()
    stem, _ = os.path.splitext(base)
    tokens = stem.split("_")
    for key in ("normal", "imbalance", "misalignment", "bearing"):
        if key in tokens or key in stem:
            return key
    return None


def _load_fft_from_disk(condition: str) -> np.ndarray:
    """
    Load a (4,512) FFT matrix from static/fft/<condition>.npy or .csv.
    Raises HTTPException if not found/invalid.
    """
    npy_path = os.path.join(FFT_DIR, f"{condition}.npy")
    if os.path.isfile(npy_path):
        try:
            arr = np.load(npy_path)
            if arr.shape == (4, 512):
                return arr.astype(float)
            raise ValueError(f"{npy_path} has wrong shape {arr.shape}, expected (4,512)")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading {npy_path}: {e}")

    csv_path = os.path.join(FFT_DIR, f"{condition}.csv")
    if os.path.isfile(csv_path):
        try:
            df = pd.read_csv(csv_path, header=None)
            arr = df.to_numpy()
            if arr.shape == (4, 512):
                return arr.astype(float)
            raise ValueError(f"{csv_path} has wrong shape {arr.shape}, expected (4,512)")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading {csv_path}: {e}")

    raise HTTPException(status_code=404, detail=f"No FFT data found for condition '{condition}'")


# ================= Routes: Pages =================
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# ================= Routes: Individual =================
@app.post("/predict-individual/")
async def predict_individual(
    file: UploadFile = File(...),
    channel: int = Form(0),
    sample_rate: Optional[float] = Form(None)  # unused in mocked FFT
):
    arr = _read_array_upload(file)
    predicted_class, confidence = _predict(arr)

    # time waveform
    signal = _pick_channel(arr, channel)
    time_series = list(range(len(signal)))

    # resolve condition (filename first, else predicted class)
    condition = _infer_condition_from_filename(file.filename) or CLASS_ID_TO_NAME.get(predicted_class, "normal")

    # FFT load (graceful)
    fft_error = None
    fft_freqs, fft_vec = [], []
    try:
        fft_bank = _load_fft_from_disk(condition)    # (4,512)
        fft_vec = fft_bank[channel].astype(float).tolist()
        fft_freqs = list(range(len(fft_vec)))
    except HTTPException as he:
        fft_error = he.detail

    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence": confidence,
        "time_series": time_series,
        "signal": signal.astype(float).tolist(),
        "fft_freqs": fft_freqs,
        "fft_magnitude": fft_vec,
        "resolved_condition": condition,
        "fft_error": fft_error
    })


# ================= Routes: Continuous =================
@app.get("/samples")
async def list_samples() -> JSONResponse:
    """
    Return a list of available sample filenames (npy/csv) under static/samples/.
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".npy", ".csv"))]
    files.sort()
    items: List[Dict[str, str]] = [{"filename": f, "condition": _infer_condition_from_filename(f) or ""} for f in files]
    return JSONResponse({"samples": items})


@app.get("/predict-sample")
async def predict_sample(
    filename: str = Query(..., description="A filename under static/samples/"),
    channel: int = Query(0)
):
    """
    Predict using a server-side sample file at static/samples/<filename>.
    Uses the same FFT-by-condition from static/fft/<condition>.npy.
    """
    path = os.path.join(SAMPLES_DIR, filename)
    arr = _read_array_path(path)
    predicted_class, confidence = _predict(arr)

    # time waveform
    signal = _pick_channel(arr, channel)
    time_series = list(range(len(signal)))

    # resolve condition via filename first
    condition = _infer_condition_from_filename(filename) or CLASS_ID_TO_NAME.get(predicted_class, "normal")

    # FFT load (graceful)
    fft_error = None
    fft_freqs, fft_vec = [], []
    try:
        fft_bank = _load_fft_from_disk(condition)    # (4,512)
        fft_vec = fft_bank[channel].astype(float).tolist()
        fft_freqs = list(range(len(fft_vec)))
    except HTTPException as he:
        fft_error = he.detail

    return JSONResponse({
        "filename": filename,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "time_series": time_series,
        "signal": signal.astype(float).tolist(),
        "fft_freqs": fft_freqs,
        "fft_magnitude": fft_vec,
        "resolved_condition": condition,
        "fft_error": fft_error
    })


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

