from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import pandas as pd
import uvicorn
import io
import os

app = FastAPI(title="Rotating Machine Fault Classifier", description="Upload a file to classify faults.")

# ---- Load model ----
MODEL_PATH = "best_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ---- Templates & static ----
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---- Class maps ----
CLASS_ID_TO_NAME = {0: "normal", 1: "imbalance", 2: "misalignment", 3: "bearing"}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_ID_TO_NAME.items()}
CHANNEL_NAMES = ["IV", "IH", "OV", "OH"]  # 0..3

# ================= Helpers =================
def _read_array(file: UploadFile) -> np.ndarray:
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
            arr = arr.T               # -> (1013,4)
        elif arr.shape == (1013, 4):
            pass
        else:
            raise ValueError(f"Expected shape (4,1013) or (1013,4), got {arr.shape}")

        return arr
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _predict(arr_1013x4: np.ndarray):
    """Returns (predicted_class:int, confidence:float)."""
    x = np.expand_dims(arr_1013x4, axis=0)  # (1,1013,4)
    preds = model.predict(x, verbose=0)
    predicted_class = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return predicted_class, confidence


def _pick_channel(arr_1013x4: np.ndarray, ch_idx: int) -> np.ndarray:
    """Returns a 1D signal for channel index (0..3)."""
    if ch_idx not in (0, 1, 2, 3):
        ch_idx = 0
    return arr_1013x4[:, ch_idx]


def _infer_condition_from_filename(filename: str) -> Optional[str]:
    """
    Infer condition from filename (supports names like normal_sample_0.npy, etc.).
    Returns one of: normal / imbalance / misalignment / bearing / None
    """
    if not filename:
        return None
    base = os.path.basename(filename).lower()
    stem, _ = os.path.splitext(base)
    # allow token or substring match
    tokens = stem.split("_")
    for key in ("normal", "imbalance", "misalignment", "bearing"):
        if key in tokens or key in stem:
            return key
    return None


# ================= FFT loader (disk only) =================
FFT_DIR = os.path.join("static", "fft")

def _load_fft_from_disk(condition: str) -> np.ndarray:
    """
    Load a (4,512) FFT matrix from:
      - static/fft/<condition>.npy
      - static/fft/<condition>.csv
    Raises HTTPException if not found/invalid.
    """
    # .npy
    npy_path = os.path.join(FFT_DIR, f"{condition}.npy")
    if os.path.isfile(npy_path):
        try:
            arr = np.load(npy_path)
            if arr.shape == (4, 512):
                return arr.astype(float)
            raise ValueError(f"{npy_path} has wrong shape {arr.shape}, expected (4,512)")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading {npy_path}: {e}")

    # .csv
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


# ================= Routes =================
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict-individual/")
async def predict_individual(
    file: UploadFile = File(...),
    channel: int = Form(0),
    sample_rate: Optional[float] = Form(None)  # unused in mocked FFT
):
    """
    Returns:
      - predicted_class, confidence
      - time waveform (selected channel)
      - FFT (selected channel) loaded from disk based on condition
    """
    arr = _read_array(file)
    predicted_class, confidence = _predict(arr)

    # time waveform
    signal = _pick_channel(arr, channel)
    time_series = list(range(len(signal)))

    # resolve condition (filename first, else predicted class)
    condition = _infer_condition_from_filename(file.filename) or CLASS_ID_TO_NAME.get(predicted_class, "normal")

    # try FFT load; if it fails, still return prediction + waveform
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


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
