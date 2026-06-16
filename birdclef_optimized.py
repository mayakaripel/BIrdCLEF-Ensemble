"""
BirdCLEF+ 2026 — FAST PRODUCTION PIPELINE (OPTIMIZED)
Optimized for:
    - Fast startup
    - GPU inference
    - Low RAM
    - Official BirdCLEF metric
    - FAST DATASET LOADING
"""

import gc
import os
import warnings
import psutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠️  TensorFlow not installed - TF models will be skipped")

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("⚠️  ONNX not installed - ONNX models will be skipped")

from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from tqdm.auto import tqdm

# =========================================================
# CONFIG
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COMP_PATH = Path("/kaggle/input/competitions/birdclef-2026")

TRAIN_CSV = COMP_PATH / "train.csv"
TAXONOMY_CSV = COMP_PATH / "taxonomy.csv"
SAMPLE_SUB_CSV = COMP_PATH / "sample_submission.csv"
TEST_SOUNDSCAPES = COMP_PATH / "test_soundscapes"

OUTPUT_PATH = Path("/kaggle/working")

SR = 32000
WINDOW_SEC = 5
HOP_SEC = 2.5

WINDOW_SAMPLES = int(SR * WINDOW_SEC)
HOP_SAMPLES = int(SR * HOP_SEC)

RARE_THRESH = 20
BATCH_SIZE = 64

# OPTIMIZATION: Parallel loading config - ADJUSTED FOR 9.89GB AVAILABLE
# With tight memory, reduce workers to stay safe
N_WORKERS = 2  # Conservative: 2 parallel loads × ~20MB = 40MB active
AUDIO_CACHE_SIZE = 256  # MB
CHUNK_SIZE = 5000  # rows per chunk for CSV reading
ENABLE_AUDIO_CACHING = False  # Keep False to avoid memory creep

print(f"DEVICE: {DEVICE}")
print(f"Workers: {N_WORKERS}")

# =========================================================
# MEMORY MONITORING (for 12GB systems)
# =========================================================

def log_memory():
    """Log current RAM usage"""
    try:
        process = psutil.Process()
        mem = process.memory_info().rss / 1024 / 1024  # MB
        percent = psutil.virtual_memory().percent
        print(f"Memory: {mem:.0f} MB ({percent:.1f}% of system)")
    except:
        pass

def cleanup_memory():
    """Force garbage collection and cache clearing"""
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =========================================================
# FAST GLOBAL TRANSFORMS
# =========================================================

MELSPEC = T.MelSpectrogram(
    sample_rate=SR,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
).to(DEVICE)

AMPLITUDE_TO_DB = T.AmplitudeToDB().to(DEVICE)

RESAMPLERS = {}

# =========================================================
# FAST RESAMPLER CACHE
# =========================================================

def get_resampler(orig_sr, target_sr):

    key = (orig_sr, target_sr)

    if key not in RESAMPLERS:
        RESAMPLERS[key] = T.Resample(orig_sr, target_sr)

    return RESAMPLERS[key]

# =========================================================
# OFFICIAL METRIC
# =========================================================

def birdclef_auc(y_true, y_pred):

    aucs = []

    for c in range(y_true.shape[1]):

        if y_true[:, c].sum() == 0:
            continue

        try:
            auc = roc_auc_score(y_true[:, c], y_pred[:, c])
            aucs.append(auc)

        except ValueError:
            continue

    return float(np.mean(aucs))

# =========================================================
# ENSEMBLE CONFIG
# =========================================================

EnsembleType = Literal[
    "weighted_average",
    "simple_average",
]

@dataclass
class ModelConfig:
    name: str
    weight: float
    path: str

@dataclass
class EnsembleConfig:
    type_add: EnsembleType
    models: List[ModelConfig]

solutions = EnsembleConfig(
    type_add="weighted_average",
    models=[
        ModelConfig(
            name="Perch-ONNX",
            weight=0.60,
            path="/kaggle/input/datasets/tuckerarrants/perch-v2-no-dft-onnx"
        ),
        ModelConfig(
            name="Distilled-SED",
            weight=0.40,
            path="/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public"
        ),
    ]
)

# =========================================================
# OPTIMIZATION: FAST DATASET INDEXING
# =========================================================

def build_file_index():
    """
    FAST: Read CSV in chunks to avoid memory spike
    """
    print("Reading train.csv...")
    
    chunks = []
    for chunk in pd.read_csv(
        TRAIN_CSV,
        usecols=["filename", "primary_label"],
        dtype={
            "filename": "string",
            "primary_label": "category"
        },
        chunksize=10000  # Process 10k rows at a time
    ):
        chunks.append(chunk)
    
    train_df = pd.concat(chunks, ignore_index=True)
    
    counts = train_df["primary_label"].value_counts()

    train_df["train_count"] = (
        train_df["primary_label"]
        .map(counts)
        .astype(np.int16)
    )

    train_df["is_rare"] = (
        train_df["train_count"] < RARE_THRESH
    ).astype(bool)

    audio_dir = str(COMP_PATH / "train_audio")
    train_df["filepath"] = audio_dir + "/" + train_df["filename"]

    print(
        f"{len(train_df)} files | "
        f"{train_df['primary_label'].nunique()} species"
    )

    return train_df

# =========================================================
# OPTIMIZATION: PARALLEL AUDIO LOADER
# =========================================================

def load_audio(path):
    """Load single audio file with validation and caching support"""
    try:
        # Check cache first
        cache_dir = os.environ.get('BIRDCLEF_CACHE')
        if cache_dir:
            cache_path = Path(cache_dir) / Path(path).stem / "audio.pt"
            if cache_path.exists():
                waveform = torch.load(cache_path)
                return waveform

        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(0, keepdim=True)

        if sr != SR:
            waveform = get_resampler(sr, SR)(waveform)

        if waveform.abs().max() < 1e-5:
            return None

        return waveform

    except Exception:
        return None


def load_audio_batch_parallel(filepaths, max_workers=N_WORKERS):
    """
    SAFE SEQUENTIAL LOADER - No threading deadlocks
    Loads files one at a time with progress bar
    Memory-safe for 9.89GB systems
    """
    audio_cache = {}
    failed = 0
    
    for fp in tqdm(filepaths, desc="Loading audio"):
        try:
            waveform = load_audio(fp)
            if waveform is not None:
                audio_cache[fp] = waveform
        except Exception as e:
            failed += 1
    
    print(f"Loaded: {len(audio_cache)}/{len(filepaths)} ({failed} failed)")
    cleanup_memory()
    
    return audio_cache

# =========================================================
# FAST PCEN
# =========================================================

def apply_pcen(waveform):

    waveform = waveform.to(DEVICE)

    spec = MELSPEC(waveform)

    spec = torch.log1p(spec)

    return spec

# =========================================================
# FAST WINDOW EXTRACTION
# =========================================================

def make_windows(waveform):

    if waveform.shape[1] < WINDOW_SAMPLES:
        return None

    windows = waveform.unfold(
        dimension=1,
        size=WINDOW_SAMPLES,
        step=HOP_SAMPLES
    )

    windows = windows.permute(1, 0, 2)

    return windows

# =========================================================
# FAST GPU INFERENCE
# =========================================================

def predict_soundscape(
    filepath,
    model,
    species_cols,
    batch_size=BATCH_SIZE,
):

    waveform = load_audio(filepath)

    if waveform is None:
        return pd.DataFrame()

    windows = make_windows(waveform)

    if windows is None:
        return pd.DataFrame()

    model.eval()

    preds = []

    with torch.no_grad():

        for i in range(0, len(windows), batch_size):

            batch = windows[i:i + batch_size]

            batch = batch.to(DEVICE)

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda")
            ):

                logits = model(batch)

                if isinstance(logits, tuple):
                    logits = logits[0]

                probs = torch.sigmoid(logits)

            preds.append(
                probs.float().cpu().numpy()
            )

    preds = np.concatenate(preds, axis=0)

    # EMA smoothing

    smooth = [preds[0]]

    for i in range(1, len(preds)):

        smooth.append(
            0.4 * preds[i] +
            0.6 * smooth[-1]
        )

    smooth = np.stack(smooth)

    # Aggregate overlapping windows

    rows = []

    soundscape_id = Path(filepath).stem

    steps_per_row = int(5 / HOP_SEC)

    for idx in range(0, len(smooth), steps_per_row):

        group = smooth[idx:idx + steps_per_row]

        agg = np.max(group, axis=0)

        end_sec = (
            (idx // steps_per_row + 1) * 5
        )

        row = {
            "row_id": f"{soundscape_id}_{end_sec}"
        }

        row.update(
            dict(zip(species_cols, agg))
        )

        rows.append(row)

    del waveform
    del windows

    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return pd.DataFrame(rows)

# =========================================================
# OUT-OF-FOLD PREDICTION & ENSEMBLE TUNING
# =========================================================

def generate_oof_predictions(
    train_df,
    model,
    species_cols,
    batch_size=BATCH_SIZE,
    sample_frac=0.2,  # Use 20% of training data for OOF
):
    """
    Generate out-of-fold predictions on training data
    Returns: oof_preds (n_samples, n_species), oof_labels (n_samples, n_species)
    """
    # Sample training files for OOF validation (full data is too slow)
    train_sample = train_df.sample(frac=sample_frac, random_state=42)
    
    preds_list = []
    labels_list = []
    
    for idx, row in tqdm(
        train_sample.iterrows(),
        total=len(train_sample),
        desc="Generating OOF predictions"
    ):
        filepath = row["filepath"]
        primary_label = row["primary_label"]
        
        # Load and predict
        waveform = load_audio(filepath)
        if waveform is None:
            continue
        
        windows = make_windows(waveform)
        if windows is None:
            continue
        
        model.eval()
        preds = []
        
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = windows[i:i + batch_size].to(DEVICE)
                
                with torch.amp.autocast(
                    device_type=DEVICE.type,
                    enabled=(DEVICE.type == "cuda")
                ):
                    logits = model(batch)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    probs = torch.sigmoid(logits)
                
                preds.append(probs.float().cpu().numpy())
        
        # Aggregate across windows (max pooling)
        preds = np.concatenate(preds, axis=0)
        pred_agg = np.max(preds, axis=0)
        
        preds_list.append(pred_agg)
        
        # Create binary label vector
        label_vec = np.zeros(len(species_cols))
        if primary_label in species_cols:
            label_vec[species_cols.index(primary_label)] = 1
        labels_list.append(label_vec)
        
        cleanup_memory()
    
    oof_preds = np.stack(preds_list)
    oof_labels = np.stack(labels_list)
    
    return oof_preds, oof_labels


def load_ensemble_models(solutions_config):
    """Load all ensemble models from disk (supports PyTorch, TensorFlow, ONNX)"""
    models = {}
    
    for model_cfg in solutions_config.models:
        print(f"  Loading {model_cfg.name}...", end=" ", flush=True)
        try:
            # Check if it's an ONNX model
            if model_cfg.path.endswith(".onnx") or "onnx" in model_cfg.path.lower():
                if not HAS_ONNX:
                    print("✗ (ONNX not available)")
                    continue
                
                # Find .onnx file in directory
                model_path = model_cfg.path
                if not model_path.endswith(".onnx"):
                    from pathlib import Path
                    model_dir = Path(model_path)
                    onnx_files = list(model_dir.glob("*.onnx"))
                    if onnx_files:
                        model_path = str(onnx_files[0])
                        print(f"(ONNX: {Path(model_path).name})", end=" ", flush=True)
                
                session = ort.InferenceSession(model_path)
                models[model_cfg.name] = session
                print("✓")
            
            # Check if it's a TensorFlow SavedModel
            elif "tensorflow" in model_cfg.path.lower() or model_cfg.path.endswith("/1"):
                if not HAS_TF:
                    print("✗ (TensorFlow not available)")
                    continue
                
                model = tf.saved_model.load(model_cfg.path)
                models[model_cfg.name] = model
                print("✓ (TensorFlow)")
            
            else:
                # PyTorch model
                model = torch.load(model_cfg.path, map_location=DEVICE)
                model.eval()
                models[model_cfg.name] = model
                print("✓ (PyTorch)")
        
        except Exception as e:
            print(f"✗ Error: {str(e)[:60]}")
    
    return models


def generate_ensemble_oof(train_df, ensemble_models, species_cols):
    """
    Generate OOF predictions for ALL models in ensemble
    Returns list of (oof_preds, oof_labels) tuples
    """
    all_oof = []
    
    for model_name, model in ensemble_models.items():
        print(f"\nGenerating OOF for {model_name}...")
        oof_preds, oof_labels = generate_oof_predictions(
            train_df,
            model,
            species_cols,
            sample_frac=0.2
        )
        all_oof.append(oof_preds)
    
    oof_labels = generate_oof_predictions(
        train_df,
        list(ensemble_models.values())[0],
        species_cols,
        sample_frac=0.2
    )[1]
    
    return all_oof, oof_labels

def optimise_ensemble_weights(
    oof_preds,
    oof_labels,
):

    n_models = len(oof_preds)

    stack = np.stack(oof_preds)

    def objective(w):

        blend = np.einsum(
            "m,mnc->nc",
            w,
            stack
        )

        try:
            return -birdclef_auc(
                oof_labels,
                blend
            )

        except:
            return np.inf

    constraints = {
        "type": "eq",
        "fun": lambda w: w.sum() - 1
    }

    bounds = [(0, 1)] * n_models

    best_score = np.inf
    best = None

    rng = np.random.default_rng(42)

    for _ in range(20):

        w0 = rng.dirichlet(
            np.ones(n_models)
        )

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.fun < best_score:
            best_score = result.fun
            best = result

    print("Optimized Weights")

    for model, weight in zip(
        solutions.models,
        best.x
    ):
        print(model.name, round(weight, 4))

    print("CV:", -best_score)

    return best.x

# =========================================================
# ENSEMBLE PREDICTION ENGINE
# =========================================================

def predict_soundscape_ensemble(
    filepath,
    models_dict,
    species_cols,
    weights,
    batch_size=BATCH_SIZE,
):
    """
    Generate ensemble predictions on a soundscape
    Returns: DataFrame with row_id and species probabilities
    """
    waveform = load_audio(filepath)
    if waveform is None:
        return pd.DataFrame()

    windows = make_windows(waveform)
    if windows is None:
        return pd.DataFrame()

    # Collect predictions from each model
    all_model_preds = []
    
    for model_name, model in models_dict.items():
        model.eval()
        preds = []

        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = windows[i:i + batch_size].to(DEVICE)

                with torch.amp.autocast(
                    device_type=DEVICE.type,
                    enabled=(DEVICE.type == "cuda")
                ):
                    logits = model(batch)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    probs = torch.sigmoid(logits)

                preds.append(probs.float().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        all_model_preds.append(preds)

    # Stack predictions: (n_models, n_windows, n_species)
    all_model_preds = np.stack(all_model_preds, axis=0)

    # EMA smoothing per model
    smoothed = []
    for model_preds in all_model_preds:
        smooth = [model_preds[0]]
        for i in range(1, len(model_preds)):
            smooth.append(0.4 * model_preds[i] + 0.6 * smooth[-1])
        smoothed.append(np.stack(smooth))
    
    smoothed = np.stack(smoothed, axis=0)

    # Weighted ensemble: (n_windows, n_species)
    ensemble_preds = np.average(
        smoothed,
        axis=0,
        weights=weights
    )

    # Aggregate overlapping windows
    rows = []
    soundscape_id = Path(filepath).stem
    steps_per_row = int(5 / HOP_SEC)

    for idx in range(0, len(ensemble_preds), steps_per_row):
        group = ensemble_preds[idx:idx + steps_per_row]
        agg = np.max(group, axis=0)
        end_sec = ((idx // steps_per_row + 1) * 5)

        row = {"row_id": f"{soundscape_id}_{end_sec}"}
        row.update(dict(zip(species_cols, agg)))
        rows.append(row)

    del waveform
    del windows
    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def build_submission_ensemble(
    sample_sub,
    models_dict,
    species_cols,
    weights,
):
    """
    Generate full submission with ensemble predictions
    Kaggle submission format: row_id + species columns with probabilities
    """
    # Debug: Check if test directory exists and has files
    if not TEST_SOUNDSCAPES.exists():
        print(f"✗ Test soundscapes directory not found: {TEST_SOUNDSCAPES}")
        print(f"  Looking in: {COMP_PATH}")
        print(f"  Available directories:")
        if COMP_PATH.exists():
            for item in COMP_PATH.iterdir():
                print(f"    - {item.name}")
        return pd.DataFrame()
    
    # Find all audio files (try multiple extensions)
    soundscapes = list(TEST_SOUNDSCAPES.glob("*.ogg"))
    if not soundscapes:
        soundscapes = list(TEST_SOUNDSCAPES.glob("*.mp3"))
    if not soundscapes:
        soundscapes = list(TEST_SOUNDSCAPES.glob("*.wav"))
    
    print(f"Found {len(soundscapes)} test soundscapes")
    
    if len(soundscapes) == 0:
        print(f"✗ No audio files found in {TEST_SOUNDSCAPES}")
        print(f"  Contents:")
        for item in sorted(TEST_SOUNDSCAPES.iterdir())[:20]:
            print(f"    - {item.name}")
        return pd.DataFrame()
    
    soundscapes = sorted(soundscapes)
    all_rows = []

    for fp in tqdm(soundscapes, desc="Ensemble inference"):
        df = predict_soundscape_ensemble(
            str(fp),
            models_dict,
            species_cols,
            weights
        )
        
        if not df.empty:
            all_rows.append(df)

    if len(all_rows) == 0:
        print("✗ No predictions generated from any soundscape")
        return pd.DataFrame()

    submission = pd.concat(all_rows, ignore_index=True)

    # Validate row_id format
    expected = set(sample_sub.row_id)
    got = set(submission.row_id)
    missing = expected - got

    if missing:
        print(f"⚠️  Missing {len(missing)} rows - filling with 0s")
        
        for row_id in missing:
            missing_row = {"row_id": row_id}
            missing_row.update({col: 0.0 for col in species_cols})
            all_rows.append(pd.DataFrame([missing_row]))
        
        submission = pd.concat(all_rows, ignore_index=True)

    # Reorder columns to match sample submission (row_id first, then species)
    submission = submission[sample_sub.columns]
    
    # Ensure all values are floats and bounded [0, 1]
    for col in species_cols:
        submission[col] = submission[col].clip(0, 1).astype(np.float32)
    
    # Sort by row_id for consistency
    submission = submission.sort_values("row_id").reset_index(drop=True)
    
    print(f"\n✓ Submission shape: {submission.shape}")
    print(f"✓ Rows: {len(submission)}, Columns: {len(submission.columns)}")
    print(f"✓ Expected rows: {len(expected)}, Got: {len(submission)}")
    
    return submission

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # Enable waveform cache if available (10x speedup)
    import os
    os.environ['BIRDCLEF_CACHE'] = '/kaggle/input/datasets/tuckerarrants/birdclef-2026-waveform-cache'
    
    print("=" * 50)
    print("BirdCLEF+ 2026 ENSEMBLE SUBMISSION PIPELINE")
    print("=" * 50)
    
    print("\n[1/6] Loading training index...")
    train_df = build_file_index()
    log_memory()

    print("\n[2/6] Reading sample submission...")
    sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

    species_cols = [
        c for c in sample_sub.columns
        if c != "row_id"
    ]

    print(f"✓ Species count: {len(species_cols)}")
    print(f"✓ Sample submission rows: {len(sample_sub)}")

    print("\n[3/6] Loading ensemble models...")
    ensemble_models = load_ensemble_models(solutions)
    
    if len(ensemble_models) == 0:
        print("✗ FATAL: No models loaded. Check model paths.")
        print(f"  Checked paths:")
        for model in solutions.models:
            print(f"    - {model.name}: {model.path}")
        exit(1)
    
    print(f"✓ Loaded {len(ensemble_models)} models: {list(ensemble_models.keys())}")
    log_memory()

    print("\n[4/6] Preparing ensemble weights...")
    weights = [model.weight for model in solutions.models]
    print("Ensemble configuration:")
    for model_cfg, w in zip(solutions.models, weights):
        print(f"  {model_cfg.name}: {w:.4f}")

    print("\n[5/6] Generating submission...")
    submission = build_submission_ensemble(
        sample_sub,
        ensemble_models,
        species_cols,
        weights
    )
    
    if len(submission) == 0:
        print("✗ FATAL: No predictions generated")
        exit(1)
    
    log_memory()

    print("\n[6/6] Saving submission...")
    submission_path = OUTPUT_PATH / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"✓ Saved to {submission_path}")
    
    print("\n" + "=" * 50)
    print("SUBMISSION VALIDATION")
    print("=" * 50)
    print(f"Shape: {submission.shape}")
    print(f"Columns: {list(submission.columns[:5])}... (showing first 5)")
    print(f"\nFirst 3 rows:")
    print(submission.head(3))
    print(f"\nValue statistics (species predictions):")
    print(f"  Min: {submission.iloc[:, 1:].values.min():.6f}")
    print(f"  Max: {submission.iloc[:, 1:].values.max():.6f}")
    print(f"  Mean: {submission.iloc[:, 1:].values.mean():.6f}")
    
    nan_count = submission.isna().sum().sum()
    if nan_count > 0:
        print(f"⚠️  WARNING: {nan_count} NaN values detected")
    else:
        print(f"✓ No NaN values")
    
    cleanup_memory()
    print("\n✅ SUBMISSION COMPLETE - Ready for Kaggle!")
