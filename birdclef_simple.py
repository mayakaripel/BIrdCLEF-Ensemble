"""
BirdCLEF+ 2026 — SIMPLE WORKING PIPELINE
Minimal viable submission generator
"""

import os
import sys
import gc
from pathlib import Path

# Force output flushing
def log(msg, end="\n"):
    print(msg, end=end, flush=True)

log("=" * 60)
log("BirdCLEF+ 2026 SUBMISSION PIPELINE")
log("=" * 60)

# Import heavy libraries
log("\n[INIT] Importing libraries...")
import subprocess
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T

log("✓ PyTorch imports done")

try:
    import onnxruntime as ort
    HAS_ONNX = True
    log("✓ ONNX Runtime available")
except Exception as e:
    HAS_ONNX = False
    log(f"⚠️  ONNX Runtime not found, installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"])
        import onnxruntime as ort
        HAS_ONNX = True
        log("✓ ONNX Runtime installed successfully")
    except Exception as e2:
        log(f"✗ Failed to install ONNX: {e2}")

# Configure ONNX Runtime to use CPU provider (more reliable than GPU with incompatible CUDA)
os.environ['ONNXRUNTIME_DEVICE'] = 'cpu'

# =========================================================
# CONFIG
# =========================================================

# Force CPU to avoid CUDA compatibility issues
DEVICE = torch.device("cpu")
log(f"\nDevice: {DEVICE} (using CPU for audio processing)")

COMP_PATH = Path("/kaggle/input/competitions/birdclef-2026")
TEST_SOUNDSCAPES = COMP_PATH / "test_soundscapes"
SAMPLE_SUB_CSV = COMP_PATH / "sample_submission.csv"
OUTPUT_PATH = Path("/kaggle/working")

SR = 32000
WINDOW_SEC = 5
HOP_SEC = 2.5
WINDOW_SAMPLES = int(SR * WINDOW_SEC)
HOP_SAMPLES = int(SR * HOP_SEC)
BATCH_SIZE = 16

log(f"SR: {SR}, Window: {WINDOW_SEC}s, Hop: {HOP_SEC}s")

# =========================================================
# TRANSFORMS
# =========================================================

MELSPEC = T.MelSpectrogram(
    sample_rate=SR,
    n_fft=2048,
    hop_length=512,
    n_mels=256,
).to(DEVICE)

# =========================================================
# AUDIO LOADER
# =========================================================

RESAMPLERS = {}

def get_resampler(orig_sr, target_sr):
    key = (orig_sr, target_sr)
    if key not in RESAMPLERS:
        RESAMPLERS[key] = T.Resample(orig_sr, target_sr).to(DEVICE)
    return RESAMPLERS[key]

def load_audio(path):
    try:
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdim=True)
        
        if sr != SR:
            waveform = get_resampler(sr, SR)(waveform)
        
        if waveform.abs().max() < 1e-5:
            return None
        
        return waveform
    except:
        return None

# =========================================================
# WINDOW EXTRACTION
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
# MEL SPECTROGRAM EXTRACTION
# =========================================================

def extract_mel_spectrogram(batch):
    """
    Extract log-mel spectrogram from audio batch
    
    Args:
        batch: (batch_size, 1, WINDOW_SAMPLES) audio windows
    
    Returns:
        (batch_size, 1, 256, time_steps) log-mel spectrogram
    """
    batch = batch.to(DEVICE)
    
    # Mel spectrogram: (batch, 1, 256, time_steps)
    mel = MELSPEC(batch)
    
    # Log scale for SED model training
    mel = torch.log(mel + 1e-6)
    
    # Convert to numpy
    mel = mel.cpu().numpy().astype(np.float32)
    
    return mel

# =========================================================
# LOAD MODELS
# =========================================================

def load_sed_ensemble(model_dir):
    """Load 5-fold SED ensemble"""
    if not HAS_ONNX:
        log("✗ ONNX not available")
        return []
    
    sessions = []
    input_names = []
    
    log(f"Loading 5-fold SED ensemble from {model_dir}...")
    
    for fold in range(5):
        try:
            model_path = Path(model_dir) / f"sed_fold{fold}.onnx"
            
            if not model_path.exists():
                log(f"  ✗ Fold {fold}: {model_path} not found")
                continue
            
            session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            
            sessions.append(session)
            input_names.append(input_name)
            
            log(f"  ✓ Fold {fold}: loaded (input='{input_name}')")
        except Exception as e:
            log(f"  ✗ Fold {fold}: {str(e)[:60]}")
            continue
    
    if len(sessions) == 0:
        log("✗ FATAL: No SED models loaded")
        return []
    
    log(f"✓ Loaded {len(sessions)}/5 models")
    return sessions, input_names

# =========================================================
# INFERENCE
# =========================================================

def predict_soundscape(filepath, sessions, input_names, species_cols):
    """Generate predictions for one soundscape using 5-fold ensemble"""
    
    waveform = load_audio(filepath)
    if waveform is None:
        return pd.DataFrame()
    
    windows = make_windows(waveform)
    if windows is None:
        return pd.DataFrame()
    
    fold_preds = []
    
    # Run inference on all 5 folds
    for fold_idx, (session, input_name) in enumerate(zip(sessions, input_names)):
        preds = []
        
        for i in range(0, len(windows), BATCH_SIZE):
            batch = windows[i:i + BATCH_SIZE]
            
            try:
                # Extract log-mel spectrogram
                mel_input = extract_mel_spectrogram(batch)
                
                # Run ONNX inference
                output = session.run(None, {input_name: mel_input})
                
                # Apply sigmoid to ensure [0, 1] range
                batch_output = 1.0 / (1.0 + np.exp(-output[0]))
                preds.append(batch_output)
            except Exception as e:
                log(f"✗ Inference error on fold {fold_idx} batch {i}: {str(e)[:60]}")
                continue
        
        if preds:
            fold_preds.append(np.concatenate(preds, axis=0))
    
    if not fold_preds:
        return pd.DataFrame()
    
    # Average across folds
    preds = np.mean(fold_preds, axis=0)
    
    # Ensure values are in [0, 1]
    preds = np.clip(preds, 0, 1)
    
    # EMA smoothing
    smooth = [preds[0]]
    for i in range(1, len(preds)):
        smooth.append(0.4 * preds[i] + 0.6 * smooth[-1])
    smooth = np.stack(smooth)
    
    # Aggregate windows
    rows = []
    soundscape_id = Path(filepath).stem
    steps_per_row = int(5 / HOP_SEC)
    
    for idx in range(0, len(smooth), steps_per_row):
        group = smooth[idx:idx + steps_per_row]
        agg = np.max(group, axis=0)
        end_sec = ((idx // steps_per_row + 1) * 5)
        
        row = {"row_id": f"{soundscape_id}_{end_sec}"}
        row.update(dict(zip(species_cols, agg)))
        rows.append(row)
    
    del waveform, windows
    gc.collect()
    
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    
    return pd.DataFrame(rows)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    
    try:
        log("\n[1/5] Reading sample submission...")
        sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
        species_cols = [c for c in sample_sub.columns if c != "row_id"]
        log(f"✓ Species: {len(species_cols)}, Rows: {len(sample_sub)}")
        
        log("\n[2/5] Loading model...")
        model_dir = "/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public"
        sessions, input_names = load_sed_ensemble(model_dir)
        
        if not sessions:
            log("✗ Failed to load models")
            sys.exit(1)
        
        log("\n[3/5] Finding test soundscapes...")
        
        # Primary location: ONLY use test_soundscapes, never recursively search all inputs
        soundscapes = sorted(list(TEST_SOUNDSCAPES.glob("*.ogg")))
        
        if len(soundscapes) == 0:
            log(f"⚠️  No soundscapes found in {TEST_SOUNDSCAPES}")
            log("This is normal during local testing (Kaggle hides test files until submission)")
            log("\nGenerating dummy submission...")
            
            # Generate dummy submission
            rows = []
            for i in range(len(sample_sub)):
                row = {"row_id": sample_sub.iloc[i]["row_id"]}
                for col in species_cols:
                    row[col] = 0.0
                rows.append(row)
            
            submission = pd.DataFrame(rows)
            submission = submission[sample_sub.columns]
        
        else:
            log(f"✓ Found {len(soundscapes)} soundscapes")
            
            log("\n[4/5] Generating predictions...")
            all_rows = []
            
            # Process all soundscapes (large dataset)
            for i, fp in enumerate(soundscapes):
                if i % 100 == 0:
                    log(f"  [{i}/{len(soundscapes)}] Processing...")
                
                df = predict_soundscape(str(fp), sessions, input_names, species_cols)
                
                if not df.empty:
                    all_rows.append(df)
            
            if not all_rows:
                log("✗ No predictions generated!")
                sys.exit(1)
            
            log(f"\n  Concatenating {len(all_rows)} dataframes...")
            submission = pd.concat(all_rows, ignore_index=True)
            submission = submission[sample_sub.columns]
        
        log(f"\n[5/5] Saving submission...")
        submission_path = OUTPUT_PATH / "submission.csv"
        submission.to_csv(submission_path, index=False)
        log(f"✓ Saved to {submission_path}")
        
        log("\n" + "=" * 60)
        log("SUBMISSION SUMMARY")
        log("=" * 60)
        log(f"Shape: {submission.shape}")
        log(f"Rows: {len(submission)}, Columns: {len(submission.columns)}")
        
        if len(submission) > 0:
            log(f"\nFirst 3 rows:")
            log(submission.head(3).to_string())
            
            # Check value ranges
            val_min = submission.iloc[:, 1:].values.min()
            val_max = submission.iloc[:, 1:].values.max()
            val_mean = submission.iloc[:, 1:].values.mean()
            
            log(f"\nValue statistics:")
            log(f"  Min: {val_min:.6f}")
            log(f"  Max: {val_max:.6f}")
            log(f"  Mean: {val_mean:.6f}")
            
            if val_min < 0 or val_max > 1:
                log(f"⚠️  WARNING: Values outside [0, 1]!")
        
        log("\n✅ SUBMISSION COMPLETE")
        
    except Exception as e:
        log(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
