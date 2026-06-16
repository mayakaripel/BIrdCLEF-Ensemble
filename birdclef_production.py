"""
BirdCLEF+ 2026 — PRODUCTION PIPELINE
Five-fold distilled SED ensemble with clip/frame fusion
Matches paper methodology exactly
Defensive coding for Kaggle's hidden test set
"""

import os
import sys
import gc
import json
from pathlib import Path
from typing import List, Tuple, Optional

# Force output flushing
def log(msg):
    print(msg, flush=True)

log("=" * 70)
log("BirdCLEF+ 2026 PRODUCTION PIPELINE")
log("=" * 70)

# Import heavy libraries
log("\n[INIT] Importing libraries...")
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
    log(f"✗ ONNX Runtime not available: {e}")
    sys.exit(1)

# Configure ONNX Runtime to use CPU provider (more reliable than GPU with incompatible CUDA)
os.environ['ONNXRUNTIME_DEVICE'] = 'cpu'

# =========================================================
# CONFIG
# =========================================================

# Force CPU to avoid CUDA compatibility issues
# ONNX Runtime will handle its own device placement
DEVICE = torch.device("cpu")
log(f"Device: {DEVICE} (using CPU for audio processing)")

COMP_PATH = Path("/kaggle/input/competitions/birdclef-2026")
TEST_SOUNDSCAPES = COMP_PATH / "test_soundscapes"
SAMPLE_SUB_CSV = COMP_PATH / "sample_submission.csv"
OUTPUT_PATH = Path("/kaggle/working")

# Audio parameters (from paper Section 3.1, 3.2)
SR = 32000           # Sample rate: 32 kHz
WINDOW_SEC = 5       # Window: 5 seconds
HOP_SEC = 2.5        # Hop: 2.5 seconds
WINDOW_SAMPLES = int(SR * WINDOW_SEC)
HOP_SAMPLES = int(SR * HOP_SEC)
BATCH_SIZE = 16

# Spectrogram parameters (from paper Section 3.2)
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256

# Fusion weights (from paper Section 3.4)
CLIP_WEIGHT = 0.3
FRAME_WEIGHT = 0.7

# EMA smoothing (from paper Section 3.6)
EMA_ALPHA = 0.4
EMA_BETA = 0.6

# Model paths
MODEL_DIR = "/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public"
NUM_FOLDS = 5

log(f"\nConfig:")
log(f"  SR: {SR}, Window: {WINDOW_SEC}s, Hop: {HOP_SEC}s")
log(f"  FFT: {N_FFT}, Hop: {HOP_LENGTH}, Mel bins: {N_MELS}")
log(f"  Clip weight: {CLIP_WEIGHT}, Frame weight: {FRAME_WEIGHT}")
log(f"  EMA: α={EMA_ALPHA}, β={EMA_BETA}")

# =========================================================
# GLOBAL TRANSFORMS
# =========================================================

MELSPEC = T.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0,
).to(DEVICE)

AMPLITUDE_TO_DB = T.AmplitudeToDB(top_db=80).to(DEVICE)

RESAMPLERS = {}

# =========================================================
# UTILITIES
# =========================================================

def get_resampler(orig_sr, target_sr):
    """Get or create resampler for given sample rates"""
    key = (orig_sr, target_sr)
    if key not in RESAMPLERS:
        RESAMPLERS[key] = T.Resample(orig_sr, target_sr).to(DEVICE)
    return RESAMPLERS[key]

def cleanup_memory():
    """Force garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except:
            pass  # Ignore CUDA errors if GPU is unavailable

def log_memory():
    """Log RAM usage"""
    try:
        import psutil
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        percent = psutil.virtual_memory().percent
        log(f"  Memory: {mem_mb:.0f} MB ({percent:.1f}% system)")
    except:
        pass

# =========================================================
# ONNX MODEL LOADING
# =========================================================

class ONNXModelEnsemble:
    """Manage 5-fold ONNX SED ensemble"""
    
    def __init__(self, model_dir: str, num_folds: int = 5):
        self.sessions = []
        self.input_names = []
        self.output_names = []
        self.fold_info = []
        
        log(f"\n[LOADING] {num_folds} SED models...")
        
        for fold in range(num_folds):
            path = Path(model_dir) / f"sed_fold{fold}.onnx"
            
            if not path.exists():
                log(f"  ✗ Fold {fold}: {path} not found")
                continue
            
            try:
                # Use CPU provider explicitly (avoid CUDA compatibility issues)
                sess = ort.InferenceSession(
                    str(path), 
                    providers=['CPUExecutionProvider']
                )
                self.sessions.append(sess)
                
                # Extract I/O names
                input_name = sess.get_inputs()[0].name
                output_names = [o.name for o in sess.get_outputs()]
                
                self.input_names.append(input_name)
                self.output_names.append(output_names)
                
                # Log model signature
                input_shape = sess.get_inputs()[0].shape
                self.fold_info.append({
                    'fold': fold,
                    'path': str(path),
                    'input_name': input_name,
                    'input_shape': list(input_shape),
                    'output_names': output_names,
                })
                
                log(f"  ✓ Fold {fold}: input='{input_name}' shape={input_shape}")
                
            except Exception as e:
                log(f"  ✗ Fold {fold}: {str(e)[:80]}")
                continue
        
        if len(self.sessions) == 0:
            log("✗ FATAL: No models loaded")
            sys.exit(1)
        
        log(f"✓ Loaded {len(self.sessions)}/{num_folds} models")
        log_memory()
    
    def inference(self, mel_input: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Run inference on all folds
        
        Args:
            mel_input: (batch, 1, 256, 313) mel spectrogram
        
        Returns:
            (clip_preds, frame_preds) — lists of (batch, 234) arrays
        """
        clip_preds = []
        frame_preds = []
        
        for fold_idx, (sess, input_name, output_names) in enumerate(
            zip(self.sessions, self.input_names, self.output_names)
        ):
            try:
                # Run model
                outputs = sess.run(None, {input_name: mel_input})
                
                # Parse outputs (typically: clip_logits, frame_logits)
                if len(outputs) >= 2:
                    clip_logits, frame_logits = outputs[0], outputs[1]
                else:
                    log(f"  ⚠️  Fold {fold_idx}: Expected 2 outputs, got {len(outputs)}")
                    clip_logits = outputs[0]
                    frame_logits = outputs[0]  # Fallback
                
                clip_preds.append(clip_logits)
                frame_preds.append(frame_logits)
                
            except Exception as e:
                log(f"  ✗ Fold {fold_idx} inference failed: {str(e)[:80]}")
                # Use zero predictions as fallback
                clip_preds.append(np.zeros((mel_input.shape[0], 234), dtype=np.float32))
                frame_preds.append(np.zeros((mel_input.shape[0], 1, 234), dtype=np.float32))
        
        return clip_preds, frame_preds

# =========================================================
# AUDIO LOADING
# =========================================================

def load_audio(path: str) -> Optional[torch.Tensor]:
    """Load audio file and convert to mono at 32 kHz"""
    try:
        waveform, sr = torchaudio.load(path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != SR:
            resampler = get_resampler(sr, SR)
            waveform = resampler(waveform)
        
        # Check for silent audio
        if waveform.abs().max() < 1e-5:
            return None
        
        return waveform
        
    except Exception as e:
        log(f"    ⚠️  Failed to load {Path(path).name}: {str(e)[:60]}")
        return None

# =========================================================
# WINDOW EXTRACTION
# =========================================================

def make_windows(waveform: torch.Tensor) -> Optional[torch.Tensor]:
    """Extract overlapping 5-second windows"""
    if waveform.shape[1] < WINDOW_SAMPLES:
        return None
    
    # Unfold: (1, num_samples) -> (1, num_windows, WINDOW_SAMPLES)
    windows = waveform.unfold(
        dimension=1,
        size=WINDOW_SAMPLES,
        step=HOP_SAMPLES
    )
    
    # Permute: (1, num_windows, WINDOW_SAMPLES) -> (num_windows, 1, WINDOW_SAMPLES)
    windows = windows.permute(1, 0, 2)
    
    return windows

# =========================================================
# MEL SPECTROGRAM EXTRACTION
# =========================================================

def extract_mel_spectrogram(batch: torch.Tensor) -> np.ndarray:
    """
    Extract mel spectrogram from audio batch
    
    Args:
        batch: (batch_size, 1, WINDOW_SAMPLES) audio windows
    
    Returns:
        (batch_size, 1, 256, 313) mel spectrogram in dB
    """
    batch = batch.to(DEVICE)
    
    # Mel spectrogram: (batch, 1, n_mels, time_steps)
    mel = MELSPEC(batch)
    
    # Log scale: convert to dB
    mel = AMPLITUDE_TO_DB(mel)
    
    # Convert to numpy
    mel = mel.cpu().numpy().astype(np.float32)
    
    # Ensure shape
    if mel.ndim == 3:
        mel = mel[:, None, :, :]  # Add channel dimension if missing
    
    return mel

# =========================================================
# INFERENCE PIPELINE
# =========================================================

def predict_soundscape(
    filepath: str,
    ensemble: ONNXModelEnsemble,
    species_cols: List[str],
) -> pd.DataFrame:
    """
    Generate predictions for one soundscape
    
    Pipeline:
    1. Load audio and extract windows
    2. For each batch:
       a. Extract mel spectrogram
       b. Run 5-fold ensemble
       c. Apply clip/frame fusion
       d. Average across folds
    3. Apply EMA smoothing
    4. Aggregate overlapping windows
    
    Returns:
        DataFrame with row_id and species probabilities
    """
    
    # Step 1: Load and window
    waveform = load_audio(filepath)
    if waveform is None:
        return pd.DataFrame()
    
    windows = make_windows(waveform)
    if windows is None:
        return pd.DataFrame()
    
    log(f"  Processing {len(windows)} windows from {Path(filepath).name}")
    
    # Step 2: Batch inference
    all_preds = []
    
    for batch_idx in range(0, len(windows), BATCH_SIZE):
        batch = windows[batch_idx : batch_idx + BATCH_SIZE]
        
        try:
            # Extract mel spectrogram
            mel_input = extract_mel_spectrogram(batch)
            
            # Validate shape
            if mel_input.shape[1:] != (1, N_MELS, 313):
                log(f"    ⚠️  Mel shape mismatch: {mel_input.shape}, expected (batch, 1, 256, 313)")
            
            # Run ensemble inference
            clip_preds, frame_preds = ensemble.inference(mel_input)
            
            # Clip/frame fusion (paper Section 3.4)
            fold_fused = []
            for clip_logits, frame_logits in zip(clip_preds, frame_preds):
                # Frame max pooling
                frame_max = frame_logits.max(axis=1)
                
                # Weighted fusion
                fused = CLIP_WEIGHT * clip_logits + FRAME_WEIGHT * frame_max
                fold_fused.append(fused)
            
            # Ensemble averaging (paper Section 3.5)
            ensemble_pred = np.mean(fold_fused, axis=0)
            
            # Logits -> probabilities
            batch_probs = 1.0 / (1.0 + np.exp(-ensemble_pred))
            
            # Clip to [0, 1]
            batch_probs = np.clip(batch_probs, 0.0, 1.0)
            
            all_preds.append(batch_probs)
            
        except Exception as e:
            log(f"    ✗ Batch {batch_idx}: {str(e)[:80]}")
            # Use uniform low probabilities as fallback
            batch_probs = np.ones((batch.shape[0], len(species_cols)), dtype=np.float32) * 0.01
            all_preds.append(batch_probs)
            continue
    
    if not all_preds:
        log(f"  ✗ No predictions generated")
        return pd.DataFrame()
    
    # Concatenate all batch predictions
    all_preds = np.concatenate(all_preds, axis=0)
    
    # Step 3: EMA smoothing (paper Section 3.6)
    smooth = [all_preds[0]]
    for i in range(1, len(all_preds)):
        smooth.append(EMA_ALPHA * all_preds[i] + EMA_BETA * smooth[-1])
    smooth = np.stack(smooth)
    
    # Step 4: Aggregate overlapping windows
    rows = []
    soundscape_id = Path(filepath).stem
    steps_per_row = int(5 / HOP_SEC)  # How many windows per 5-sec output
    
    for idx in range(0, len(smooth), steps_per_row):
        group = smooth[idx : idx + steps_per_row]
        
        # Maximum across overlapping windows
        agg = np.max(group, axis=0)
        
        # Time coordinate
        end_sec = ((idx // steps_per_row + 1) * 5)
        
        # Build row
        row = {"row_id": f"{soundscape_id}_{end_sec}"}
        row.update(dict(zip(species_cols, agg)))
        rows.append(row)
    
    # Cleanup
    del waveform, windows
    cleanup_memory()
    
    log(f"  ✓ Generated {len(rows)} predictions")
    
    return pd.DataFrame(rows)

# =========================================================
# MAIN PIPELINE
# =========================================================

if __name__ == "__main__":
    
    try:
        # [1/5] Read sample submission
        log("\n[1/5] Reading sample submission...")
        sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
        species_cols = [c for c in sample_sub.columns if c != "row_id"]
        log(f"✓ Species: {len(species_cols)}, Expected rows: {len(sample_sub)}")
        
        # [2/5] Load ensemble
        log("\n[2/5] Loading SED ensemble...")
        ensemble = ONNXModelEnsemble(MODEL_DIR, NUM_FOLDS)
        
        # [3/5] Find test soundscapes
        log("\n[3/5] Finding test soundscapes...")
        
        if not TEST_SOUNDSCAPES.exists():
            log(f"✗ Test soundscapes directory not found: {TEST_SOUNDSCAPES}")
            log(f"  Available in {COMP_PATH}:")
            for item in sorted(COMP_PATH.iterdir())[:10]:
                log(f"    - {item.name}")
            sys.exit(1)
        
        # Try multiple audio formats
        soundscapes = list(TEST_SOUNDSCAPES.glob("*.ogg"))
        if not soundscapes:
            soundscapes = list(TEST_SOUNDSCAPES.glob("*.mp3"))
        if not soundscapes:
            soundscapes = list(TEST_SOUNDSCAPES.glob("*.wav"))
        
        soundscapes = sorted(soundscapes)
        log(f"✓ Found {len(soundscapes)} soundscapes")
        
        if len(soundscapes) == 0:
            log(f"✗ No audio files in {TEST_SOUNDSCAPES}")
            log(f"  Contents: {list(TEST_SOUNDSCAPES.iterdir())[:5]}")
            sys.exit(1)
        
        # [4/5] Generate predictions
        log("\n[4/5] Generating predictions...")
        all_rows = []
        
        for i, filepath in enumerate(soundscapes):
            log(f"\n  [{i+1}/{len(soundscapes)}] {filepath.name}")
            df = predict_soundscape(str(filepath), ensemble, species_cols)
            
            if not df.empty:
                all_rows.append(df)
            else:
                log(f"    ⚠️  No rows generated")
        
        if not all_rows:
            log("\n✗ FATAL: No predictions generated from any soundscape")
            sys.exit(1)
        
        submission = pd.concat(all_rows, ignore_index=True)
        
        # Ensure all columns present
        for col in species_cols:
            if col not in submission.columns:
                log(f"  ⚠️  Missing column: {col} (adding zeros)")
                submission[col] = 0.0
        
        # Reorder to match sample submission
        submission = submission[sample_sub.columns]
        
        # Validate predictions
        for col in species_cols:
            submission[col] = submission[col].clip(0.0, 1.0).astype(np.float32)
        
        log(f"\n  Submission shape: {submission.shape}")
        log(f"  Expected rows: {len(sample_sub)}, Got: {len(submission)}")
        log(f"  Prediction range: [{submission.iloc[:, 1:].min().min():.6f}, {submission.iloc[:, 1:].max().max():.6f}]")
        
        # [5/5] Save
        log("\n[5/5] Saving submission...")
        submission_path = OUTPUT_PATH / "submission.csv"
        submission.to_csv(submission_path, index=False)
        log(f"✓ Saved to {submission_path}")
        
        # Summary
        log("\n" + "=" * 70)
        log("SUBMISSION COMPLETE")
        log("=" * 70)
        log(f"Shape: {submission.shape}")
        log(f"Sample rows:\n{submission.head(3).to_string()}")
        log(f"\nValue stats:")
        log(f"  Min: {submission.iloc[:, 1:].values.min():.6f}")
        log(f"  Max: {submission.iloc[:, 1:].values.max():.6f}")
        log(f"  Mean: {submission.iloc[:, 1:].values.mean():.6f}")
        log(f"  Median: {np.median(submission.iloc[:, 1:].values):.6f}")
        
        nan_count = submission.isna().sum().sum()
        if nan_count > 0:
            log(f"  ⚠️  NaN count: {nan_count}")
        else:
            log(f"  ✓ No NaN values")
        
        log("\n✅ SUCCESS - Ready for Kaggle submission!")
        cleanup_memory()
        
    except Exception as e:
        log(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
