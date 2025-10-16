#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stream streaming-optimized OSC → StreamDiffusion viewer
- Attempts to use StreamDiffusion streaming API when available.
- Falls back to minimal-step StableDiffusion pipeline if not.
- Maintains a persistent prepared pipeline; reuses it to avoid re-initialization.
- Provides a streaming generation mode that emits frames at target FPS (best-effort).
"""

import sys
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
)
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal

from pythonosc import dispatcher
from pythonosc import osc_server

# ---- Config ----
OSC_IP = "127.0.0.1"
OSC_PORT = 8001
OSC_ADDRESS = "/prompt"

SHARED_FOLDER = Path("./generated_images")
SHARED_FOLDER.mkdir(parents=True, exist_ok=True)

# Model & mode defaults
MODEL_ID = "SG161222/Realistic-Vision-v3.0"# "stabilityai/sd-turbo"   # small / fast model if available
USE_STREAMDIFFUSION = True          # try to use streamdiffusion
USE_TENSORRT_IF_AVAILABLE = True
DEBUG = True

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# Streaming target (best-effort)
STREAM_TARGET_FPS = 30             # goal; actual depends on hardware
STREAM_DEFAULT_FRAMES = 300        # default number of frames to stream per prompt (can be interrupted)
STREAM_FRAME_INTERVAL = 1.0 / STREAM_TARGET_FPS

# Low-latency minimal settings
LOW_NUM_INFERENCE_STEPS = 1
LOW_GUIDANCE = 0.0
LOW_T_INDEX = [0]

# High-quality settings (used for warmup/test only)
HQ_NUM_INFERENCE_STEPS = 20
HQ_GUIDANCE = 7.5
HQ_T_INDEX = [0, 12, 25, 37, 45]

SAVE_IMAGES = True

# ---- Try imports ----
try:
    import torch
except Exception:
    torch = None

try:
    # streamdiffusion library (may be installed by user)
    import streamdiffusion
    from streamdiffusion import StreamDiffusion
    try:
        from streamdiffusion.image_utils import postprocess_image
    except Exception:
        postprocess_image = None
    STREAMDIFFUSION_AVAILABLE = True
except Exception:
    StreamDiffusion = None
    postprocess_image = None
    STREAMDIFFUSION_AVAILABLE = False

# diffusers fallback
try:
    from diffusers import StableDiffusionPipeline, DDIMScheduler, EulerAncestralDiscreteScheduler
    DIFFUSERS_AVAILABLE = True
except Exception:
    StableDiffusionPipeline = None
    DIFFUSERS_AVAILABLE = False

# ---- GUI helper classes (simplified) ----
class SignalEmitter(QObject):
    image_ready = pyqtSignal(str)       # filepath
    status_update = pyqtSignal(str)
    prompt_received = pyqtSignal(str)

class AfterimageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")
    def add_image_from_path(self, path):
        pm = QPixmap(path)
        if not pm.isNull():
            self.setPixmap(pm.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

# Minimal viewer with mode button
from PyQt6.QtGui import QPixmap
class ViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OSC → StreamDiffusion (streaming-optimized)")
        self.resize(900, 700)
        self.signal = SignalEmitter()

        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        self.image_label = AfterimageLabel()
        layout.addWidget(self.image_label)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color:white;background:rgba(0,0,0,0.6);padding:6px;")
        layout.addWidget(self.status_label)

        self.mode_button = QPushButton("Stream Mode: AUTO")
        layout.addWidget(self.mode_button)
        self.mode_button.setEnabled(False)

        # connect signals
        self.signal.image_ready.connect(self.on_image_ready)
        self.signal.status_update.connect(self.on_status)

    def on_status(self, s):
        self.status_label.setText(s)
        if DEBUG:
            print("[STATUS]", s)

    def on_image_ready(self, path):
        self.image_label.add_image_from_path(path)

# ---- Generator class: tries to use streaming if available ----
class StreamingGenerator:
    def __init__(self):
        self.pipe = None
        self.is_stream = False
        self.device = None
        self.prepared_prompt = None
        self.pipeline_lock = threading.Lock()
        self.streaming_thread = None
        self.stop_stream_flag = threading.Event()
        self.last_generated_path = None

    def initialize(self):
        """Initialize pipeline depending on availability."""
        if USE_STREAMDIFFUSION and STREAMDIFFUSION_AVAILABLE and torch is not None:
            return self._init_streamdiffusion()
        elif DIFFUSERS_AVAILABLE and torch is not None:
            return self._init_diffusers_fallback()
        else:
            print("[ERROR] No suitable backend available (torch + diffusers or streamdiffusion required).")
            return False

    def _init_streamdiffusion(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[INIT] StreamDiffusion -> device={self.device}")
            if self.device.type != "cuda":
                print("[WARN] CUDA not available; streamdiffusion performance will be poor. Falling back to diffusers pipeline.")
                return self._init_diffusers_fallback()

            # Load a diffusers pipeline first (StreamDiffusion wraps it)
            from diffusers import StableDiffusionPipeline, AutoencoderTiny
            base_pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, safety_checker=None).to(self.device)
            # instantiate StreamDiffusion wrapper
            self.pipe = StreamDiffusion(pipe=base_pipe, t_index_list=LOW_T_INDEX, torch_dtype=torch.float16, cfg_type="none", width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
            # try tiny VAE
            try:
                self.pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=self.device, dtype=torch.float16)
                print("[INIT] AutoencoderTiny loaded.")
            except Exception as e:
                print("[WARN] AutoencoderTiny not loaded:", e)
            # enable xformers if available
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

            # detect if streaming API exists
            self.is_stream = hasattr(self.pipe, "stream") or hasattr(self.pipe, "stream_inference") or hasattr(self.pipe, "__call__") and hasattr(self.pipe, "prepare")
            print(f"[INIT] Stream API available? {self.is_stream}")

            # Warmup prepare with small prompt (use low settings)
            try:
                self.pipe.prepare(prompt="warmup", num_inference_steps=LOW_NUM_INFERENCE_STEPS, guidance_scale=LOW_GUIDANCE)
                # if pipe has call that returns quickly, run one
                try:
                    out = self.pipe()
                    if DEBUG:
                        print("[INIT] warmup call returned:", type(out))
                except Exception:
                    pass
            except Exception as e:
                print("[WARN] prepare/warmup error:", e)

            return True
        except Exception as e:
            print("[ERROR] StreamDiffusion init failed:", e)
            traceback.print_exc()
            return self._init_diffusers_fallback()

    def _init_diffusers_fallback(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[INIT] Diffusers fallback -> device={self.device}")
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, safety_checker=None).to(self.device)
            # small scheduler enable
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass
            self.is_stream = False
            return True
        except Exception as e:
            print("[ERROR] diffusers init failed:", e)
            traceback.print_exc()
            return False

    def _save_image_and_emit(self, pil_img, shared_folder, signal_emitter, frame_index=None):
        # Save image to file and emit path via signal_emitter
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"stream_{timestamp}" + (f"_f{frame_index}" if frame_index is not None else "") + ".png"
        path = shared_folder / name
        try:
            pil_img.save(path)
            self.last_generated_path = path
            if signal_emitter:
                signal_emitter.image_ready.emit(str(path))
            if DEBUG:
                print(f"[FRAME] Saved {path}")
        except Exception as e:
            print("[ERROR] save image failed:", e)

    def prepare_stream_for_prompt(self, prompt, num_inference_steps=LOW_NUM_INFERENCE_STEPS, guidance=LOW_GUIDANCE, t_index_list=None):
        """Prepare pipeline for streaming a given prompt. This should be fast (no reloading)."""
        with self.pipeline_lock:
            if hasattr(self.pipe, "prepare"):
                try:
                    # use provided t_index_list or default minimal
                    t_list = t_index_list if t_index_list is not None else LOW_T_INDEX
                    if hasattr(self.pipe, "t_index_list"):
                        # try to set attr (some versions use this)
                        try:
                            self.pipe.t_index_list = t_list
                        except Exception:
                            pass
                    # call prepare
                    self.pipe.prepare(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance)
                    self.prepared_prompt = prompt
                    if DEBUG:
                        print(f"[PREPARE] prepared prompt (steps={num_inference_steps}, t_index={t_list})")
                    return True
                except Exception as e:
                    print("[WARN] prepare() failed:", e)
                    traceback.print_exc()
                    return False
            else:
                # no prepare available; fallback
                self.prepared_prompt = prompt
                return True

    def _stream_worker(self, shared_folder, signal_emitter, frames=STREAM_DEFAULT_FRAMES, interval=STREAM_FRAME_INTERVAL):
        """
        Worker that calls the stream API and emits frames.
        This tries multiple fallback strategies depending on actual API:
        1) pipe.stream(...) if available and returns an iterable of images/arrays
        2) repeated pipe() calls after prepare() (some StreamDiffusion variants produce quick incremental outputs)
        3) for diffusers fallback: generate images in a tight loop with minimal steps
        """
        if DEBUG:
            print("[WORKER] stream worker started; is_stream=", self.is_stream)
        frame_idx = 0
        self.stop_stream_flag.clear()
        start_time = time.time()
        while not self.stop_stream_flag.is_set() and frame_idx < frames:
            try:
                t0 = time.time()
                pil_img = None

                # Strategy A: direct stream() method that yields / returns incremental images
                if hasattr(self.pipe, "stream"):
                    try:
                        result = self.pipe.stream()  # many implementations: stream() yields images or returns a frame
                        # result might be iterable, or a single item
                        if hasattr(result, "__iter__") and not isinstance(result, (Image.Image, np.ndarray, bytes, str)):
                            # iterate quickly but pick first/last depending on implementation
                            # attempt to get first available image
                            got = None
                            for r in result:
                                got = r
                                break
                            out = got
                        else:
                            out = result
                        # convert to PIL robustly
                        pil_img = self._to_pil(out)
                    except Exception as e:
                        # stream() call failed; fallback to call()
                        if DEBUG:
                            print("[WORKER] pipe.stream() error, fallback to pipe()", e)
                        try:
                            out = self.pipe()
                            pil_img = self._to_pil(out)
                        except Exception as e2:
                            if DEBUG:
                                print("[WORKER] fallback pipe() error:", e2)
                            pil_img = None

                # Strategy B: if no stream(), use call() which may be very fast in streamdiffusion wrapper
                elif hasattr(self.pipe, "__call__") and hasattr(self.pipe, "prepare"):
                    # If prepared_prompt differs, re-prepare with low-latency settings
                    if self.prepared_prompt is None:
                        # try to prepare with prompt
                        self.prepare_stream_for_prompt(self.prepared_prompt or " ", num_inference_steps=LOW_NUM_INFERENCE_STEPS, guidance=LOW_GUIDANCE, t_index_list=LOW_T_INDEX)
                    try:
                        out = self.pipe()  # many streamdiffusion implementations make this incremental when prepared
                        pil_img = self._to_pil(out)
                    except Exception as e:
                        if DEBUG:
                            print("[WORKER] pipe() error:", e)
                        pil_img = None

                # Strategy C: diffusers fallback - generate but keep it minimal
                else:
                    # Use diffusers pipeline single-step minimal generation
                    try:
                        # seeded generator to reduce jitter if needed
                        generator_obj = torch.Generator(device=self.device).manual_seed(int(time.time() * 1000) % (2**31 - 1))
                        out = self.pipe(
                            self.prepared_prompt or "a photo",
                            num_inference_steps=max(1, LOW_NUM_INFERENCE_STEPS),
                            guidance_scale=LOW_GUIDANCE,
                            height=IMAGE_HEIGHT,
                            width=IMAGE_WIDTH,
                            generator=generator_obj,
                        )
                        # out.images[0] expected
                        if hasattr(out, "images"):
                            pil_img = out.images[0]
                        else:
                            pil_img = None
                    except Exception as e:
                        if DEBUG:
                            print("[WORKER] diffusers loop error:", e)
                        pil_img = None

                if pil_img is None:
                    # if we failed to get an image, produce dummy and continue
                    if DEBUG:
                        print("[WORKER] no image produced for frame", frame_idx)
                    time.sleep(0.01)
                else:
                    # save & emit
                    self._save_image_and_emit(pil_img, SHARED_FOLDER, signal_emitter, frame_index=frame_idx)
                frame_idx += 1

                # throttle to target FPS (best-effort)
                elapsed = time.time() - t0
                to_wait = interval - elapsed
                if to_wait > 0:
                    time.sleep(to_wait)

            except Exception as e:
                print("[WORKER] unexpected error:", e)
                traceback.print_exc()
                time.sleep(0.05)
        if DEBUG:
            print("[WORKER] exiting stream worker; frames produced:", frame_idx, "duration(s):", time.time() - start_time)

    def start_stream(self, prompt, shared_folder, signal_emitter, frames=STREAM_DEFAULT_FRAMES, fps=STREAM_TARGET_FPS):
        """Public method: prepare and start streaming frames. Runs in background thread and returns immediately."""
        if not self.pipe:
            raise RuntimeError("Pipeline not initialized")
        # prepare using minimal low-latency settings
        self.prepare_stream_for_prompt(prompt, num_inference_steps=LOW_NUM_INFERENCE_STEPS, guidance=LOW_GUIDANCE, t_index_list=LOW_T_INDEX)
        # stop any existing stream
        self.stop_stream()
        self.stop_stream_flag.clear()
        interval = 1.0 / max(1, fps)
        self.streaming_thread = threading.Thread(target=self._stream_worker, args=(shared_folder, signal_emitter, frames, interval), daemon=True)
        self.streaming_thread.start()
        if DEBUG:
            print("[STREAM] started streaming thread")

    def stop_stream(self):
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.stop_stream_flag.set()
            self.streaming_thread.join(timeout=2.0)
            if DEBUG:
                print("[STREAM] stop requested and thread joined")
        self.streaming_thread = None

    def generate_one(self, prompt, shared_folder, signal_emitter):
        """Synchronous single-image generation used when streaming not desired."""
        with self.pipeline_lock:
            if self.is_stream:
                # prepare then call once
                self.prepare_stream_for_prompt(prompt, num_inference_steps=LOW_NUM_INFERENCE_STEPS, guidance=LOW_GUIDANCE, t_index_list=LOW_T_INDEX)
                try:
                    out = self.pipe()
                    pil = self._to_pil(out)
                    self._save_image_and_emit(pil, shared_folder, signal_emitter)
                    return pil
                except Exception as e:
                    print("[GEN_ONE] stream call failed:", e)
                    traceback.print_exc()
                    return self._dummy_image(prompt)
            else:
                # diffusers pipeline
                try:
                    generator_obj = torch.Generator(device=self.device).manual_seed(int(time.time() * 1000) % (2**31 - 1))
                    res = self.pipe(prompt, num_inference_steps=20, guidance_scale=7.5, height=IMAGE_HEIGHT, width=IMAGE_WIDTH, generator=generator_obj)
                    img = res.images[0]
                    self._save_image_and_emit(img, shared_folder, signal_emitter)
                    return img
                except Exception as e:
                    print("[GEN_ONE] diffusers gen failed:", e)
                    traceback.print_exc()
                    return self._dummy_image(prompt)

    def _to_pil(self, out):
        """Robust conversion of outputs to PIL.Image."""
        try:
            if out is None:
                return None
            # if streamdiffusion provides numpy / tensor / PIL
            if isinstance(out, Image.Image):
                return out
            import numpy as _np
            if isinstance(out, _np.ndarray):
                arr = out
            else:
                # torch tensor?
                try:
                    import torch as _torch
                    if isinstance(out, _torch.Tensor):
                        arr = out.detach().cpu().numpy()
                    else:
                        # maybe a dict with 'image' key
                        if isinstance(out, dict):
                            for k in ("image", "images", "output"):
                                if k in out:
                                    return self._to_pil(out[k])
                        # fallback: try to convert to array
                        arr = _np.array(out)
                except Exception:
                    arr = _np.array(out)
            # squeeze
            while arr.ndim > 3:
                arr = arr.squeeze(0)
            # normalize floats in -1..1 or 0..1
            if arr.dtype.kind == 'f':
                if arr.min() < 0:
                    arr = (arr + 1.0) / 2.0
                arr = (arr * 255.0).clip(0, 255).astype("uint8")
            # channel first -> HWC
            if arr.ndim == 3 and arr.shape[0] in (1,3,4):
                arr = np.transpose(arr, (1,2,0))
            # gray -> rgb
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            return Image.fromarray(arr.astype("uint8"), mode="RGB")
        except Exception as e:
            print("[TO_PIL] conversion failed:", e)
            traceback.print_exc()
            return self._dummy_image("convert_failed")

    def _dummy_image(self, prompt):
        # Simple placeholder
        img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (64, 64, 96))
        return img

# ---- Instantiate generator (global) ----
generator = StreamingGenerator()
init_ok = generator.initialize()
if not init_ok:
    print("[WARN] pipeline initialization returned False — check logs. Continuing, but generation may fail.")

# ---- generate_image wrapper used by OSC handler ----
def generate_image(prompt, shared_folder, signal_emitter, stream_mode=True, stream_frames=STREAM_DEFAULT_FRAMES, target_fps=STREAM_TARGET_FPS):
    """
    If stream_mode True and stream API available, starts streaming frames for 'stream_frames'.
    Otherwise, generates a single image.
    """
    try:
        signal_emitter.status_update.emit(f"生成開始: {prompt[:60]}...")
        if stream_mode and generator.is_stream:
            if DEBUG:
                print("[TASK] starting stream for prompt:", prompt)
            # start stream in background and return immediately
            generator.start_stream(prompt, shared_folder, signal_emitter, frames=stream_frames, fps=target_fps)
        else:
            # synchronous single image generation (fast fallback)
            img = generator.generate_one(prompt, shared_folder, signal_emitter)
            # image emitted by generator
            if DEBUG:
                print("[TASK] generate_one finished")
    except Exception as e:
        signal_emitter.status_update.emit(f"生成エラー: {e}")
        print("[ERROR] generate_image exception:", e)
        traceback.print_exc()

# ---- OSC handler ----
def osc_handler(address, *args, shared_folder=None, signal_emitter=None):
    if len(args) >= 1:
        prompt = str(args[0])
    else:
        print(f"[ERROR] invalid OSC args: {args}")
        return
    print(f"[OSC] {address} -> {prompt}")
    if signal_emitter:
        signal_emitter.status_update.emit(f"OSC受信: {prompt[:50]}")
        signal_emitter.prompt_received.emit(prompt)
    # stop previous stream if any and start new
    generator.stop_stream()
    # start generation in background thread (non-blocking)
    t = threading.Thread(target=generate_image, args=(prompt, SHARED_FOLDER, signal_emitter, True, STREAM_DEFAULT_FRAMES, STREAM_TARGET_FPS), daemon=True)
    t.start()

# ---- OSC server starter ----
def start_osc_server(signal_emitter):
    disp = dispatcher.Dispatcher()
    disp.map(OSC_ADDRESS, lambda addr, *args: osc_handler(addr, *args, shared_folder=SHARED_FOLDER, signal_emitter=signal_emitter))
    server = osc_server.ThreadingOSCUDPServer((OSC_IP, OSC_PORT), disp)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[OSC] listening on {OSC_IP}:{OSC_PORT} {OSC_ADDRESS}")
    return server

# ---- Main / UI wiring ----
def main():
    app = QApplication(sys.argv)
    w = ViewerWindow()

    # wire signals between main window and generator wrapper
    start_osc_server(w.signal)

    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
