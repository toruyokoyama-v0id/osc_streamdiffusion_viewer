#!/usr/bin/env python3

-- coding: utf-8 --
import sys
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage

from pythonosc import dispatcher
from pythonosc import osc_server

============== 設定 ==============
OSC_IP = "127.0.0.1"
OSC_PORT = 8001
OSC_ADDRESS = "/prompt"

軽量＆高速向けモデル（推奨）
MODEL_ID = "stabilityai/sd-turbo"

画像サイズ
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

ストリーミング（疑似）設定
STREAM_TARGET_FPS = 8 # 実機GPUに合わせて調整（8〜12くらいが現実的）
STREAM_DEFAULT_FRAMES = 300
STREAM_FRAME_INTERVAL = 1.0 / STREAM_TARGET_FPS

単発/ループ時の生成設定（単色回避）
LOW_NUM_INFERENCE_STEPS = 2 # sd-turbo の推奨は 1〜2
LOW_GUIDANCE = 1.5 # 0.0 だと単色になりやすい環境があるため >0 に

保存設定（毎フレームは保存しない）
SAVE_FOLDER = Path("./generated_images")
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

DEBUG = True

============== 依存ロード（torch/diffusers） ==============
try:
import torch
except Exception:
torch = None

try:
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
DIFFUSERS_AVAILABLE = True
except Exception:
StableDiffusionPipeline = None
EulerAncestralDiscreteScheduler = None
DIFFUSERS_AVAILABLE = False

============== シグナル ==============
class SignalEmitter(QObject):
image_ready_qimage = pyqtSignal(object) # QImage
status_update = pyqtSignal(str)
prompt_received = pyqtSignal(str)

============== ビューア ==============
class ViewerWindow(QMainWindow):
def init(self):
super().init()
self.setWindowTitle("OSC → 画像ストリーミング（diffusers）")
self.resize(900, 700)


    central = QWidget()  
    layout = QVBoxLayout(central)  
    self.setCentralWidget(central)  

    self.image_label = QLabel()  
    self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  
    self.image_label.setStyleSheet("background-color: black;")  
    layout.addWidget(self.image_label)  

    self.status_label = QLabel("準備中...")  
    self.status_label.setStyleSheet("color:white;background:rgba(0,0,0,0.6);padding:6px;")  
    layout.addWidget(self.status_label)  

    self.save_button = QPushButton("最後のフレームを保存")  
    self.save_button.clicked.connect(self.save_last_frame)  
    layout.addWidget(self.save_button)  

    self.signal = SignalEmitter()  
    self.signal.image_ready_qimage.connect(self.on_image_ready_qimage)  
    self.signal.status_update.connect(self.on_status)  

    self.last_frame_qimage = None  

def on_status(self, s: str):  
    self.status_label.setText(s)  
    if DEBUG:  
        print("[STATUS]", s)  

def on_image_ready_qimage(self, qimg: QImage):  
    if qimg is None or qimg.isNull():  
        return  
    self.last_frame_qimage = qimg  
    pm = QPixmap.fromImage(qimg)  
    if not pm.isNull():  
        self.image_label.setPixmap(  
            pm.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)  
        )  

def save_last_frame(self):  
    if self.last_frame_qimage is None or self.last_frame_qimage.isNull():  
        self.on_status("保存できるフレームがありません")  
        return  
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  
    path = SAVE_FOLDER / f"frame_{ts}.png"  
    ok = self.last_frame_qimage.save(str(path))  
    if ok:  
        self.on_status(f"保存しました: {path}")  
    else:  
        self.on_status("保存に失敗しました")  
============== 生成器（diffusersのみ） ==============
class DiffusersStreamer:
def init(self):
self.pipe = None
self.device = None
self.is_initialized = False


    self.pipeline_lock = threading.Lock()  
    self.streaming_thread = None  
    self.stop_stream_flag = threading.Event()  
    self.prepared_prompt = None  

def initialize(self) -> bool:  
    if torch is None or not DIFFUSERS_AVAILABLE:  
        print("[ERROR] torch または diffusers が見つかりません")  
        return False  

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"[INIT] device={self.device}")  

    try:  
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32  
        self.pipe = StableDiffusionPipeline.from_pretrained(  
            MODEL_ID, torch_dtype=dtype, safety_checker=None  
        ).to(self.device)  

        # スケジューラ（任意）  
        try:  
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)  
            print("[INIT] EulerAncestralDiscreteScheduler 設定")  
        except Exception:  
            pass  

        if self.device.type == "cuda":  
            try:  
                self.pipe.enable_xformers_memory_efficient_attention()  
                print("[INIT] xformers enabled")  
            except Exception:  
                pass  

        # ウォームアップ（数百ms〜数秒）  
        _ = self.pipe(  
            "warmup test photo",  
            num_inference_steps=max(1, LOW_NUM_INFERENCE_STEPS),  
            guidance_scale=max(0.5, LOW_GUIDANCE),  
            height=IMAGE_HEIGHT,  
            width=IMAGE_WIDTH,  
        ).images[0]  

        self.is_initialized = True  
        print("[INIT] diffusers 初期化完了")  
        return True  
    except Exception as e:  
        print("[ERROR] diffusers 初期化失敗:", e)  
        traceback.print_exc()  
        return False  

def stop_stream(self):  
    if self.streaming_thread and self.streaming_thread.is_alive():  
        self.stop_stream_flag.set()  
        self.streaming_thread.join(timeout=2.0)  
        if DEBUG:  
            print("[STREAM] stop requested & joined")  
    self.streaming_thread = None  

def start_stream(self, prompt: str, signal: SignalEmitter, frames=STREAM_DEFAULT_FRAMES, fps=STREAM_TARGET_FPS):  
    if not self.is_initialized or self.pipe is None:  
        raise RuntimeError("Pipeline not initialized")  

    self.stop_stream()  
    self.stop_stream_flag.clear()  
    self.prepared_prompt = prompt  

    interval = max(1.0 / max(1, int(fps)), 0.001)  
    self.streaming_thread = threading.Thread(  
        target=self._stream_worker,  
        args=(signal, frames, interval),  
        daemon=True,  
    )  
    self.streaming_thread.start()  
    if DEBUG:  
        print("[STREAM] streaming thread started")  

def generate_one(self, prompt: str) -> Image.Image:  
    if not self.is_initialized or self.pipe is None:  
        raise RuntimeError("Pipeline not initialized")  
    with self.pipeline_lock:  
        generator_obj = torch.Generator(device=self.device)  
        seed = int(time.time() * 1000) % (2**31 - 1)  
        generator_obj.manual_seed(seed)  
        res = self.pipe(  
            prompt,  
            num_inference_steps=max(2, LOW_NUM_INFERENCE_STEPS),  
            guidance_scale=max(1.0, LOW_GUIDANCE),  
            height=IMAGE_HEIGHT,  
            width=IMAGE_WIDTH,  
            generator=generator_obj,  
        )  
        return res.images[0]  

def _stream_worker(self, signal: SignalEmitter, frames: int, interval: float):  
    frame_idx = 0  
    start_time = time.time()  
    if DEBUG:  
        print("[WORKER] stream worker start")  

    while not self.stop_stream_flag.is_set() and frame_idx < frames:  
        t0 = time.time()  
        try:  
            pil_img = self.generate_one(self.prepared_prompt or "high quality photo")  
            if pil_img is not None:  
                qimg = self._pil_to_qimage(pil_img)  
                signal.image_ready_qimage.emit(qimg)  
            else:  
                time.sleep(0.005)  
            frame_idx += 1  

            # FPS 調整  
            elapsed = time.time() - t0  
            to_wait = interval - elapsed  
            if to_wait > 0:  
                time.sleep(to_wait)  
        except Exception as e:  
            print("[WORKER] error:", e)  
            traceback.print_exc()  
            time.sleep(0.05)  

    if DEBUG:  
        print("[WORKER] exit; frames:", frame_idx, "dur:", time.time() - start_time)  

@staticmethod  
def _pil_to_qimage(pil: Image.Image) -> QImage:  
    if pil is None:  
        return QImage()  
    try:  
        pil = pil.convert("RGBA")  
        arr = np.array(pil)  # H x W x 4  
        h, w, ch = arr.shape  
        bytes_per_line = ch * w  
        qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)  
        return qimg.copy()  # 独立コピー  
    except Exception:  
        # 失敗時  
        dummy = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (64, 64, 96)).convert("RGBA")  
        arr = np.array(dummy)  
        h, w, ch = arr.shape  
        return QImage(arr.data, w, h, ch * w, QImage.Format.Format_RGBA8888).copy()  
============== グローバル生成器 ==============
generator = DiffusersStreamer()

============== タスクラッパ ==============
def generate_image(prompt: str, signal_emitter: SignalEmitter, stream=True, frames=STREAM_DEFAULT_FRAMES, fps=STREAM_TARGET_FPS):
try:
signal_emitter.status_update.emit(f"生成開始: {prompt[:60]}...")
if stream:
generator.start_stream(prompt, signal_emitter, frames=frames, fps=fps)
else:
pil = generator.generate_one(prompt)
qimg = generator._pil_to_qimage(pil)
signal_emitter.image_ready_qimage.emit(qimg)
signal_emitter.status_update.emit("単発生成完了")
except Exception as e:
signal_emitter.status_update.emit(f"生成エラー: {e}")
print("[ERROR] generate_image:", e)
traceback.print_exc()

============== OSC ==============
def osc_handler(address, *args, signal_emitter: SignalEmitter = None):
if len(args) >= 1:
prompt = str(args[0])
else:
print(f"[ERROR] invalid OSC args: {args}")
return
print(f"[OSC] {address} -> {prompt}")
if signal_emitter:
signal_emitter.status_update.emit(f"OSC受信: {prompt[:50]}")
signal_emitter.prompt_received.emit(prompt)


# 前回ストリーム停止 → 新規開始  
generator.stop_stream()  
t = threading.Thread(  
    target=generate_image,  
    args=(prompt, signal_emitter, True, STREAM_DEFAULT_FRAMES, STREAM_TARGET_FPS),  
    daemon=True,  
)  
t.start()  
def start_osc_server(signal_emitter: SignalEmitter):
disp = dispatcher.Dispatcher()
disp.map(OSC_ADDRESS, lambda addr, *args: osc_handler(addr, *args, signal_emitter=signal_emitter))
server = osc_server.ThreadingOSCUDPServer((OSC_IP, OSC_PORT), disp)
t = threading.Thread(target=server.serve_forever, daemon=True)
t.start()
print(f"[OSC] listening on {OSC_IP}:{OSC_PORT} {OSC_ADDRESS}")
return server

============== Main ==============
def main():
app = QApplication(sys.argv)
w = ViewerWindow()


ok = generator.initialize()  
if not ok:  
    w.signal.status_update.emit("初期化失敗（torch/diffusers/モデルを確認）")  
else:  
    w.signal.status_update.emit("初期化完了。OSCで /prompt に文字列を送ってください。")  

start_osc_server(w.signal)  

w.show()  
sys.exit(app.exec())  
if name == "main":
main()
