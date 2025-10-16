def osc_handler(address, *args, shared_folder=None, signal_emitter=None):
    if len(args) >= 1:
        prompt = str(args[0])
    else:
        print(f"[ERROR] OSCメッセージの形式が不正: {args}")
        return
    
    print(f"[OSC受信] {address}: {prompt}")
    signal_emitter.status_update.emit(f"OSC受信: {prompt[:30]}...")
    signal_emitter.prompt_received.emit(prompt)
    
    thread = threading.Thread(
        target=generate_image,
        args=(prompt, shared_folder, signal_emitter),
        daemon=True,
    )
    thread.start()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import threading
import time
from pathlib import Path
from datetime import datetime
from queue import Queue
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QTextEdit, QVBoxLayout,
    QWidget, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal, QObject, pyqtProperty
from PyQt6.QtGui import QPixmap, QPainter, QImage, QFont
from PIL import Image

from pythonosc import dispatcher
from pythonosc import osc_server

# StreamDiffusion関連
try:
    import torch
    from streamdiffusion import StreamDiffusion
    from streamdiffusion.image_utils import postprocess_image
    STREAMDIFFUSION_AVAILABLE = True
except ImportError:
    STREAMDIFFUSION_AVAILABLE = False
    print("[WARNING] StreamDiffusionがインストールされていません")


# ============================================
# 設定（グローバル変数でモード切替可能に）
# ============================================
OSC_IP = "127.0.0.1"
OSC_PORT = 8001
OSC_ADDRESS = "/prompt"

# モデル選択
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # ★ 標準モデルに変更
# MODEL_ID = "KBlueLeaf/kohaku-v2.1"  # アニメ風（LCM-LoRA必須）
# MODEL_ID = "stabilityai/sd-turbo"  # 高速だが品質低い

USE_LCM_LORA = False  # ★ まずLCM-LoRAなしでテスト

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# モード切替用のグローバル変数
CURRENT_MODE = "high"  # "high" or "low"
T_INDEX_LIST_HIGH = [0, 12, 25, 37, 45]  # 高品質モード（5点サンプリング）
T_INDEX_LIST_LOW = [0, 24, 45]  # 低遅延モード（3点サンプリング）
T_INDEX_LIST = T_INDEX_LIST_HIGH  # デフォルトは高品質

CFG_TYPE = "none"
NUM_INFERENCE_STEPS = 50  # 固定: StreamDiffusion推奨値
GUIDANCE_SCALE = 0.0

SHARED_FOLDER = Path("./generated_images")

# 残像エフェクト設定
AFTERIMAGE_DURATION = 15000
MAX_AFTERIMAGES = 15
AFTERIMAGE_START_OPACITY = 0.9

# タイプライターエフェクト設定
ENABLE_TYPEWRITER = True
TYPEWRITER_SPEED = 50

# 画像保存設定
SAVE_IMAGES = True

# デバッグモード
DEBUG_MODE = True
USE_DUMMY_MODE = False
USE_STREAMDIFFUSION = False  # ★ まず通常パイプラインでテスト

# マルチモニター設定
USE_FULLSCREEN = False
IMAGE_DISPLAY_MONITOR = 0
PROMPT_DISPLAY_MONITOR = 1


# ============================================
# 残像レイヤークラス
# ============================================
class AfterimageLayer:
    def __init__(self, pixmap, start_time):
        self.pixmap = pixmap
        self.start_time = start_time
        self.duration = AFTERIMAGE_DURATION
        self.start_opacity = AFTERIMAGE_START_OPACITY
    
    def get_opacity(self, current_time):
        elapsed = current_time - self.start_time
        if elapsed >= self.duration:
            return 0.0
        progress = elapsed / self.duration
        opacity = self.start_opacity * (1.0 - (progress ** 3))
        return max(0.0, opacity)
    
    def is_expired(self, current_time):
        elapsed = current_time - self.start_time
        return elapsed >= self.duration


class AfterimageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setAutoFillBackground(True)
        self.afterimage_layers = []
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_afterimages)
        self.update_timer.start(16)
    
    def add_image(self, pixmap):
        current_time = time.time() * 1000
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        layer = AfterimageLayer(scaled_pixmap, current_time)
        self.afterimage_layers.append(layer)
        if len(self.afterimage_layers) > MAX_AFTERIMAGES:
            self.afterimage_layers.pop(0)
        self.update()
    
    def update_afterimages(self):
        current_time = time.time() * 1000
        self.afterimage_layers = [
            layer for layer in self.afterimage_layers
            if not layer.is_expired(current_time)
        ]
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        current_time = time.time() * 1000
        for layer in self.afterimage_layers:
            opacity = layer.get_opacity(current_time)
            if opacity > 0:
                painter.setOpacity(opacity)
                pixmap = layer.pixmap
                x = (self.width() - pixmap.width()) // 2
                y = (self.height() - pixmap.height()) // 2
                painter.drawPixmap(x, y, pixmap)
        painter.end()


class SignalEmitter(QObject):
    image_ready = pyqtSignal(str)
    status_update = pyqtSignal(str)
    prompt_received = pyqtSignal(str)
    mode_changed = pyqtSignal(str)


class PromptWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("受信プロンプト")
        self.setGeometry(920, 100, 500, 600)
        self.setStyleSheet("background-color: black;")
        
        if USE_FULLSCREEN:
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: black;")
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.typewriter_label = QLabel("")
        self.typewriter_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 32px;
                font-family: 'Courier New', monospace;
                background-color: black;
                padding: 20px;
            }
        """)
        self.typewriter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.typewriter_label.setWordWrap(True)
        layout.addWidget(self.typewriter_label)
        
        self.current_text = ""
        self.target_text = ""
        self.char_index = 0
        self.typewriter_timer = QTimer()
        self.typewriter_timer.timeout.connect(self.type_next_char)
        self.typewriter_speed = TYPEWRITER_SPEED
        
    def add_prompt(self, prompt):
        if ENABLE_TYPEWRITER:
            self.typewriter_timer.stop()
            self.target_text = prompt
            self.char_index = 0
            self.current_text = ""
            self.typewriter_timer.start(self.typewriter_speed)
        else:
            self.typewriter_label.setText(prompt)
    
    def type_next_char(self):
        if self.char_index < len(self.target_text):
            self.current_text += self.target_text[self.char_index]
            self.typewriter_label.setText(self.current_text)
            self.char_index += 1
        else:
            self.typewriter_timer.stop()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.close()
        else:
            super().keyPressEvent(event)


class ImageViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StreamDiffusion Viewer - モード切替対応")
        self.setGeometry(100, 100, 800, 700)
        self.setStyleSheet("background-color: black;")
        
        if USE_FULLSCREEN:
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = AfterimageLabel()
        main_layout.addWidget(self.image_label)
        
        self.status_label = QLabel("待機中...", self)
        self.status_label.setStyleSheet("""
            color: white; 
            background-color: rgba(0, 0, 0, 180);
            padding: 10px;
            font-size: 14px;
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.status_label.setWordWrap(True)
        
        self.mode_button = QPushButton("現在: 高品質モード (5サンプル) | クリックで低遅延モードへ")
        self.mode_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                padding: 12px 24px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.mode_button.clicked.connect(self.toggle_mode)
        main_layout.addWidget(self.mode_button, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.signal_emitter = SignalEmitter()
        self.signal_emitter.image_ready.connect(self.on_image_ready)
        self.signal_emitter.status_update.connect(self.on_status_update)
        self.signal_emitter.mode_changed.connect(self.on_mode_changed)
        
        self.prompt_window = None
        self.current_mode = "high"
        
    def toggle_mode(self):
        global CURRENT_MODE, T_INDEX_LIST
        
        if self.current_mode == "high":
            self.current_mode = "low"
            CURRENT_MODE = "low"
            T_INDEX_LIST = T_INDEX_LIST_LOW.copy()
            self.mode_button.setText("現在: 低遅延モード (3サンプル) | クリックで高品質モードへ")
            self.mode_button.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    border-radius: 8px;
                    padding: 12px 24px;
                    margin: 10px;
                }
                QPushButton:hover {
                    background-color: #F57C00;
                }
                QPushButton:pressed {
                    background-color: #E65100;
                }
            """)
            status_msg = f"モード切替: {self.current_mode} (T_INDEX={T_INDEX_LIST})"
        else:
            self.current_mode = "high"
            CURRENT_MODE = "high"
            T_INDEX_LIST = T_INDEX_LIST_HIGH.copy()
            self.mode_button.setText("現在: 高品質モード (5サンプル) | クリックで低遅延モードへ")
            self.mode_button.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    border-radius: 8px;
                    padding: 12px 24px;
                    margin: 10px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
                QPushButton:pressed {
                    background-color: #0D47A1;
                }
            """)
            status_msg = f"モード切替: {self.current_mode} (T_INDEX={T_INDEX_LIST})"
        
        print(f"[MODE] {status_msg}")
        self.status_label.setText(status_msg)
        self.signal_emitter.status_update.emit(status_msg)
        self.signal_emitter.mode_changed.emit(self.current_mode)
        
        if generator.is_streamdiffusion and generator.is_initialized:
            print(f"[MODE] StreamDiffusion再初期化中... (T_INDEX={T_INDEX_LIST})")
            try:
                del generator.pipe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                generator._initialize_streamdiffusion()
                print("[MODE] 再初期化完了")
            except Exception as e:
                print(f"[MODE] 再初期化エラー: {e}")
                import traceback
                traceback.print_exc()
    
    def on_mode_changed(self, mode):
        print(f"[SIGNAL] モード変更通知受信: {mode}")
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.status_label.setMaximumWidth(self.width() - 20)
        self.status_label.adjustSize()
    
    def on_status_update(self, status):
        self.status_label.setText(status)
        self.status_label.adjustSize()
        print(f"[STATUS] {status}")
    
    def on_image_ready(self, image_path):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.image_label.add_image(pixmap)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.close()
        else:
            super().keyPressEvent(event)


class StreamDiffusionGenerator:
    def __init__(self):
        self.pipe = None
        self.is_initialized = False
        self.generation_times = []
        self.max_history = 10
        self.is_streamdiffusion = False
        self.current_prompt = None
        
    def initialize(self):
        global USE_DUMMY_MODE
        if USE_DUMMY_MODE:
            print("[INFO] ダミーモードで動作します")
            self.is_initialized = True
            return True
        if USE_STREAMDIFFUSION and STREAMDIFFUSION_AVAILABLE:
            return self._initialize_streamdiffusion()
        else:
            print("[INFO] StreamDiffusionが無効またはインストールされていません")
            return self._initialize_normal_pipeline()
    
    def _initialize_streamdiffusion(self):
        try:
            print(f"[INFO] StreamDiffusion初期化中（{CURRENT_MODE}モード）...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[INFO] デバイス: {device}")
            if device.type != "cuda":
                print("[WARNING] CUDAが使用できません。通常パイプラインを使用します")
                return self._initialize_normal_pipeline()
            
            print(f"[INFO] モデル読み込み中: {MODEL_ID}")
            
            from diffusers import StableDiffusionPipeline, AutoencoderTiny, LCMScheduler
            
            pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                safety_checker=None,
            ).to(device)
            
            print("[SUCCESS] ベースモデル読み込み完了")
            
            # LCM-LoRAを適用
            if USE_LCM_LORA:
                print("[INFO] LCM-LoRA適用中...")
                try:
                    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
                    pipe.fuse_lora()
                    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                    print("[SUCCESS] LCM-LoRA適用完了")
                    print(f"[INFO] Scheduler: {type(pipe.scheduler).__name__}")
                except Exception as e:
                    print(f"[ERROR] LCM-LoRA適用失敗: {e}")
                    import traceback
                    traceback.print_exc()
                    print("[WARNING] LCM-LoRAなしで続行します")
            
            print(f"[INFO] T_INDEX_LIST設定: {T_INDEX_LIST}")
            print(f"[INFO] CFG_TYPE設定: {CFG_TYPE}")
            print(f"[INFO] 画像サイズ: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
            
            self.pipe = StreamDiffusion(
                pipe=pipe,
                t_index_list=T_INDEX_LIST,
                torch_dtype=torch.float16,
                cfg_type=CFG_TYPE,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
            )
            
            print("[SUCCESS] StreamDiffusionパイプライン作成完了")
            
            print("[INFO] TinyVAE読み込み中...")
            try:
                self.pipe.vae = AutoencoderTiny.from_pretrained(
                    "madebyollin/taesd"
                ).to(device=device, dtype=torch.float16)
                print("[SUCCESS] TinyVAE読み込み完了")
            except Exception as e:
                print(f"[WARNING] TinyVAE読み込み失敗: {e}")
            
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("[SUCCESS] xformers有効化")
            except Exception as e:
                print(f"[WARNING] xformers無効: {e}")
            
            self.pipe.enable_similar_image_filter = False
            
            print("[INFO] ウォームアップ中...")
            try:
                # 具体的なプロンプトでウォームアップ
                warmup_prompt = "a beautiful anime girl with purple hair, detailed face, high quality"
                print(f"[INFO] ウォームアッププロンプト: {warmup_prompt}")
                
                self.pipe.prepare(
                    prompt=warmup_prompt,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                )
                print("[INFO] prepare()完了")
                
                # 実際に1枚生成
                print("[INFO] テスト画像生成中...")
                output = self.pipe()
                print(f"[INFO] テスト画像生成完了: type={type(output)}")
                
                # 画像確認
                if hasattr(output, 'shape'):
                    print(f"[INFO] 出力shape: {output.shape}")
                
                self.current_prompt = warmup_prompt
                print("[SUCCESS] ウォームアップ完了")
            except Exception as e:
                print(f"[ERROR] ウォームアップエラー: {e}")
                if DEBUG_MODE:
                    import traceback
                    traceback.print_exc()
                print("[WARNING] ウォームアップ失敗。実行は続行しますが、最初の生成が遅くなる可能性があります")
            
            self.is_streamdiffusion = True
            self.is_initialized = True
            print("[SUCCESS] StreamDiffusion初期化完了")
            print(f"[INFO] 現在のモード: {CURRENT_MODE}")
            print(f"[INFO] T_INDEX_LIST: {T_INDEX_LIST}")
            print(f"[INFO] NUM_INFERENCE_STEPS: {NUM_INFERENCE_STEPS}")
            return True
            
        except Exception as e:
            print(f"[ERROR] StreamDiffusion初期化失敗: {e}")
            print("[INFO] 通常パイプラインにフォールバック")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            return self._initialize_normal_pipeline()
    
    def _initialize_normal_pipeline(self):
        try:
            print("[INFO] 通常のStable Diffusionパイプライン初期化中...")
            from diffusers import StableDiffusionPipeline
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[INFO] デバイス: {device}")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                safety_checker=None,
            ).to(device)
            if device.type == "cuda":
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("[INFO] xformers enabled")
                except:
                    pass
            print("[INFO] ウォームアップ中...")
            _ = self.pipe(
                "test",
                num_inference_steps=1,
                guidance_scale=0.0,
            ).images[0]
            self.is_streamdiffusion = False
            self.is_initialized = True
            print("[SUCCESS] 通常パイプライン初期化完了")
            return True
        except Exception as e:
            print(f"[ERROR] 初期化失敗: {e}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            return False
    
    def generate(self, prompt):
        print(f"[GENERATE] generate()呼び出し: prompt='{prompt[:50]}...'")
        print(f"[GENERATE] USE_DUMMY_MODE={USE_DUMMY_MODE}, is_initialized={self.is_initialized}")
        
        if USE_DUMMY_MODE or not self.is_initialized:
            print("[GENERATE] ダミー画像を生成します")
            return self._generate_dummy(prompt)
        if len(prompt.strip()) < 3:
            print(f"[SKIP] プロンプトが短すぎます: '{prompt}'")
            return None
        
        start_time = time.time()
        try:
            if DEBUG_MODE:
                print(f"[DEBUG] プロンプト: {prompt}")
                print(f"[DEBUG] モード: {CURRENT_MODE} ({'StreamDiffusion' if self.is_streamdiffusion else '通常'})")
                print(f"[DEBUG] T_INDEX: {T_INDEX_LIST}")
                print(f"[DEBUG] is_streamdiffusion: {self.is_streamdiffusion}")
            
            if self._contains_japanese(prompt):
                print("[DEBUG] 日本語を検出しました。翻訳を試みます...")
                try:
                    from googletrans import Translator
                    translator = Translator()
                    original_prompt = prompt
                    prompt = translator.translate(prompt, src='ja', dest='en').text
                    print(f"[DEBUG] 翻訳成功:")
                    print(f"  元: {original_prompt[:100]}...")
                    print(f"  訳: {prompt[:100]}...")
                except Exception as trans_error:
                    print(f"[WARNING] 翻訳失敗: {trans_error}")
                    print("[WARNING] 日本語のまま処理を続行します（品質低下の可能性）")
            
            # プロンプトの品質チェック（強化版）
            # 視覚的な名詞が含まれているかチェック
            visual_nouns = [
                'woman', 'man', 'person', 'face', 'portrait', 'landscape', 'mountain', 'ocean', 
                'sunset', 'city', 'building', 'car', 'animal', 'cat', 'dog', 'tree', 'flower',
                'sky', 'cloud', 'beach', 'forest', 'river', 'lake', 'bird', 'house', 'girl', 
                'boy', 'child', 'baby', 'street', 'road', 'garden', 'park', 'scene'
            ]
            
            has_visual_content = any(noun in prompt.lower() for noun in visual_nouns)
            is_too_short = len(prompt.split()) < 3
            
            # 抽象的キーワード
            abstract_keywords = [
                '提案', 'フォーム', '考え', 'という', '思う', '研究', '定義', '領域',
                'suggestion', 'form', 'thinking', 'that', 'this', 'research', 'define', 
                'area', 'concept', 'idea', 'theory', 'important', 'carefully', 'general'
            ]
            has_abstract = any(word in prompt.lower() for word in abstract_keywords)
            
            # プロンプトが不適切な場合
            if is_too_short or not has_visual_content or has_abstract:
                print("[WARNING] 不適切なプロンプトを検出しました")
                print(f"[WARNING] 元のプロンプト: {prompt[:100]}")
                print(f"[WARNING] 判定: 短すぎる={is_too_short}, 視覚的内容なし={not has_visual_content}, 抽象的={has_abstract}")
                print("[INFO] ランダムな高品質プロンプトに置き換えます")
                
                import random
                quality_prompts = [
                    "a beautiful young woman with long flowing hair, golden hour lighting, bokeh background, professional portrait",
                    "majestic mountain landscape with snow peaks, dramatic clouds, sunset colors, 8k photography",
                    "a cute fluffy cat with blue eyes sitting on a windowsill, soft natural lighting",
                    "modern city skyline at night, neon lights reflecting on water, cinematic composition",
                    "serene beach scene at sunset, gentle waves, palm trees silhouette, warm colors",
                    "portrait of an elderly man with wise eyes, dramatic side lighting, black background",
                    "enchanted forest with sunbeams filtering through trees, misty atmosphere, magical mood",
                    "elegant woman in red dress, fashion photography, studio lighting, white background",
                    "powerful waterfall in lush green jungle, long exposure, tropical paradise",
                    "cozy coffee shop interior, warm ambient lighting, bokeh lights, inviting atmosphere"
                ]
                prompt = random.choice(quality_prompts)
                print(f"[INFO] 新プロンプト: {prompt}")
            
            # プロンプトが政治・ニュース系の場合
            if any(word in prompt for word in ['選挙', '政治', '国政', '議員', 'election', 'politics', 'government']):
                print("[WARNING] ニュース/政治系のテキストを検出しました")
                print("[INFO] 風景画像に置き換えます")
                prompt = "beautiful natural landscape with mountains and pristine lake, golden hour, professional photography"
            
            enhanced_prompt = f"{prompt}, photorealistic, professional photography, 8k uhd, highly detailed face, perfect face, sharp focus, natural lighting, realistic, portrait photography"
            negative_prompt = "low quality, blurry, bad, worst quality, anime, cartoon, painting, illustration, drawing, art, sketch, deformed face, ugly face, bad anatomy, disfigured, mutated"
            
            print(f"[DEBUG] 最終プロンプト: {enhanced_prompt[:150]}...")
            
            if self.is_streamdiffusion:
                print(f"[GENERATE] StreamDiffusionで生成開始")
                if self.current_prompt != enhanced_prompt:
                    print(f"[DEBUG] 新しいプロンプトでprepare()実行")
                    print(f"[DEBUG] enhanced_prompt: {enhanced_prompt[:100]}...")
                    self.pipe.prepare(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                    )
                    self.current_prompt = enhanced_prompt
                    print("[DEBUG] prepare()完了")
                else:
                    print("[DEBUG] 同じプロンプトなのでprepare()スキップ")
                
                print("[DEBUG] pipe()実行中...")
                output_image = self.pipe()
                print(f"[DEBUG] pipe()完了: type={type(output_image)}")
                
                print("[DEBUG] PIL変換開始...")
                output = self._convert_output_to_pil(output_image)
                print(f"[DEBUG] PIL変換完了: {output.size if output else 'None'}")
            
            else:
                # 通常パイプライン（20ステップで生成）
                print(f"[GENERATE] 通常パイプラインで生成開始")
                import random
                
                seed = random.randint(0, 2147483647)
                
                # ステップ数を20に固定
                inference_steps = 20
                guide_scale = 7.5
                
                print(f"[DEBUG] 通常モード: steps={inference_steps}, scale={guide_scale}")
                print(f"[DEBUG] 実際の設定確認: inference_steps={inference_steps}, guidance_scale={guide_scale}")
                
                generator_obj = torch.Generator(device=self.pipe.device).manual_seed(seed)
                print(f"[DEBUG] パイプライン実行開始 (seed={seed}, steps={inference_steps})")
                
                result = self.pipe(
                    enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=inference_steps,
                    guidance_scale=guide_scale,
                    height=IMAGE_HEIGHT,
                    width=IMAGE_WIDTH,
                    generator=generator_obj,
                )
                output = result.images[0]
                print(f"[DEBUG] 通常パイプライン完了: {output.size}")
            
            if output is None:
                print("[ERROR] 出力画像がNone!")
                return self._generate_dummy("Output is None")
            
            generation_time = time.time() - start_time
            self.generation_times.append(generation_time)
            if len(self.generation_times) > self.max_history:
                self.generation_times.pop(0)
            avg_time = sum(self.generation_times) / len(self.generation_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            mode = f"StreamDiffusion({CURRENT_MODE})" if self.is_streamdiffusion else "通常"
            print(f"[PERF] 生成時間: {generation_time:.2f}秒 | 平均FPS: {fps:.2f} fps | モード: {mode}")
            print(f"[SUCCESS] 画像生成成功: size={output.size}, mode={output.mode}")
            return output
        except Exception as e:
            print(f"[ERROR] 画像生成失敗: {e}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            print("[FALLBACK] ダミー画像を返します")
            return self._generate_dummy(prompt)
    
    def _convert_output_to_pil(self, output_image):
        try:
            if DEBUG_MODE:
                print(f"[DEBUG] 出力タイプ: {type(output_image)}")
                if hasattr(output_image, 'shape'):
                    print(f"[DEBUG] 出力shape: {output_image.shape}")
                elif hasattr(output_image, 'size'):
                    print(f"[DEBUG] 出力size: {output_image.size}")
            
            if isinstance(output_image, Image.Image):
                if DEBUG_MODE:
                    print(f"[DEBUG] PIL Image: {output_image.size}, mode: {output_image.mode}")
                return output_image
            
            if isinstance(output_image, torch.Tensor):
                output_np = output_image.detach().cpu().numpy()
            elif isinstance(output_image, np.ndarray):
                output_np = output_image.copy()
            else:
                print(f"[WARNING] 未対応の出力型: {type(output_image)}")
                return self._generate_dummy(f"Unknown type: {type(output_image)}")
            
            if DEBUG_MODE:
                print(f"[DEBUG] NumPy変換後shape: {output_np.shape}, dtype: {output_np.dtype}, min: {output_np.min():.3f}, max: {output_np.max():.3f}")
            
            while len(output_np.shape) > 3:
                output_np = output_np.squeeze(0)
                if DEBUG_MODE:
                    print(f"[DEBUG] squeeze後shape: {output_np.shape}")
            
            # 正規化の修正: -1~1の範囲を0~255に変換
            if output_np.dtype == np.float16 or output_np.dtype == np.float32:
                # 値の範囲を確認
                min_val = output_np.min()
                max_val = output_np.max()
                
                if DEBUG_MODE:
                    print(f"[DEBUG] 正規化前: min={min_val:.3f}, max={max_val:.3f}")
                
                # -1~1の範囲の場合
                if min_val < 0:
                    # -1~1 を 0~1 に変換
                    output_np = (output_np + 1.0) / 2.0
                    if DEBUG_MODE:
                        print(f"[DEBUG] -1~1 -> 0~1変換後: min={output_np.min():.3f}, max={output_np.max():.3f}")
                
                # コントラスト強化: 0~1の範囲を最大限に活用
                min_val_normalized = output_np.min()
                max_val_normalized = output_np.max()
                
                if max_val_normalized > min_val_normalized:
                    # 値を0~1の全範囲にストレッチ
                    output_np = (output_np - min_val_normalized) / (max_val_normalized - min_val_normalized)
                    if DEBUG_MODE:
                        print(f"[DEBUG] コントラスト強化後: min={output_np.min():.3f}, max={output_np.max():.3f}")
                
                # 0~1 を 0~255 に変換
                output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
                
                if DEBUG_MODE:
                    print(f"[DEBUG] 0~255変換後: min={output_np.min()}, max={output_np.max()}")
            else:
                output_np = output_np.clip(0, 255).astype(np.uint8)
            
            if len(output_np.shape) == 3:
                if output_np.shape[0] in [1, 3, 4]:
                    output_np = np.transpose(output_np, (1, 2, 0))
                    if DEBUG_MODE:
                        print(f"[DEBUG] transpose後shape: {output_np.shape}")
            
            if len(output_np.shape) == 2:
                output_np = np.stack([output_np] * 3, axis=-1)
                if DEBUG_MODE:
                    print(f"[DEBUG] グレースケール->RGB shape: {output_np.shape}")
            
            elif output_np.shape[-1] == 1:
                output_np = np.repeat(output_np, 3, axis=-1)
                if DEBUG_MODE:
                    print(f"[DEBUG] 1ch->RGB shape: {output_np.shape}")
            
            elif output_np.shape[-1] == 4:
                output_np = output_np[:, :, :3]
                if DEBUG_MODE:
                    print(f"[DEBUG] RGBA->RGB shape: {output_np.shape}")
            
            if DEBUG_MODE:
                print(f"[DEBUG] 最終shape: {output_np.shape}, dtype: {output_np.dtype}")
            
            if output_np.shape[-1] == 3 and len(output_np.shape) == 3:
                return Image.fromarray(output_np, mode='RGB')
            else:
                print(f"[WARNING] 想定外のshape: {output_np.shape}")
                return self._generate_dummy("Shape conversion error")
                
        except Exception as e:
            print(f"[ERROR] 画像変換エラー: {e}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            return self._generate_dummy("Conversion error")
    
    def _contains_japanese(self, text):
        for char in text:
            if '\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FFF':
                return True
        return False
    
    def _generate_dummy(self, prompt):
        """ダミー画像を生成（テスト・デバッグ用）"""
        # カラフルなグラデーション背景を生成
        import random
        from PIL import ImageDraw, ImageFont
        
        # ランダムな明るい色
        colors = [
            (255, 182, 193),  # ライトピンク
            (173, 216, 230),  # ライトブルー
            (144, 238, 144),  # ライトグリーン
            (255, 218, 185),  # ピーチ
            (221, 160, 221),  # プラム
            (255, 250, 205),  # レモンシフォン
        ]
        color = random.choice(colors)
        
        dummy_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color=color)
        draw = ImageDraw.Draw(dummy_image)
        
        # テキストを描画
        text_lines = [
            "DUMMY MODE",
            f"Prompt: {prompt[:30]}",
            "",
            "Set USE_DUMMY_MODE=False",
            "for real generation"
        ]
        
        y_offset = 50
        for line in text_lines:
            # 影をつける
            draw.text((12, y_offset + 2), line, fill=(0, 0, 0))
            draw.text((10, y_offset), line, fill=(255, 255, 255))
            y_offset += 30
        
        print(f"[DUMMY] ダミー画像生成: {prompt[:30]}...")
        return dummy_image


generator = StreamDiffusionGenerator()


def generate_image(prompt, shared_folder, signal_emitter):
    print(f"\n{'='*60}")
    print(f"[TASK] 画像生成タスク開始")
    print(f"[TASK] プロンプト: {prompt}")
    print(f"[TASK] 共有フォルダ: {shared_folder}")
    print(f"{'='*60}\n")
    
    try:
        signal_emitter.status_update.emit(f"画像生成中 ({CURRENT_MODE}): {prompt[:40]}...")
        
        print("[TASK] generator.generate()呼び出し...")
        image = generator.generate(prompt)
        print(f"[TASK] generator.generate()完了: image={type(image)}")
        
        if image is None:
            print("[TASK] 画像がNoneのためスキップ")
            signal_emitter.status_update.emit(f"スキップ: {prompt[:30]}")
            return
        
        print(f"[TASK] 画像取得成功: size={image.size}, mode={image.mode}")
        
        if SAVE_IMAGES:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # マイクロ秒を追加して重複防止
            filename = f"generated_{timestamp}.png"
            filepath = shared_folder / filename
            print(f"[TASK] 画像を保存中: {filepath}")
            image.save(filepath)
            print(f"[TASK] 保存完了")
            signal_emitter.status_update.emit(f"生成完了 ({CURRENT_MODE}): {filename}")
            signal_emitter.image_ready.emit(str(filepath))
            print(f"[SUCCESS] 画像保存: {filepath}")
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                filepath = tmp.name
                print(f"[TASK] 一時ファイルに保存中: {filepath}")
                image.save(filepath)
            signal_emitter.status_update.emit(f"生成完了 ({CURRENT_MODE})（未保存）")
            signal_emitter.image_ready.emit(str(filepath))
            print(f"[SUCCESS] 画像生成完了（保存なし）")
    except Exception as e:
        error_msg = f"エラー: {str(e)}"
        signal_emitter.status_update.emit(error_msg)
        print(f"[ERROR] 画像生成タスク失敗: {e}")
        import traceback
        traceback.print_exc()


def osc_handler(address, *args, shared_folder=None, signal_emitter=None):
    if len(args) >= 1:
        prompt = str(args[0])
    else:
        print(f"[ERROR] OSCメッセージの形式が不正: {args}")
        return
    print(f"[OSC受信] {address}: {prompt}")
    signal_emitter.status_update.emit(f"OSC受信: {prompt[:30]}...")
    signal_emitter.prompt_received.emit(prompt)
    thread = threading.Thread(
        target=generate_image,
        args=(prompt, shared_folder, signal_emitter),
        daemon=True,
    )
    thread.start()


def start_osc_server(shared_folder, signal_emitter):
    disp = dispatcher.Dispatcher()
    disp.map(OSC_ADDRESS, lambda addr, *args: osc_handler(
        addr, *args, shared_folder=shared_folder, signal_emitter=signal_emitter
    ))
    try:
        server = osc_server.ThreadingOSCUDPServer((OSC_IP, OSC_PORT), disp)
        print(f"[OSC] サーバー起動成功: {OSC_IP}:{OSC_PORT} (address: {OSC_ADDRESS})")
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        return server
    except OSError as e:
        print(f"[ERROR] OSCサーバー起動失敗: ポート {OSC_PORT} が既に使用されています")
        raise


def main():
    SHARED_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 共有フォルダ作成: {SHARED_FOLDER.absolute()}")
    print("[INFO] 初期化中...")
    if not generator.initialize():
        print("[WARNING] 初期化に失敗しました。")
    
    app = QApplication(sys.argv)
    main_window = ImageViewerWindow()
    prompt_window = PromptWindow()
    screens = app.screens()
    
    if len(screens) > IMAGE_DISPLAY_MONITOR:
        screen = screens[IMAGE_DISPLAY_MONITOR]
        geometry = screen.geometry()
        if USE_FULLSCREEN:
            main_window.setGeometry(geometry)
            main_window.showFullScreen()
        else:
            main_window.setGeometry(geometry.x() + 100, geometry.y() + 100, 800, 700)
            main_window.show()
    else:
        main_window.show()
    
    if len(screens) > PROMPT_DISPLAY_MONITOR:
        screen = screens[PROMPT_DISPLAY_MONITOR]
        geometry = screen.geometry()
        if USE_FULLSCREEN:
            prompt_window.setGeometry(geometry)
            prompt_window.showFullScreen()
        else:
            prompt_window.setGeometry(geometry.x() + 920, geometry.y() + 100, 500, 600)
            prompt_window.show()
    else:
        prompt_window.show()
    
    main_window.signal_emitter.prompt_received.connect(prompt_window.add_prompt)
    main_window.prompt_window = prompt_window
    
    try:
        osc_server_instance = start_osc_server(SHARED_FOLDER, main_window.signal_emitter)
        mode = f"StreamDiffusion({CURRENT_MODE})" if generator.is_streamdiffusion else "通常パイプライン"
        startup_msg = f"""起動完了 ({mode})
OSC: {OSC_IP}:{OSC_PORT}
モデル: {MODEL_ID}
現在のモード: {CURRENT_MODE}
T_INDEX: {T_INDEX_LIST}
総ステップ数: {NUM_INFERENCE_STEPS}
予想FPS: {'15-20fps(高品質)' if CURRENT_MODE == 'high' else '25-30fps(低遅延)'}

💡 ボタンをクリックでモード切替可能"""
        print(f"[INFO] {startup_msg}")
        main_window.signal_emitter.status_update.emit(startup_msg)
        
        print("\n" + "="*60)
        print("StreamDiffusion モード切替システム起動完了")
        print("="*60)
        print("【モード切替】")
        print(f"  - 高品質モード: {len(T_INDEX_LIST_HIGH)}サンプル、~15-20fps、顔の品質UP")
        print(f"  - 低遅延モード: {len(T_INDEX_LIST_LOW)}サンプル、~25-30fps、高速生成")
        print("  - 画面下部のボタンをクリックで切替")
        print("【重要】")
        print(f"  - NUM_INFERENCE_STEPS固定: {NUM_INFERENCE_STEPS}")
        print(f"  - T_INDEX範囲: 0～{NUM_INFERENCE_STEPS-1}")
        print("="*60 + "\n")
        
    except Exception as error:
        error_msg = f"起動エラー: {str(error)}"
        print(f"[ERROR] {error_msg}")
        main_window.show()
        prompt_window.show()
        main_window.signal_emitter.status_update.emit(error_msg)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
