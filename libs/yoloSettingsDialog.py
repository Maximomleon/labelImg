import os
import psutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class YoloSettingsDialog(QDialog):
    def __init__(self, parent=None, model_path="", classes_path="", conf=0.25, imgsz=1280, scale="nano"):
        super(YoloSettingsDialog, self).__init__(parent)
        self.setWindowTitle("Configuración de Autodetección YOLO y Diagnóstico de PC")
        self.resize(550, 420)

        layout = QVBoxLayout()

        # --- Hardware Diagnostics Box ---
        hw_group = QGroupBox("💻 Diagnóstico de Componentes de la PC")
        hw_layout = QVBoxLayout()

        cpu_threads = psutil.cpu_count(logical=True)
        ram_gb = psutil.virtual_memory().total / (1024**3)

        has_cuda = False
        gpu_info = "No se detectó GPU NVIDIA con CUDA (Modo CPU Activo)"
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                gpu_info = f"NVIDIA {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB VRAM)"
        except Exception:
            pass

        # Formulate Recommendation
        if has_cuda:
            rec_text = "💡 Recomendación: Tu PC cuenta con GPU NVIDIA. Puedes usar modelos **Small (s)** o **Medium (m)** sin pérdida de velocidad."
            default_rec_scale = "small"
        else:
            rec_text = "💡 Recomendación: Al usar CPU sin GPU dedicada, se recomienda **Nano (yolo26n-seg)** para inferencia fluida e instantánea."
            default_rec_scale = "nano"

        hw_label = QLabel(
            f"• <b>Procesador:</b> {cpu_threads} Hilos de procesamiento<br>"
            f"• <b>Memoria RAM:</b> {ram_gb:.1f} GB<br>"
            f"• <b>Tarjeta Gráfica:</b> {gpu_info}<br><br>"
            f"<font color='#007ACC'>{rec_text}</font>"
        )
        hw_label.setWordWrap(True)
        hw_layout.addWidget(hw_label)
        hw_group.setLayout(hw_layout)
        layout.addWidget(hw_group)

        # --- Model Scale Selection ---
        scale_layout = QHBoxLayout()
        scale_label = QLabel("Escala del Modelo (Arquitectura):")
        self.scale_combo = QComboBox()
        self.scale_combo.addItems([
            "Nano (yolo26n-seg) - Ultra Rápido (Recomendado CPU)",
            "Small (yolo26s-seg) - Balanceado",
            "Medium (yolo26m-seg) - Alta Precisión (Recomendado GPU)"
        ])
        
        # Set current scale index
        s_lower = str(scale).lower()
        if "medium" in s_lower or s_lower == "m":
            self.scale_combo.setCurrentIndex(2)
        elif "small" in s_lower or s_lower == "s":
            self.scale_combo.setCurrentIndex(1)
        else:
            self.scale_combo.setCurrentIndex(0)
            
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.scale_combo)
        layout.addLayout(scale_layout)

        # --- 1. Model Path ---
        model_layout = QHBoxLayout()
        model_label = QLabel("Modelo YOLO (.pt):")
        self.model_line = QLineEdit(model_path)
        self.model_btn = QPushButton("Examinar...")
        self.model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_line)
        model_layout.addWidget(self.model_btn)
        layout.addLayout(model_layout)

        # --- 2. Classes Path ---
        classes_layout = QHBoxLayout()
        classes_label = QLabel("Archivo de Clases (.txt):")
        self.classes_line = QLineEdit(classes_path)
        self.classes_btn = QPushButton("Examinar...")
        self.classes_btn.clicked.connect(self.browse_classes)
        classes_layout.addWidget(classes_label)
        classes_layout.addWidget(self.classes_line)
        classes_layout.addWidget(self.classes_btn)
        layout.addLayout(classes_layout)

        # --- 3. Confidence Threshold ---
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Umbral de Confianza (conf):")
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(float(conf))
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_spin)
        layout.addLayout(conf_layout)

        # --- 4. Image Size (imgsz) ---
        imgsz_layout = QHBoxLayout()
        imgsz_label = QLabel("Resolución de Inspección (imgsz):")
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["640", "800", "1024", "1280", "1600", "1920"])
        index = self.imgsz_combo.findText(str(imgsz))
        if index >= 0:
            self.imgsz_combo.setCurrentIndex(index)
        else:
            self.imgsz_combo.setCurrentText("1280")
        imgsz_layout.addWidget(imgsz_label)
        imgsz_layout.addWidget(self.imgsz_combo)
        layout.addLayout(imgsz_layout)

        # Buttons (OK / Cancel)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self.setLayout(layout)

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Modelo YOLO (.pt)", "/home/maximo", "Modelos YOLO (*.pt)")
        if path:
            self.model_line.setText(path)

    def browse_classes(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo de Clases (.txt)", "/home/maximo", "Archivos de texto (*.txt)")
        if path:
            self.classes_line.setText(path)

    def get_values(self):
        idx = self.scale_combo.currentIndex()
        scale_key = "nano" if idx == 0 else ("small" if idx == 1 else "medium")
        return {
            "model_path": self.model_line.text().strip(),
            "classes_path": self.classes_line.text().strip(),
            "conf": self.conf_spin.value(),
            "imgsz": int(self.imgsz_combo.currentText()),
            "scale": scale_key
        }
