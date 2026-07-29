import os
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class YoloSettingsDialog(QDialog):
    def __init__(self, parent=None, model_path="", classes_path="", conf=0.25, imgsz=1280):
        super(YoloSettingsDialog, self).__init__(parent)
        self.setWindowTitle("Configuracion de Autodeteccion YOLO")
        self.resize(500, 250)

        layout = QVBoxLayout()

        # 1. Model Path
        model_layout = QHBoxLayout()
        model_label = QLabel("Modelo YOLO (.pt):")
        self.model_line = QLineEdit(model_path)
        self.model_btn = QPushButton("Examinar...")
        self.model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_line)
        model_layout.addWidget(self.model_btn)
        layout.addLayout(model_layout)

        # 2. Classes Path
        classes_layout = QHBoxLayout()
        classes_label = QLabel("Archivo de Clases (.txt):")
        self.classes_line = QLineEdit(classes_path)
        self.classes_btn = QPushButton("Examinar...")
        self.classes_btn.clicked.connect(self.browse_classes)
        classes_layout.addWidget(classes_label)
        classes_layout.addWidget(self.classes_line)
        classes_layout.addWidget(self.classes_btn)
        layout.addLayout(classes_layout)

        # 3. Confidence Threshold
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Umbral de Confianza (conf):")
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(float(conf))
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_spin)
        layout.addLayout(conf_layout)

        # 4. Image Size (imgsz)
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
        return {
            "model_path": self.model_line.text().strip(),
            "classes_path": self.classes_line.text().strip(),
            "conf": self.conf_spin.value(),
            "imgsz": int(self.imgsz_combo.currentText())
        }
