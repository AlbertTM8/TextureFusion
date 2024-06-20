from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QSlider, QFileDialog, QPushButton, QHBoxLayout, QComboBox, QCheckBox
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage
import numpy as np
from PIL import Image
import scipy.ndimage
import sys
from module_normals_to_height import apply

class NormalMapApp(QMainWindow):
    normalMapUpdated = Signal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Normal Map Generator')
        
        # Main widget
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        # Layouts
        self.layout = QVBoxLayout()
        self.controls_layout = QVBoxLayout()
        self.filter_layout = QHBoxLayout()
        self.invert_layout = QHBoxLayout()
        self.slider_layout = QHBoxLayout()
        
        # Filter dropdown
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("Sobel")
        self.filter_combo.addItem("Scharr")
        self.filter_combo.setCurrentIndex(1)
        self.filter_combo.currentTextChanged.connect(self.update_normal_map)
        self.filter_layout.addWidget(QLabel('Filter:'))
        self.filter_layout.addWidget(self.filter_combo)
        
        # Invert checkboxes
        self.invert_r_checkbox = QCheckBox('Invert R')
        self.invert_r_checkbox.stateChanged.connect(self.update_normal_map)
        self.invert_layout.addWidget(self.invert_r_checkbox)
        
        self.invert_g_checkbox = QCheckBox('Invert G')
        self.invert_g_checkbox.stateChanged.connect(self.update_normal_map)
        self.invert_layout.addWidget(self.invert_g_checkbox)
        
        # Sliders
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setMinimum(1)
        self.strength_slider.setMaximum(10)
        self.strength_slider.setValue(5)
        self.strength_slider.valueChanged.connect(self.update_normal_map)
        self.slider_layout.addWidget(QLabel('Strength:'))
        self.slider_layout.addWidget(self.strength_slider)
        
        self.level_slider = QSlider(Qt.Horizontal)
        self.level_slider.setMinimum(0)
        self.level_slider.setMaximum(10)
        self.level_slider.setValue(4)
        self.level_slider.valueChanged.connect(self.update_normal_map)
        self.slider_layout.addWidget(QLabel('Level:'))
        self.slider_layout.addWidget(self.level_slider)
    
        
        # Set main layout
        self.layout.addLayout(self.filter_layout)
        self.layout.addLayout(self.invert_layout)
        self.layout.addLayout(self.slider_layout)
        self.main_widget.setLayout(self.layout)

        
        # Placeholder for height map
        self.height_map = None
        
    # def load_image(self):
    #     options = QFileDialog.Options()
    #     file_name, _ = QFileDialog.getOpenFileName(self, "Open Height Map Image", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
    #     if file_name:
    #         self.height_map = self.load_height_map(file_name)
    #         self.update_normal_map()
    
    def load_height_map(self, height_map):
        # image = image.convert('L')
        # height_map = np.array(image).astype(np.float32) / 255.0
        self.height_map = height_map
        self.update_normal_map()
        
    
    def sobel_filter(self, height_map):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        Gx = scipy.ndimage.convolve(height_map, Kx)
        Gy = scipy.ndimage.convolve(height_map, Ky)
        return Gx, Gy

    def scharr_filter(self, height_map):
        Kx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
        Ky = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
        Gx = scipy.ndimage.convolve(height_map, Kx)
        Gy = scipy.ndimage.convolve(height_map, Ky)
        return Gx, Gy

    def calculate_normal_map(self, height_map, filter_type='sobel', strength=2.5, level=7, invertR=1, invertG=1):
        if filter_type == 'sobel':
            Gx, Gy = self.sobel_filter(height_map)
        else:
            Gx, Gy = self.scharr_filter(height_map)

        dz = 1.0 / strength * (1.0 + 2 ** level)

        normal_map = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.float32)
        normal_map[..., 0] = -Gx * invertR * 255.0
        normal_map[..., 1] = -Gy * invertG * 255.0
        normal_map[..., 2] = dz

        norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
        normal_map = normal_map / norm

        normal_map = (normal_map + 1) / 2 * 255
        normal_map = normal_map.astype(np.uint8)
        return normal_map
    
    def update_normal_map(self):
        if self.height_map is None:
            return
        filter_type = self.filter_combo.currentText().lower()
        strength = self.strength_slider.value() / 10.0
        level = self.level_slider.value()
        invertR = -1 if self.invert_r_checkbox.isChecked() else 1
        invertG = -1 if self.invert_g_checkbox.isChecked() else 1
        
        self.normal_map = self.calculate_normal_map(self.height_map, filter_type=filter_type, strength=strength, level=level, invertR=invertR, invertG=invertG)
        

        self.normalMapUpdated.emit(self.normal_map)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = NormalMapApp()
    ex.show()
    sys.exit(app.exec())
