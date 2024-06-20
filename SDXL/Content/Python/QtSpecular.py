from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QSlider, QFileDialog, QPushButton, QHBoxLayout, QComboBox, QCheckBox
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage
import numpy as np
from PIL import Image
import sys

class SpecularApp(QMainWindow):
    specularMapUpdated = Signal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Specular Map Viewer')
        
        # Main widget
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        # Layouts
        self.layout = QVBoxLayout()
        self.controls_layout = QHBoxLayout()
        
        # Sliders
        self.range_slider = QSlider(Qt.Horizontal)
        self.range_slider.setMinimum(0)
        self.range_slider.setMaximum(100)
        self.range_slider.setValue(50)
        self.range_slider.valueChanged.connect(self.update_specular)
        self.controls_layout.addWidget(QLabel('Range:'))
        self.controls_layout.addWidget(self.range_slider)
        
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setValue(50)
        self.strength_slider.valueChanged.connect(self.update_specular)
        self.controls_layout.addWidget(QLabel('Strength:'))
        self.controls_layout.addWidget(self.strength_slider)
        
        self.mean_slider = QSlider(Qt.Horizontal)
        self.mean_slider.setMinimum(0)
        self.mean_slider.setMaximum(100)
        self.mean_slider.setValue(80)
        self.mean_slider.valueChanged.connect(self.update_specular)
        self.controls_layout.addWidget(QLabel('Mean:'))
        self.controls_layout.addWidget(self.mean_slider)
        
        # Checkboxes
        self.invert_checkbox = QCheckBox('Invert')
        self.invert_checkbox.stateChanged.connect(self.update_specular)
        self.controls_layout.addWidget(self.invert_checkbox)
        
        self.flip_y_checkbox = QCheckBox('Flip Y')
        self.flip_y_checkbox.stateChanged.connect(self.update_specular)
        self.controls_layout.addWidget(self.flip_y_checkbox)
        
        # Dropdown for falloff
        self.falloff_dropdown = QComboBox()
        self.falloff_dropdown.addItems(['None', 'Linear', 'Square'])
        self.falloff_dropdown.setCurrentIndex(1)
        self.falloff_dropdown.currentTextChanged.connect(self.update_specular)

        self.controls_layout.addWidget(QLabel('Falloff:'))
        self.controls_layout.addWidget(self.falloff_dropdown)
        
        self.layout.addLayout(self.controls_layout)
        

        # Set main layout
        self.main_widget.setLayout(self.layout)
        
        # Placeholder for height map
        self.height_map = None
        
    def load_height_map(self, height_map):
        # image = image.convert('L')
        # height_map = np.array(image).astype(np.float32) / 255.0
        self.height_map = height_map
        self.update_specular()
    
    def specular_shader(self, height_map, range_val, strength, mean, invert, falloff, flip_y):
        """
        Applies the specular shader logic to the height map.
        """
        # Flip Y if necessary
        if flip_y:
            height_map = np.flip(height_map, axis=0)
        
        # Calculate the percentage distance to mean
        perc_dist_to_mean = (range_val - np.abs(height_map - mean)) / range_val
        
        # Apply falloff
        if falloff == "None":  # No FallOff
            perc_dist_to_mean = np.where(perc_dist_to_mean > 0.0, 1.0, 0.0)
        elif falloff == "Linear":  # Linear
            perc_dist_to_mean = np.where(perc_dist_to_mean > 0.0, perc_dist_to_mean, 0.0)
        elif falloff == "Square":  # Square
            perc_dist_to_mean = np.where(perc_dist_to_mean > 0.0, np.sqrt(perc_dist_to_mean), 0.0)
        
        # Create the specular map
        specular_map = np.broadcast_to(perc_dist_to_mean[..., np.newaxis], perc_dist_to_mean.shape + (3,)) * strength
        
        # Invert if necessary
        if invert:
            specular_map = 1.0 - specular_map
        
        return specular_map

    def update_specular(self):
        if self.height_map is None:
            return
        
        range_val = self.range_slider.value() / 100.0
        strength = self.strength_slider.value() / 100.0
        mean = self.mean_slider.value() / 100.0
        invert = self.invert_checkbox.isChecked()
        flip_y = self.flip_y_checkbox.isChecked()
        falloff = self.falloff_dropdown.currentText()
        
        specular_map = self.specular_shader(self.height_map, range_val, strength, mean, invert, falloff, flip_y)
        # specular_image = Image.fromarray((specular_map * 255).astype(np.uint8))

        # qimage = QImage(specular_image.tobytes(), specular_image.width, specular_image.height, QImage.Format_RGB888)
        # pixmap = QPixmap.fromImage(qimage)
        # self.image_label.setPixmap(pixmap)
        self.specularMapUpdated.emit(specular_map)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SpecularApp()
    ex.show()
    sys.exit(app.exec())
