from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QSlider, QFileDialog, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage
import numpy as np
from PIL import Image
import sys
from utils_image import load_image

class OcclusionApp(QMainWindow):
    occlusionMapUpdated = Signal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Occlusion Map Viewer')
        
        # Main widget
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        # Layouts
        self.layout = QVBoxLayout()
        self.controls_layout = QHBoxLayout()
        

        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setMinimum(-1000)
        self.height_slider.setMaximum(100)
        self.height_slider.setValue(-50)
        self.height_slider.valueChanged.connect(self.update_occlusion)
        self.controls_layout.addWidget(QLabel('Height:'))
        self.controls_layout.addWidget(self.height_slider)
        
        self.layout.addLayout(self.controls_layout)
        
        # Light direction controls
        self.light_x_slider = QSlider(Qt.Horizontal)
        self.light_x_slider.setMinimum(-100)
        self.light_x_slider.setMaximum(100)
        self.light_x_slider.setValue(0)
        self.light_x_slider.valueChanged.connect(self.update_occlusion)
        self.controls_layout.addWidget(QLabel('Light X:'))
        self.controls_layout.addWidget(self.light_x_slider)
        
        self.light_y_slider = QSlider(Qt.Horizontal)
        self.light_y_slider.setMinimum(-100)
        self.light_y_slider.setMaximum(100)
        self.light_y_slider.setValue(100)
        self.light_y_slider.valueChanged.connect(self.update_occlusion)
        self.controls_layout.addWidget(QLabel('Light Y:'))
        self.controls_layout.addWidget(self.light_y_slider)
        
        self.light_z_slider = QSlider(Qt.Horizontal)
        self.light_z_slider.setMinimum(-100)
        self.light_z_slider.setMaximum(100)
        self.light_z_slider.setValue(100)
        self.light_z_slider.valueChanged.connect(self.update_occlusion)
        self.controls_layout.addWidget(QLabel('Light Z:'))
        self.controls_layout.addWidget(self.light_z_slider)
        
        # Set main layout
        self.main_widget.setLayout(self.layout)
        
        # Placeholder for normal map
        self.normal_map = None

    def load_maps(self, normal_map, height_map):
        normal_map = (normal_map / 255.0) * 2.0 - 1.0  # Normalize to range [-1, 1]
        self.normal_map = normal_map
        #if height map is 2d convert to 3d
        if len(height_map.shape) == 2:
            height_map_3d = np.stack((height_map, height_map, height_map), axis=-1)
        #SCALE IMAGE TO 0-1
        # height_map_3d = (height_map_3d - np.min(height_map_3d)) / (np.max(height_map_3d) - np.min(height_map_3d))
        self.height_map = height_map_3d
        self.update_occlusion()
    
    
    def compute_occlusion(self, normals, light_dir, height_map, height_factor=0.5):
        # Normalize the light direction vector
        light_dir = np.array(light_dir)
        light_dir /= np.linalg.norm(light_dir)
        
        # Compute the dot product between normals and light direction for the entire image
        occlusion = np.einsum('ijk,k->ij', normals, light_dir)
        
        # Normalize the height map to the range [0, 1]
        height_map_normalized = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
        height_map_normalized = height_map_normalized[:, :, 0]
        height_map = 1.0 - height_map_normalized
        # Subtract the scaled height map from the occlusion values to darken lower areas
        occlusion += height_map * height_factor

        # Clip the values to be in the range [0, 1]
        occlusion = np.clip(occlusion, 0, 1)
        
        return occlusion

    
    
    def update_occlusion(self):
        if self.normal_map is None:
            return
        
        normals = self.normal_map
        
        light_x = self.light_x_slider.value() / 100.0
        light_y = self.light_y_slider.value() / 100.0
        light_z = self.light_z_slider.value() / 100.0
        light_dir = np.array([light_x, light_y, light_z])
        light_dir /= np.linalg.norm(light_dir)
        
        occlusion_map = self.compute_occlusion(normals, light_dir, self.height_map, self.height_slider.value() / 100.0)
        occlusion_map = np.clip(occlusion_map, 0, 1)
        # occlusion_image = Image.fromarray((occlusion_map * 255).astype(np.uint8))

        
        # qimage = QImage(occlusion_image.tobytes(), occlusion_image.width, occlusion_image.height, QImage.Format_Grayscale8)
        # pixmap = QPixmap.fromImage(qimage)
        # self.image_label.setPixmap(pixmap)
        self.occlusionMapUpdated.emit(occlusion_map)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = OcclusionApp()
    ex.show()
    sys.exit(app.exec())
