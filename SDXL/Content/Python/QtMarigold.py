import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QCheckBox, QSpinBox, QVBoxLayout, QWidget, QHBoxLayout, QTabWidget, QSizePolicy, QSpacerItem
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from PIL import Image
from utils_Qt import addFormRow, upload_image, save_image, save_images
from utils_image import display_image
from diffusers import DiffusionPipeline
from PIL.ImageQt import ImageQt
from utils_image import calculate_normal_map, load_image, display_image, numpy_to_PIL
import numpy as np
import torch
import threading
from QtNormal import NormalMapApp
from QtOcclusion import OcclusionApp
from QtSpecular import SpecularApp
from CallbackThread import ThreadWithCallback
import os
import unreal

class MarigoldWindow(QMainWindow):
    def __init__(self, filepath):
        super().__init__()
        self.filePath = filepath
        buttonstyle = """
        QPushButton {
            background-color: #007bff;
            color: white;
            font-weight: bold;
            padding: 16px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #0056b3;
        }
    """

        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.setWindowTitle("Marigold")
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()
        self.depthImage = None
        self.normalImage = None
        self.occlusionImage = None
        self.specularImage = None
        self.image_name = None

        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 50, 300, 300)
        self.image_label.setScaledContents(True)
        self.image_label.setMaximumSize(200, 200)
        
        self.height_image_label = QLabel(self)
        self.height_image_label.setGeometry(50, 50, 300, 300)
        self.height_image_label.setScaledContents(True)
        self.image_label.setMaximumSize(200, 200)

        self.normal_image_label = QLabel(self)
        self.normal_image_label.setGeometry(50, 50, 300, 300)
        self.normal_image_label.setScaledContents(True)
        self.image_label.setMaximumSize(200, 200)

        self.occlusion_image_label = QLabel(self)
        self.occlusion_image_label.setGeometry(50, 50, 300, 300)
        self.occlusion_image_label.setScaledContents(True)
        self.image_label.setMaximumSize(200, 200)

        self.specular_image_label = QLabel(self)
        self.specular_image_label.setGeometry(50, 50, 300, 300)
        self.specular_image_label.setScaledContents(True)
        self.image_label.setMaximumSize(200, 200)

        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.setGeometry(50, 400, 150, 30)
        self.upload_button.clicked.connect(self.upload_image)

        self.input_resolution_spinbox = QSpinBox(self)
        self.input_resolution_spinbox.setMaximum(2048)
        self.input_resolution_spinbox.setMinimum(0)
        self.input_resolution_spinbox.setEnabled(False)

        self.invert_button = QPushButton("Invert Height Map", self)
        self.invert_button.clicked.connect(self.invert_image)
        
        self.match_input_resolution_checkbox = QCheckBox("Match Input Resolution", self)
        self.match_input_resolution_checkbox.setGeometry(50, 450, 200, 30)
        self.match_input_resolution_checkbox.setChecked(True)
        self.match_input_resolution_checkbox.stateChanged.connect(self.toggle_spinbox)
        self.match_input_resolution_checkbox.stateChanged.connect(self.update_color)

        self.denoising_steps_spinbox = QSpinBox(self)
        self.denoising_steps_spinbox.setGeometry(250, 450, 100, 30)
        self.denoising_steps_spinbox.setMinimum(1)
        self.denoising_steps_spinbox.setMaximum(20)
        self.denoising_steps_spinbox.setValue(6)

        self.tiling_checkbox = QCheckBox("Tiling", self)
        self.tiling_checkbox.setGeometry(50, 500, 100, 30)
        self.tiling_checkbox.setChecked(True)



        self.ensemble_steps_spinbox = QSpinBox(self)
        self.ensemble_steps_spinbox.setGeometry(250, 500, 100, 30)
        self.ensemble_steps_spinbox.setMinimum(1)
        self.ensemble_steps_spinbox.setMaximum(20)
        self.ensemble_steps_spinbox.setValue(6)

        self.process_button = QPushButton("Process", self)
        self.process_button.setGeometry(50, 550, 100, 30)
        self.process_button.clicked.connect(self.start_thread)
        self.process_button.setStyleSheet(buttonstyle)

        self.save_height_button = QPushButton("Save Generated Maps", self)
        self.save_height_button.setGeometry(200, 550, 100, 30)
        self.save_height_button.clicked.connect(self.save_images)


        #-------------------------------------------------------FORMATTING------------------------------------------------------------
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        image_layout = QHBoxLayout()

        infotabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = NormalMapApp()
        self.tab3 = OcclusionApp()
        self.tab4 = SpecularApp()
        self.tab2.normalMapUpdated.connect(self.normal_image_setup)
        self.tab3.occlusionMapUpdated.connect(self.occlusion_image_setup)
        self.tab4.specularMapUpdated.connect(self.specular_image_setup)

        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.height_image_label)
        image_layout.addWidget(self.normal_image_label)
        image_layout.addWidget(self.occlusion_image_label)
        image_layout.addWidget(self.specular_image_label)
        layout.addLayout(image_layout)
        marigold_layout = QVBoxLayout()
        marigold_layout.addItem(spacer)
        marigold_layout.addWidget(self.upload_button)
        addFormRow(marigold_layout, "Denoising Steps:", self.denoising_steps_spinbox)
        addFormRow(marigold_layout, "Ensemble Steps:", self.ensemble_steps_spinbox)
        addFormRow(marigold_layout, "Output Resolution:",self.match_input_resolution_checkbox, self.input_resolution_spinbox)
        marigold_layout.addWidget(self.tiling_checkbox)
        marigold_layout.addWidget(self.process_button)
        marigold_layout.addWidget(self.invert_button)
        marigold_layout.addWidget(self.save_height_button)
        self.tab1.setLayout(marigold_layout)

        infotabs.addTab(self.tab1, "Marigold")
        infotabs.addTab(self.tab2, "Normal Map")
        infotabs.addTab(self.tab3, "Occlusion Map")
        infotabs.addTab(self.tab4, "Specular Map")
        layout.addWidget(infotabs)
                

        self.pipeline = None
        self.image_path = None

    def upload_image(self):
        file_path = upload_image(self, self.filePath)
        load_image(file_path, self.image_label)
        self.image_name = os.path.basename(file_path)
        self.image_path = file_path
    
    def start_thread(self):
        self.process_button.setEnabled(False)
        self.process_button.setText("Processing...")
        thread = ThreadWithCallback(
        target=self.process_image,
        callback=self.stop_thread,
        args=()
        )
        thread.start()
        
    def stop_thread(self):
        print("stop thread")
        self.process_button.setEnabled(True)
        self.process_button.setText("Process")
        self.tab2.load_height_map(self.depthImage)
        # Display processed image (depth map)
        display_image(self.depthImage, self.height_image_label)
        
    def process_image(self):
        if self.image_path:
            # Load the image
            image = Image.open(self.image_path)
            targets = []
            # Load pipeline
            if self.pipeline is None:
                print("hello")
                modelFolderPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
                marigoldPath = os.path.join(modelFolderPath, "marigold-lcm-v1-0-6.20.2024")
                self.pipeline = DiffusionPipeline.from_pretrained(
                    marigoldPath,
                    custom_pipeline="marigold_depth_estimation",
                    torch_dtype=torch.float16,
                    variant="fp16"
                )
                self.pipeline.to("cuda")

                for item in self.pipeline.components:
                    if "unet" in item or "vae" in item or "text_encoder" in item:
                        module = getattr(self.pipeline, item, None)  # Attempt to retrieve variable by name
                        if module is not None:
                            targets.append(module)
            
                self.conv_layers = []
                self.conv_layers_original_paddings = []
                for target in targets:
                    for module in target.modules():
                        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
                            self.conv_layers.append(module)
                            self.conv_layers_original_paddings.append(module.padding_mode)

            # Process image with pipeline
            for cl, opad in zip(self.conv_layers, self.conv_layers_original_paddings):
                if self.tiling_checkbox.isChecked():
                    cl.padding_mode = "circular"
                else:
                    cl.padding_mode = opad
            pipeline_output = self.pipeline(
                image,
                denoising_steps=self.denoising_steps_spinbox.value(),
                ensemble_size=self.ensemble_steps_spinbox.value(),
                processing_res=self.input_resolution_spinbox.value() if self.input_resolution_spinbox.isEnabled() else 0,
                match_input_res=self.match_input_resolution_checkbox.isChecked(),
            )

            # Predicted depth map
            self.depthImage: np.ndarray = pipeline_output.depth_np        
            self.depthImage = 1.0 - self.depthImage  
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def invert_image(self):
        if self.depthImage is not None:
            self.depthImage = 1.0 - self.depthImage
            display_image(self.depthImage, self.height_image_label)

    def toggle_spinbox(self, state):
        print(state)
        if state == 2:
            self.input_resolution_spinbox.setEnabled(False)
        else:
            self.input_resolution_spinbox.setEnabled(True)
    def update_color(self, state):
        if state == Qt.Checked:
            self.match_input_resolution_checkbox.setStyleSheet("QCheckBox { color: red; }")
        else:
            self.match_input_resolution_checkbox.setStyleSheet("QCheckBox { color: black; }")
            
    def normal_image_setup(self, normal_map):
        display_image(normal_map, self.normal_image_label)
        self.normalImage = normal_map
        # Pass the QImage to load_maps method if it expects QImage
        self.tab3.load_maps(normal_map, self.depthImage)
        self.tab4.load_height_map(self.depthImage)
        print("normal image setup")

    def occlusion_image_setup(self, occlusion_map):
        self.occlusionImage = occlusion_map
        display_image(occlusion_map, self.occlusion_image_label)
        print("occlusion image setup")

    def specular_image_setup(self, specular_map):
        self.specularImage = specular_map
        display_image(specular_map, self.specular_image_label)
        print("specular image setup")

    
    def save_images(self):
        if self.image_name is not None:
            #remove .png from the image name
            name = self.image_name.split(".")[0]
        images = [self.depthImage, self.normalImage, self.occlusionImage, self.specularImage]
        PIL_images = []
        for image in images:
            npImage = numpy_to_PIL(image)
            PIL_images.append(npImage)
        names = [name + "_height_map", name + "_normal_map", name + "_occlusion_map", name + "_specular_map"]
        save_images(PIL_images, names, self)
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MarigoldWindow(unreal.Paths.project_intermediate_dir())
    window.show()
    sys.exit(app.exec())
