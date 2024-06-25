from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget, QFileDialog, QLayout
from PIL import Image
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap, QImage
from PIL.ImageQt import ImageQt
from utils_image import numpy_to_PIL
import numpy as np

def addFormRow(layout, labelText, widget, optionalWidget=None, enabled=True):
        """
        Add a row to a form layout with a label and a widget.
        :param layout: The layout to add the row to.
        :param labelText: The text of the label.
        :param widget: The widget to add to the row.
        :param optionalWidget: An optional additional widget to add to the row.
        :param enabled: Whether the widget should be enabled.
        :return: The layout of the row.
        """
        rowLayout = QHBoxLayout()
        label = QLabel(labelText)
        rowLayout.addWidget(label)
        rowLayout.addWidget(widget)
        if optionalWidget:
            widget.setEnabled(enabled)
            rowLayout.addWidget(optionalWidget)
        layout.addLayout(rowLayout)
        return rowLayout

def upload_image(parent=None, filepath=None):
    """
    Opens a file dialog to select an image file and returns the selected file path.
    :param parent: QWidget that will act as the parent of this file dialog. Can be None.
    :return: The path of the selected file as a string, or None if no file is selected.
    """
    file_dialog = QFileDialog(parent)
    file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
    file_dialog.setViewMode(QFileDialog.Detail)
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    file_dialog.setDirectory(filepath)
    if file_dialog.exec():
        file_path = file_dialog.selectedFiles()[0]
        return file_path
    return None

def save_image(image, filepath, parent=None, title="Save Image"):
    """
    Opens a file dialog to save an image. 
    :param image: QImage or similar image object that has a save method.
    :param parent: QWidget that will act as the parent of this file dialog. Can be None.
    :return: None
    """
    #if image is numpy
    if isinstance(image, np.ndarray):
        image = numpy_to_PIL(image)
    if image:
        file_path, _ = QFileDialog.getSaveFileName(parent, title, filepath, "Images (*.png *.jpg *.bmp)")
        if file_path:
            image.save(file_path)
            return file_path

def save_images(image_list, name_list, parent):
    """
    Opens a file dialog to save multiple images.
    :param image_list: List of QImage or similar image objects that have a save method.
    :param name_list: List of names for the images.
    :param parent: QWidget that will act as the parent of this file dialog.
    :return: None
    """
    options = QFileDialog.Options()
    
    if len(image_list) != len(name_list):
        raise ValueError("The number of images must match the number of names.")

    for img, name in zip(image_list, name_list):
        file_dialog = QFileDialog()
        file_dialog.setOptions(options)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setDefaultSuffix('png')  # Set the default file extension
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        file_dialog.selectFile(name)  # Pre-select the file name
        file_dialog.setWindowTitle("Save " + name)

        if file_dialog.exec_():
            file_name = file_dialog.selectedFiles()[0]
            img.save(file_name)


def hide_widgets_in_layout(layout):
    """Recursively hide all widgets in a given layout."""
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget():  # Check if the item is a widget
            item.widget().hide()
        elif item.layout():  # Check if the item is a sub-layout
            hide_widgets_in_layout(item.layout()) 

def show_widgets_in_layout(layout):
    """Recursively hide all widgets in a given layout."""
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget():  # Check if the item is a widget
            item.widget().show()
        elif item.layout():  # Check if the item is a sub-layout
            hide_widgets_in_layout(item.layout()) 


class ClickableLabel(QLabel):
    """A QLabel that emits a custom signal when clicked."""
    clicked = Signal()  # Define a custom signal

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_stylesheet = """
            QLabel {
                border: 3px solid white;
            }
        """
        self.hover_stylesheet = """
            QLabel {
                border: 5px solid #007bff;;
            }
        """
        self.setStyleSheet(self.default_stylesheet)
        self.setAlignment(Qt.AlignCenter)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()  # Emit the custom signal
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self.setStyleSheet(self.hover_stylesheet)  # Change the color when hovered
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(self.default_stylesheet)  # Reset to the default color when the mouse leaves
        super().leaveEvent(event)