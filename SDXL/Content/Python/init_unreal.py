import unreal
import subprocess
from pathlib import Path
import os
import sys
from installPackages import install, run_accelerate_config

PYTHON_INTERPRETER_PATH = unreal.get_interpreter_executable_path()
assert Path(PYTHON_INTERPRETER_PATH).exists(), f"Python not found at '{PYTHON_INTERPRETER_PATH}'"
file_path = Path(PYTHON_INTERPRETER_PATH)

import pkg_resources

def Menu():
    menus = unreal.ToolMenus.get()

    # __file__ gives the path of the current file; os.path.abspath ensures it's absolute.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    install_file_path = os.path.join(current_dir, "installPackages.py")
    main_file_path = os.path.join(current_dir, "QtWindow.py")
    image_standalone_file_path = os.path.join(current_dir, "ImageStandalone.py")
    material_file_path = os.path.join(current_dir, "QtMarigold.py")
    material_standalone_file_path = os.path.join(current_dir, "MaterialsStandalone.py")


    main_menu = menus.find_menu("LevelEditor.MainMenu")
    if not main_menu:
        print("Failed to find the 'Main' menu. Something is wrong in the force!")

    # Define a function to create an entry
    def create_menu_entry(entry_name, entry_label, command_to_run):
        entry = unreal.ToolMenuEntry(
                                    name=entry_name,
                                    type=unreal.MultiBlockType.MENU_ENTRY,
                                    insert_position=unreal.ToolMenuInsert("", unreal.ToolMenuInsertType.FIRST)
        )
        entry.set_label(entry_label)
        entry.set_string_command(unreal.ToolMenuStringCommandType.PYTHON, custom_type='ExecuteFile', string=command_to_run)
        return entry

    # Create the main TextureDiffusion submenu
    submenu_name = "TextureDiffusion"
    submenu_label = "TextureDiffusion"
    texture_diffusion_submenu = main_menu.add_sub_menu("LevelEditor.MainMenu", "ToolsSection", submenu_name, submenu_label)

    # Create and add entries to the TextureDiffusion submenu
    entry1 = create_menu_entry(f"{submenu_name}.Entry1", "Install Python Dependencies", install_file_path.replace('\\', '/'))
    entry2 = create_menu_entry(f"{submenu_name}.Entry2", "Open TextureDiffusion Standalone (Editor freezes) (Recommended)", image_standalone_file_path.replace('\\', '/'))
    entry3 = create_menu_entry(f"{submenu_name}.Entry3", "Open TextureDiffusion", main_file_path.replace('\\', '/'))

    texture_diffusion_submenu.add_menu_entry("Scripts", entry1)
    texture_diffusion_submenu.add_menu_entry("Scripts", entry2)
    texture_diffusion_submenu.add_menu_entry("Scripts", entry3)

    # Create a sub-submenu under TextureDiffusion
    sub_submenu_name = "Additional Tools"
    sub_submenu_label = "Additional Tools"
    additional_tools_submenu = texture_diffusion_submenu.add_sub_menu("LevelEditor.MainMenu", submenu_name, sub_submenu_name, sub_submenu_label)

    # Create and add entries to the Additional Tools sub-submenu
    additional_entry1 = create_menu_entry(f"{sub_submenu_name}.Entry1", "Tool 1", material_file_path.replace('\\', '/'))
    additional_entry2 = create_menu_entry(f"{sub_submenu_name}.Entry2", "Tool 2", material_standalone_file_path.replace('\\', '/'))
    
    additional_tools_submenu.add_menu_entry("Scripts", additional_entry1)
    additional_tools_submenu.add_menu_entry("Scripts", additional_entry2)

    # Refresh the UI
    menus.refresh_all_widgets()

Menu()
# install()
