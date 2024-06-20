import unreal
import subprocess
from pathlib import Path
import os
import sys
from installPackages import install, run_accelerate_config

PYTHON_INTERPRETER_PATH = unreal.get_interpreter_executable_path()
assert Path(PYTHON_INTERPRETER_PATH).exists(), f"Python not found at '{PYTHON_INTERPRETER_PATH}'"
file_path = Path(PYTHON_INTERPRETER_PATH)
# parent_dir = file_path.parent
# sitepackages = os.path.join(parent_dir, "Lib")
# sitepackages = os.path.join(sitepackages, "site-packages")
# sys.path.append(sitepackages)

    
import pkg_resources

def Menu():
    menus = unreal.ToolMenus.get()

    # __file__ gives the path of the current file; os.path.abspath ensures it's absolute.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    install_file_path = os.path.join(current_dir, "installPackages.py")
    main_file_path = os.path.join(current_dir, "QtWindow.py")
    standalone_file_path = os.path.join(current_dir, "standalone.py")

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

    # Create the submenu
    submenu_name = "StableDiffusionTool"
    submenu_label = "Stable Diffusion Tool"
    submenu = main_menu.add_sub_menu("LevelEditor.MainMenu", "ToolsSection", submenu_name, submenu_label)

    # Create and add three entries to the submenu
    entry1 = create_menu_entry(f"{submenu_name}.Entry1", "Install Dependencies", install_file_path.replace('\\', '/'))
    entry2 = create_menu_entry(f"{submenu_name}.Entry2", "Open Tool Parallel", main_file_path.replace('\\', '/'))
    entry3 = create_menu_entry(f"{submenu_name}.Entry3", "Open Tool Standalone (Editor freezes to save VRAM)", standalone_file_path.replace('\\', '/'))

    submenu.add_menu_entry("Scripts", entry1)
    submenu.add_menu_entry("Scripts", entry2)
    submenu.add_menu_entry("Scripts", entry3)

    # Refresh the UI
    menus.refresh_all_widgets()

Menu()
# install()
