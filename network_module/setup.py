##### setup_module #####
#orginated by OL 13.04.2026

#This is the setup module to define the base settings around the igc modules including the global functions for path handling epsg etc.

import platform
import os

#get system path and adapting it to the respective system and its notation
def get_system_path(*paths):
    if platform.system() == "Windows":
        print(str(os.path.join(*paths).replace("/", "\\")))
        return os.path.join(*paths).replace("/", "\\")
    if platform.system() == "Linux":
        print(str(os.path.join("..", *paths).replace("\\", "/")))
        return os.path.join("..", *paths).replace("\\", "/")
    else:
        return os.path.join(*paths).replace("\\", "/")
