##### setup_module #####
#orginated by OL 13.04.2026

#This is the setup module to define the base settings around the igc modules including the global functions for path handling epsg etc.

import platform
import os
import psutil

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

def get_ram_usage(iteration):
    # Get the current process
    process = psutil.Process()
    # Get the memory info
    mem_info = process.memory_info()
    # Get the total physical memory
    total_mem = psutil.virtual_memory().total
    # Calculate RAM usage in MB
    ram_usage_mb = mem_info.rss / (1024 * 1024)
    # Calculate RAM usage as a percentage of total memory
    ram_usage_percentage = (mem_info.rss / total_mem) * 100
    # Print the RAM usage information
    print(f"Iteration {iteration + 1}: RAM usage is {ram_usage_mb:.2f} MB ({ram_usage_percentage:.2f}% of total memory)")
    # Return the RAM usage values if needed for further processing
    return