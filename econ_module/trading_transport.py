##### trading_transport_module #####
#orginated by OL 22.07.2026
#%%
### This is the script to use methods developed from the network module to determine the transport costs between any given combination of input countries and different transport commodities ###

import time
import os
import sys
from model_settings import get_settings

START = time.perf_counter()

print('Execute in Directory:')
print(os.getcwd() + "\n")

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths
try:
    data_path = os.path.join(script_dir, "data")
    output_path = os.path.join(script_dir, "output")
except:
    data_path = os.path.join(script_dir, "data")
    output_path = os.path.join(script_dir, "output")

STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')