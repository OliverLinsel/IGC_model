##### network_module #####
#orginated by OL 10.03.2026

#This is the network module created for the IGC.NRW research project. It is subdivided into four submodules: Determining the geoscope, Creating the path template, creating the sources and creating the sinks.
#Additionally there is a visualization script to check the created data and the resulting network. The module is designed to be flexible and adaptable to different scenarios and data inputs, while also being efficient and scalable for larger datasets.

import time
import os
from setup import get_system_path

#delete after development
case_study = "igc_nrw"
data_path = r"..\data_module\Data"
output_path = r"output"
#end of delete after development

START = time.perf_counter() 

print('Execute in Directory:')
print(os.getcwd() + "\n")

def get_sinks(data_path, output_path):
    test_sinks = "Test sinks"
    return test_sinks

test_sinks_o = get_sinks(data_path = "test_data_path", output_path = "test_output_path")
print(str(test_sinks_o) + "\n")

STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')