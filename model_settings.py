### This is the central definition file for all model aspects for better overview
### Default parameters as well as custom settings are defined. Only change default parameters if you know what you are doing.

import os

print('Execute in Directory:')
print(os.getcwd() + "\n")

# Get the directory where this script is located
script_dir = os.getcwd()

### centrally define case study, can be overwritten by calling get_settings with another case_study argument
case_study = "h2bb"
 
### Define the default parameters ###
defaults = {
    "case_study": "default",# return case study name if required for e.g. visualizations
    "transport_costs": 30,  # €/MWh
    "commodity_price": 50,  # €/MWh
}

### Define case study specific parameters ###
### Feel free to add case study parameters at will, ideally also provide a default value
case_studies = {
    "h2bb": {
        "case_study": "h2bb",   # return case study name if required for e.g. visualizations
        "transport_costs": 25,  # €/MWh
        "commodity_price": 45,  # €/MWh
    },
    "igc_nrw": {
        "case_study": "igc_nrw",# return case study name if required for e.g. visualizations
        "transport_costs": 35,  # €/MWh
        "commodity_price": 55,  # €/MWh
    },
}

def get_settings(case_study_arg=None, parameter=None):
    # Use the case study argument to get the respective parameter from the dictionary
    # If case_study_arg is not given, the centrally defined case_study from this file will be used
    # If parameter is None, return the entire settings dictionary for the case study
    # If parameter is provided, return the specific parameter for the case study

    # Determine which case study to use
    current_case_study = case_study_arg if case_study_arg is not None else case_study

    # Get the settings for the current case study, or use defaults if the case study is not defined
    settings = case_studies.get(current_case_study, defaults.copy())

    if parameter is None:
        return settings
    else:
        # Check if the parameter exists in either the case study settings or the defaults
        if parameter in settings or parameter in defaults:
            return settings.get(parameter, defaults.get(parameter))
        else:
            # Return an error code if the parameter is not found in either dictionary
            return print("Error: -1 Parameter not found: check if case_study is correctly defined or default value provided in model_settings.py")  # Error code for parameter not found
    

# Example usage:
if __name__ == "__main__":
    # Get the entire settings dictionary for the current case study
    print(get_settings())

    # Get a specific parameter for the current case study
    print(get_settings(parameter="transport_costs"))

    # Get a specific parameter for a different case study
    print(get_settings(case_study_arg="other_case", parameter="commodity_price"))

    # Try to get a parameter that is not defined in either the case study settings or the defaults
    print(get_settings(parameter="undefined_parameter"))

#If you want to build your own script and include the model_settings.py you can do it like this:
# # Get the directory where this script is located
# script_dir = os.getcwd() #os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(script_dir)
# sys.path.append(parent_dir)
# from model_settings import get_settings