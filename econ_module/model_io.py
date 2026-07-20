#### This is the model_io file to give easy handling options toother modules for saving and loading of the market models and results ####

import os
import xarray as xr
import json
import linopy

#### Define the base functionalities to save and load model runs ####

def load_model_run(output_path, name):
    run_dir = os.path.join(output_path, "model", name)
    data_1D = xr.open_dataset(os.path.join(run_dir, "input_1D.nc"))
    data_2D = xr.open_dataset(os.path.join(run_dir, "input_2D.nc"))
    sol = xr.open_dataset(os.path.join(run_dir, "solution.nc"))
 
    meta_fp = os.path.join(run_dir, "metadata.json")
    meta = None
    if os.path.exists(meta_fp):
        with open(meta_fp) as f:
            meta = json.load(f)
 
    return data_1D, data_2D, sol, meta

def save_model_run(output_path, data_1D, data_2D, sol_ds, name, meta=None):
    run_dir = os.path.join(output_path, "model", name)
    os.makedirs(run_dir, exist_ok=True)
 
    data_1D.to_netcdf(os.path.join(run_dir, "input_1D.nc"), encoding={v: {"zlib": True} for v in data_1D.data_vars})
    data_2D.to_netcdf(os.path.join(run_dir, "input_2D.nc"), encoding={v: {"zlib": True} for v in data_2D.data_vars})
    sol_ds.to_netcdf(os.path.join(run_dir, "solution.nc"), encoding={v: {"zlib": True} for v in sol_ds.data_vars})
 
    if meta:
        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str) 
    return run_dir

def save_complete_model(model, output_path, name):
    run_dir = os.path.join(output_path, "model", name)
    os.makedirs(run_dir, exist_ok=True)
    model.to_netcdf(os.path.join(run_dir, "model.nc"))
    return run_dir

def load_complete_model(output_path, name):
    run_dir = os.path.join(output_path, "model", name)
    model = linopy.read_netcdf(os.path.join(run_dir, "model.nc"))
    return model