
import os
import subprocess


def run_simulation(dem_fname: str, init_fname: str, vent_fname: str, sim_index: int) -> None:   
    # sim_index is useful for indexing directories and creating unique names
    
    out_dir = "runs/run-" + str(sim_index) + "-dir"

    if os.path.isdir(out_dir):
        print("Directory " + out_dir + " already exists: skipping simulation")
        return

    gpuflow = "../../gpuflow/gpuflow"
    subprocess.run([gpuflow, dem_fname, init_fname, vent_fname, 'state', '--single-state', '-d', out_dir], stdout = subprocess.DEVNULL) 


if __name__  == "__main__": 
    pass 
