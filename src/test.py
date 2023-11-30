import os
from nocturne.envs.wrappers import create_env
from pyvirtualdisplay import Display
import yaml

def set_display_window():
    """Set a virtual display for headless machines."""
    if "DISPLAY" not in os.environ:
        disp = Display()
        disp.start()
set_display_window()
cfg_path = "/home/cdpg/chris/ditto-nocturne/nocturne/cfgs/config.yaml"
data = {}
with open(cfg_path, 'r') as file:
    # Load the YAML file
    data = yaml.safe_load(file)
env = create_env(data)
env.reset()
img = env.render()
print(img)