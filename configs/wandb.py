from configs.globals import *

WANDB_PROJECT = 'AICROWD-SUADD'
ENTITY = None # set this to team name if working in a team
CLASSES = {i:c for i,c in enumerate([
    'water', 'asphalt', 'grass', 'human', 
    'animal', 'high_vegetation', 'ground_vehicle', 'façade',
     'wire', 'garden_furniture', 'concrete', 'roof',
      'gravel', 'soil', 'primeair_pattern', 'snow', 'unknown'])}
RAW_DATA = 'suadd'
RAW_DATA_TEST = 'suadd-test'

global DATA_NAME
if debug:
    DATA_NAME = RAW_DATA_TEST
else:
    DATA_NAME = RAW_DATA