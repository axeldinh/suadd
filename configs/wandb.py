from configs.globals import debug

WANDB_PROJECT = 'AICROWD-SUADD'
ENTITY = None  # set this to team name if working in a team
RAW_DATA = 'suadd'
RAW_DATA_TEST = 'suadd-test'

global DATA_NAME
if debug:
    DATA_NAME = RAW_DATA_TEST
else:
    DATA_NAME = RAW_DATA
