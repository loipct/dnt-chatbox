import os 
import yaml


# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the path to the YAML configuration file
config_file = os.path.join(script_dir, 'cfg.yaml')

database_config  = None 
llm_model_config = None 
def get_config():
    global database_config, llm_model_config
    # Load the YAML configuration file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        # Extracting configurations
    database_config = config[0]
    llm_model_config = config[1]
    
get_config()