import os

def setup_output_dir(config):
    output_dir = os.path.join(config.base_data_dir, "rom_output")
    os.makedirs(output_dir, exist_ok=True)
    config.output_dir = output_dir

def format_metadata(index, label):
    return f'#@   {index} "{label}"'.ljust(64) + '""\n'

