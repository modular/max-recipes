import sys
import honcho.manager
import subprocess
import atexit
import os
import tomli
from dotenv import load_dotenv


TASKS = [ "llama", "ui" ]


def initial_setup():
    # Read the storage location from pyproject.toml
    with open("pyproject.toml", "rb") as f:
        pyproject_data = tomli.load(f)
        data_dir = pyproject_data.get("tool", {}).get("pixi", {}).get("activation", {}).get("env", {}).get("STORAGE_LOCATION")
        if data_dir is None:
            raise ValueError("STORAGE_LOCATION not found in pyproject.toml")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Check if .env exists and create an empty file if not
    env_file = os.path.join(data_dir, ".env")
    if not os.path.exists(env_file):
        with open(env_file, "w"):
            pass  # Create empty file
        print(f"Created empty file: {env_file}")


def run():   
    load_dotenv(".env.max")  
    env = os.environ.copy()
    
    manager = honcho.manager.Manager()

    for task in TASKS:
        manager.add_process(task, f"magic run {task}", env=env)

    try:
        manager.loop()
        sys.exit(manager.returncode)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(1)


def cleanup():
    subprocess.run(["magic", "run", "clean"])


if __name__ == "__main__":
    atexit.register(cleanup)
    initial_setup()
    run()
