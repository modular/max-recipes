import sys
import honcho.manager
import subprocess
import atexit
import os
import tomli
from dotenv import load_dotenv


TASKS = [ "llama", "ui" ]


def run_tasks():
    """
    Runs the tasks specified in the TASKS list. This includes loading environment
    variables, setting up the task manager, and starting the tasks.
    """   
    try:
        load_dotenv(".env.max")  
        env = os.environ.copy()
        
        manager = honcho.manager.Manager()

        for task in TASKS:
            manager.add_process(task, f"magic run {task}", env=env)

        manager.loop()
        sys.exit(manager.returncode)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def initial_setup():
    """
    Initializes persistent storage for AnythingLLM. Reads storage location from
    the pyproject.toml file. If the directory and/or .env file don't already exist,
    it creates the directory and ensures an empty .env file is present within it.
    """

    with open("pyproject.toml", "rb") as f:
        pyproject_data = tomli.load(f)
        data_dir = (
            pyproject_data.get("tool", {})
                .get("pixi", {})
                .get("activation", {})
                .get("env", {})
                .get("UI_STORAGE_LOCATION")
        )
        if data_dir is None:
            raise ValueError("UI_STORAGE_LOCATION not found in pyproject.toml")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    env_file = os.path.join(data_dir, ".env")
    if not os.path.exists(env_file):
        open(env_file, "w").close()  # Create empty file
        print(f"Created empty file: {env_file}")


def cleanup():
    """Checks if the `clean` task exists in pyproject.toml and runs it."""

    with open("pyproject.toml", "rb") as f:
        pyproject_data = tomli.load(f)
        tasks = pyproject_data.get("tool", {}).get("pixi", {}).get("tasks", {})
        
        if "clean" in tasks:
            print("Running cleanup task...")
            subprocess.run(["magic", "run", "clean"])
        else:
            print("Cleanup task not found in pyproject.toml")


if __name__ == "__main__":
    atexit.register(cleanup)
    initial_setup()
    run_tasks()
