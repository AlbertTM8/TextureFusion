import subprocess
import unreal

# Define the command as a list of arguments
command = ["accelerate", "config", "default"]

# Execute the command
try:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print("Output:", result.stdout)
    if result.stderr:
        print("Error:", result.stderr)
except subprocess.CalledProcessError as e:
    print("An error occurred while running the command.")
    print(e)
