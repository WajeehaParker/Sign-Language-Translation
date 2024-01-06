import subprocess

try:
    subprocess.run(['python', 'source/hand detection.py'], check=True)
    subprocess.run(['python', 'source/create_model.py'], check=True)
    subprocess.run(['python', 'source/predict.py'], check=True)
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
