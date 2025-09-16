import subprocess
import sys
import os

scripts = [
    "preprocess.py",
    "division.py",
    "selector database.py"
]

def run_script(script):
    print(f"\n Ejecutando: {script}")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error ejecutando {script}:\n{result.stderr}")
    else:
        print(f"Finalizado: {script}")
        print(result.stdout)

if __name__ == "__main__":
    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f" No se encontró el script: {script}")
