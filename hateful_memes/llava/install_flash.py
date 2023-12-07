import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "flash-attn",  "--no-build-isolation"])