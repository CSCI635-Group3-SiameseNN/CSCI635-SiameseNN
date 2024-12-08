import os
import kagglehub

# Move one directory up from the current directory
os.chdir("..")

# Download the AT&T dataset here (in the parent directory)
path = kagglehub.dataset_download("kasikrit/att-database-of-faces")
print("Path to dataset files:", path)