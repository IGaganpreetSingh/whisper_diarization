#!/bin/bash

# Update package list and install prerequisites
apt-get update
apt-get install -y software-properties-common

# Add the deadsnakes PPA repository for newer Python versions
add-apt-repository ppa:deadsnakes/ppa
apt-get update

# Install Python 3.10 and necessary packages
apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils

# Remove existing symlinks for python and python3 if they exist
update-alternatives --remove-all python 2>/dev/null
update-alternatives --remove-all python3 2>/dev/null

# Set Python 3.10 as the default python and python3
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Set Python 3.10 as the default choice for both python and python3
update-alternatives --set python /usr/bin/python3.10
update-alternatives --set python3 /usr/bin/python3.10

# Install pip for Python 3.10
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.10 get-pip.py

# Set up pip and pip3 symlinks for Python 3.10
update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.10 1
update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.10 1

# Set pip for Python 3.10 as the default pip
update-alternatives --set pip /usr/local/bin/pip3.10
update-alternatives --set pip3 /usr/local/bin/pip3.10

# Verify the installation
echo "Python version set to:"
python --version

echo "Python3 version set to:"
python3 --version

echo "pip version set to:"
pip --version

echo "pip3 version set to:"
pip3 --version


git clone https://github.com/IGaganpreetSingh/whisper_diarization.git
cd whisper-diarization
apt update && sudo apt install -y cython3 ffmpeg
pip install -r requirements.txt
pip install python-multipart fastapi uvicorn
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip uninstall huggingface_hub -y
pip install huggingface-hub==0.20.3

