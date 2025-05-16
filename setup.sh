#!/bin/bash

# Install tmux
echo "Installing tmux..."
sudo yum install -y tmux

# Install boost
echo "Installing boost..."
sudo yum install -y boost boost-devel

# Install htop
echo "Installing htop..."
sudo yum install -y htop

# Download and install Miniconda
echo "Downloading Miniconda installer..."
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Check if download was successful
if [ ! -f Miniconda3-latest-Linux-x86_64.sh ]; then
    echo "Failed to download Miniconda installer"
    exit 1
fi

# Install Miniconda
echo "Installing Miniconda..."
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Remove the installer
echo "Removing installer..."
rm Miniconda3-latest-Linux-x86_64.sh

# Initialize conda for bash
echo "Initializing conda..."
source $HOME/miniconda3/bin/activate

# Create conda environment from environment.yaml
echo "Creating conda environment..."
conda env create -n yndpd -f environment.yaml

# Install ipykernel in the new environment
echo "Installing Jupyter kernel..."
conda activate yndpd
python -m ipykernel install --user --name=yndpd

# Compile PrivBayes
echo "Compiling PrivBayes..."
cd ydnpd/harness/synthesis/privbayes

# Run make
echo "Running make..."
make

# Check if compilation was successful
if [ ! -f privBayes.bin ]; then
    echo "Error: Compilation failed - privBayes.bin was not created"
    exit 1
fi

# Create linux_bin directory if it doesn't exist
mkdir -p linux_bin

# Copy the binary
echo "Copying privBayes.bin to linux_bin directory..."
cp privBayes.bin linux_bin/

# Check if copy was successful
if [ $? -eq 0 ]; then
    echo "Successfully copied privBayes.bin to linux_bin directory"
else
    echo "Error: Failed to copy privBayes.bin"
    exit 1
fi

echo "Setup complete! To activate the environment, use: conda activate yndpd"
