#!/bin/bash

# Step 1: Navigate to /workspace
cd /workspace || exit 1

# Step 2: Clone the repository
git clone https://github.com/zjir/gnidart.git

# Step 3: Navigate into the cloned repository
cd gnidart || exit 1

# Step 7: Create data directory
mkdir -p data

# Step 8: Update apt and install unzip
sudo apt-get update && sudo apt-get install -y unzip

# Step 4: Create Python virtual environment
python3 -m venv tlob-env

# Step 5: Activate the virtual environment
source ./tlob-env/bin/activate

# Step 6: Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

