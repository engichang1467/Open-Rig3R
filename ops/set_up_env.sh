#!/bin/bash

# Check if uv is installed
if command -v uv >/dev/null 2>&1; then
    echo "uv is already installed."
else
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Check if the virtual environment directory exists
if [ -d ".venv" ]; then
    echo "Virtual environment '.venv' already exists. Skipping creation."
else
    echo "Creating Python 3.11 uv virtual environment..."
    uv venv --python python3.11
fi

echo "Setup complete! To activate the environment, run:"
echo "  source .venv/bin/activate"





# curl -LsSf https://astral.sh/uv/install.sh | sh


# uv venv --python python3.11