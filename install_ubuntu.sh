#!/bin/bash
# Ubuntu Installation Script for Bettensor MLB TUI
# Run this script to install all required dependencies

echo "🚀 Installing Bettensor MLB TUI Dependencies on Ubuntu"
echo "=" * 60

# Update package list
echo "📦 Updating package list..."
sudo apt update

# Install Python and pip if not already installed
echo "🐍 Installing Python 3 and pip..."
sudo apt install -y python3 python3-pip python3-venv

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev

# Install PostgreSQL (required for the TUI database)
echo "🗄️ Installing PostgreSQL..."
sudo apt install -y postgresql postgresql-contrib

# Install required Python packages
echo "📚 Installing Python packages..."
pip3 install --upgrade pip

# Core dependencies
pip3 install bittensor
pip3 install psycopg2-binary
pip3 install prompt-toolkit
pip3 install rich

# Additional dependencies from requirements.txt
pip3 install numpy pandas scikit-learn
pip3 install requests aiohttp
pip3 install python-dotenv
pip3 install joblib

echo "✅ Installation complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Set up PostgreSQL database"
echo "2. Configure database connection"
echo "3. Run the MLB TUI interface"
echo ""
echo "Run: ./setup_postgres.sh to set up the database"
