#!/bin/bash

# Elastic Beanstalk Platform Upgrade Script

# Ensure you're in the project root directory
echo "📋 Preparing for Elastic Beanstalk Platform Upgrade..."

# Verify EB CLI is installed
if ! command -v eb &> /dev/null; then
    echo "❌ ERROR: EB CLI is not installed."
    echo "Install it using: pip install awsebcli"
    exit 1
fi

# Check git status (avoid upgrading with uncommitted changes)
if [[ -n $(git status -s) ]]; then
    echo "⚠️ WARNING: You have uncommitted changes."
    git status
    read -p "Do you want to continue? (y/N) " confirm
    if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
        echo "Upgrade cancelled."
        exit 1
    fi
fi

# Initialize EB application (if not already done)
echo "🌐 Setting AWS region to eu-west-2..."
eb init --region eu-west-2

# List current environments
echo "🔍 Current Elastic Beanstalk Environments:"
eb list

# Confirm environment name
read -p "Enter the environment name to upgrade (default: project-pilot-env-3): " ENV_NAME
ENV_NAME=${ENV_NAME:-project-pilot-env-3}

# Select new platform
echo "🚀 Selecting new platform version..."
echo "Upgrading to: Python 3.13 running on 64bit Amazon Linux 2023"
eb platform select --platform "python-3.13" --region eu-west-2

# Deploy to the selected environment
echo "📦 Deploying to $ENV_NAME..."
eb use "$ENV_NAME"
eb deploy

# Verify deployment status
echo "🕵️ Checking environment health..."
eb health "$ENV_NAME"

# Open environment in browser
read -p "Do you want to open the environment in a web browser? (y/N) " open_browser
if [[ $open_browser == [yY] || $open_browser == [yY][eE][sS] ]]; then
    eb open
fi

echo "✅ Platform upgrade process completed!"
echo "IMPORTANT: Verify your application functionality thoroughly after the upgrade."