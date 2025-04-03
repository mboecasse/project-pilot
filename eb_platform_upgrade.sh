#!/bin/bash

# ⚠️ IMPORTANT: PLATFORM UPGRADE REQUIRED ⚠️
# The current platform "Python 3.8 running on 64bit Amazon Linux 2/3.8.0" is DEPRECATED
# This script provides instructions for upgrading to a supported platform

# Instructions for upgrading Elastic Beanstalk platform

# Option 1: AWS Console Method
# 1. Go to the Elastic Beanstalk console
# 2. Select your environment (project-pilot-env-3)
# 3. Click "Change version" button in the Platform section
# 4. Select the latest Python platform version
# 5. Click "Apply"

# Option 2: AWS CLI Method
# Get latest platform version
LATEST_PLATFORM=$(aws elasticbeanstalk list-available-solution-stacks --query "SolutionStacks[?contains(@, 'running Python 3.11') && contains(@, 'Amazon Linux 2023')] | [0]" --output text)

echo "Latest available Python platform: $LATEST_PLATFORM"

# Update environment to latest platform
aws elasticbeanstalk update-environment \
  --environment-name project-pilot-env-3 \
  --solution-stack-name "$LATEST_PLATFORM"

# Option 3: EB CLI Method (Recommended)
# Run these commands from your project directory
eb use project-pilot-env-3
eb platform select
# Then select the latest Python platform version
eb deploy

# Update the platform in your configuration
cat > .ebextensions/00_environment_setup.config << 'EOL'
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: wsgi:application
    PythonVersion: 3.11

  aws:elasticbeanstalk:application:environment:
    # Python environment variables
    PYTHONPATH: "/var/app/current:$PYTHONPATH"
    FLASK_CONFIG: "production"
    AWS_REGION: "eu-west-2"
    SECRET_NAME: "manager-bot-secrets"
    
    # Rust and Cargo paths
    PATH: "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    
packages:
  yum:
    git: []
    postgresql-devel: []
    python3-devel: []
    gcc: []
    gcc-c++: []
    make: []
    openssl-devel: []
EOL

# Update the GitHub workflow to use the new platform
sed -i 's/project-pilot-env-3/project-pilot-env-3/g' .github/workflows/deploy.yml