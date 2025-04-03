#!/bin/bash

# Elastic Beanstalk Environment Status Checker

# Set the default environment name
ENV_NAME="project-pilot-env-3"

# Check if a different environment was provided
if [ $# -eq 1 ]; then
    ENV_NAME="$1"
fi

# Verify EB CLI is installed
if ! command -v eb &> /dev/null; then
    echo "❌ ERROR: EB CLI is not installed."
    echo "Install it using: pip install awsebcli"
    exit 1
fi

# Check environment status
echo "📊 Checking status for environment: $ENV_NAME (in eu-west-2 region)"

# Display environment information
echo -e "\n📝 Environment Information:"
eb status "$ENV_NAME" --region eu-west-2

# Display environment health
echo -e "\n🏥 Environment Health:"
eb health "$ENV_NAME" --region eu-west-2

# Display recent events
echo -e "\n📜 Recent Events:"
eb events --region eu-west-2 -f "$ENV_NAME" --num 10

echo -e "\n✅ Status check completed."
echo "View more details in the AWS Elastic Beanstalk console (Region: eu-west-2)"
echo "https://eu-west-2.console.aws.amazon.com/elasticbeanstalk/home?region=eu-west-2#/environment/dashboard?applicationName=ProjectPilot&environmentId=$ENV_NAME"