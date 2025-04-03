"""
ProjectPilot - AI-powered project management system
AWS Secrets Manager utilities.
"""

import boto3
import json
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def get_secret(secret_name="manager-bot-secrets", region_name="eu-west-2"):
    """
    Retrieve a secret from AWS Secrets Manager
    
    Args:
        secret_name: Name of the secret in AWS Secrets Manager
        region_name: AWS region where the secret is stored
        
    Returns:
        dict: The secret values as a dictionary
    """
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        logger.error(f"Error retrieving secret {secret_name}: {str(e)}")
        # Fall back to environment variables in development
        if 'dev' in region_name.lower():
            logger.warning("Using fallback development credentials")
            return {}
        raise

    # Decode and parse the secret
    if 'SecretString' in get_secret_value_response:
        secret = get_secret_value_response['SecretString']
        return json.loads(secret)
    else:
        logger.error("Secret is not a string")
        return {}