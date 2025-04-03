# ProjectPilot Deployment Guide

This document provides detailed instructions for deploying ProjectPilot to AWS Elastic Beanstalk.

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Elastic Beanstalk CLI installed (`pip install awsebcli`)
- Git
- GitHub account with access to the repository

## AWS Resources Setup

### 1. Secrets Manager

ProjectPilot uses an existing AWS Secrets Manager secret named `manager-bot-secrets`. This secret should contain:

- `SLACK_SIGNING_SECRET`: Used as application secret key
- `GITHUB_TOKEN`: For GitHub API integration
- `OPENAI_API_KEY`: For OpenAI API access
- `ANTHROPIC_API_KEY`: For Anthropic Claude API access
- `S3_BUCKET_NAME`: For file storage
- `DATABASE_URL` (optional): Production database connection string

### 2. IAM Policy for Secrets Access

Create an IAM policy to allow the Elastic Beanstalk environment to access the secrets:

```bash
aws iam create-policy \
  --policy-name ProjectPilotSecretsReadPolicy \
  --policy-document file://secrets-manager-read-policy.json
```

Attach this policy to the Elastic Beanstalk instance profile:

```bash
aws iam attach-role-policy \
  --role-name aws-elasticbeanstalk-ec2-role \
  --policy-arn <ARN_FROM_CREATE_POLICY_COMMAND>
```

## Elastic Beanstalk Deployment

### 1. Create the Application

```bash
aws elasticbeanstalk create-application \
  --application-name ProjectPilot \
  --description "AI-powered project management system"
```

### 2. Create the Environment

```bash
aws elasticbeanstalk create-environment \
  --application-name ProjectPilot \
  --environment-name project-pilot-env \
  --solution-stack-name "64bit Amazon Linux 2 v3.5.0 running Python 3.8" \
  --option-settings file://eb-options.json
```

### 3. Initialize EB CLI and Deploy

From your project directory:

```bash
eb init ProjectPilot --region eu-west-2
# Select Python platform version when prompted
eb deploy project-pilot-env
```

### 4. Verify Deployment

```bash
eb status project-pilot-env
eb health project-pilot-env
eb logs project-pilot-env
```

## GitHub Actions CI/CD

The repository includes a GitHub Actions workflow in `.github/workflows/deploy.yml` that handles:

1. Running tests
2. Building the application
3. Deploying to AWS Elastic Beanstalk

To set up GitHub Actions, add these repository secrets:

- `AWS_ACCESS_KEY_ID`: AWS access key with deployment permissions
- `AWS_SECRET_ACCESS_KEY`: Corresponding AWS secret key

## Database Setup

For production, we recommend using Amazon RDS:

1. Create a PostgreSQL database in RDS
2. Add the connection string to your Secrets Manager secret as `DATABASE_URL`
3. Ensure the Elastic Beanstalk security group can access the RDS security group

## Troubleshooting

### Common Issues

1. **Deployment Fails**: Check CloudWatch logs and EB logs (`eb logs`)
2. **Application Starts but Fails**: Verify environment variables and secrets access
3. **Database Connection Issues**: Check security groups and connection strings
4. **Permission Errors**: Verify IAM roles and policies

### Logs

Access logs through:
- AWS Console: CloudWatch Logs
- EB CLI: `eb logs`
- Application logs: `/var/log/web.stdout.log` on the EC2 instance

## Maintenance

### Updating the Application

1. Push changes to GitHub main branch to trigger automatic deployment, or
2. Use the EB CLI locally:
   ```bash
   git pull
   eb deploy
   ```

### Scaling

Modify the environment configuration in AWS Console or with:

```bash
aws elasticbeanstalk update-environment \
  --environment-name project-pilot-env \
  --option-settings Namespace=aws:autoscaling:asg,OptionName=MaxSize,Value=4
```