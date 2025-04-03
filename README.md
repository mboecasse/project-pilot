# ProjectPilot - AI-powered Project Management System

ProjectPilot is a comprehensive project management system that leverages AI capabilities to help teams plan, track, and execute projects more efficiently. The system provides an integrated framework for project management with intelligent worker bots powered by multiple AI providers.

## Features

- **Project Management**: Create, track, and manage projects with detailed progress tracking
- **Task Management**: Break down projects into manageable tasks with dependencies, priorities, and assignments
- **AI Worker Bots**: Deploy specialized AI bots for different project roles (architect, developer, tester, devops)
- **Multi-AI Integration**: Seamless integration with OpenAI, Anthropic Claude, and AWS Bedrock
- **Authentication System**: Secure user authentication and authorization
- **RESTful API**: Comprehensive API for integration with other systems
- **Analytics Dashboard**: Visual insights into project progress and performance
- **Responsive UI**: Clean, modern interface for desktop and mobile devices

## Technology Stack

- **Backend**: Python with Flask framework
- **Database**: SQLAlchemy ORM with PostgreSQL (production) or SQLite (development)
- **Authentication**: Flask-Login for session management
- **Migration**: Flask-Migrate for database schema evolution
- **AWS Integration**: Boto3 for AWS services (Bedrock, Secrets Manager)
- **Deployment**: AWS Elastic Beanstalk for production environments
- **CI/CD**: GitHub Actions for automated testing and deployment

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- AWS account (for production deployment and Bedrock access)
- API keys for OpenAI and/or Anthropic (optional)

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/project-pilot.git
   cd project-pilot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (create a `.env` file):
   ```
   FLASK_APP=wsgi.py
   FLASK_ENV=development
   SECRET_KEY=your-secret-key-here
   DATABASE_URL=sqlite:///dev-projectpilot.db
   OPENAI_API_KEY=your-openai-api-key  # Optional
   ANTHROPIC_API_KEY=your-anthropic-api-key  # Optional
   ```

5. Initialize the database:
   ```bash
   flask db init
   flask db migrate -m "Initial migration"
   flask db upgrade
   ```

6. Run the application:
   ```bash
   python wsgi.py
   ```

The application will be available at `http://localhost:8000`.

### AWS Deployment

#### Elastic Beanstalk Setup

1. Install the AWS CLI and EB CLI:
   ```bash
   pip install awscli awsebcli
   ```

2. Configure AWS credentials:
   ```bash
   aws configure
   ```

3. Initialize Elastic Beanstalk:
   ```bash
   eb init -p python-3.11 ProjectPilot
   ```

4. Create an environment:
   ```bash
   eb create project-pilot-env
   ```

5. Set environment variables:
   ```bash
   eb setenv FLASK_CONFIG=production SECRET_NAME=manager-bot-secrets AWS_REGION=eu-west-2
   ```

#### Platform Upgrade (Important)

If you're using a deprecated platform (Python 3.8 on Amazon Linux 2), you need to upgrade to a supported platform:

1. Run the included upgrade script:
   ```bash
   ./eb_platform_upgrade.sh
   ```

2. Or manually upgrade using the EB CLI:
   ```bash
   eb platform select
   # Select "Python 3.11 running on 64bit Amazon Linux 2023"
   eb deploy
   ```

The application is configured to use Python 3.11 on Amazon Linux 2023, which provides:
- Improved performance
- Better security
- Longer support lifecycle
- Access to newer Python packages

#### AWS Secrets Manager Setup

ProjectPilot uses the existing `manager-bot-secrets` secret in AWS Secrets Manager. This secret should contain the following key-value pairs:

```json
{
  "SLACK_SIGNING_SECRET": "used-as-application-secret-key",
  "DATABASE_URL": "postgresql://username:password@your-rds-instance.region.rds.amazonaws.com:5432/database",
  "OPENAI_API_KEY": "your-openai-api-key",
  "ANTHROPIC_API_KEY": "your-anthropic-api-key",
  "GITHUB_TOKEN": "your-github-token",
  "S3_BUCKET_NAME": "your-s3-bucket-name"
}
```

For more detailed deployment information, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Architecture

ProjectPilot follows a modular architecture with the following components:

- **App Factory**: Flask application factory pattern for flexible configuration
- **Blueprints**: Modular routes organized by function (auth, projects, tasks, API)
- **Models**: SQLAlchemy models for data persistence
- **Services**: Integration with external AI services
- **Worker Bots**: Specialized AI agents for different project roles
- **Templates**: Jinja2 templates for rendering HTML views

## AI Integration

ProjectPilot integrates with multiple AI providers through a unified interface:

- **OpenAI**: GPT models for natural language tasks
- **Anthropic Claude**: Claude models for complex reasoning
- **AWS Bedrock**: Managed foundation models from multiple providers

The `three_ai_workflow.py` module orchestrates these integrations, providing redundancy and optimized selection based on task requirements.

## Project Structure

```
projectpilot/
├── .ebextensions/              # Elastic Beanstalk configuration
├── .github/workflows/          # GitHub Actions workflows
├── app/                        # Main application package
│   ├── models/                 # Database models
│   ├── routes/                 # Route blueprints
│   ├── services/               # External service integrations
│   ├── bots/                   # Worker bot implementations
│   ├── static/                 # Static assets
│   ├── templates/              # HTML templates
│   └── utils/                  # Utility functions
├── tests/                      # Test suite
├── migrations/                 # Database migrations
├── wsgi.py                     # WSGI entry point
└── requirements.txt            # Python dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT models
- Anthropic for Claude models
- AWS for Bedrock and infrastructure services
- Flask and SQLAlchemy communities for excellent frameworks