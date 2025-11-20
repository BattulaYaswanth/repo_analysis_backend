# AI Developer Productivity Platform Backend

## Overview
Backend service for an AI Developer Productivity Platform designed to enhance developer workflows and productivity.

## Getting Started

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation
```bash
git clone <repository-url>
cd ai-developer-productivity-platform-backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Server
```bash
python app.py
# or
flask run
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `POST /api/auth/refresh` - Refresh authentication token

### User Profile
- `GET /api/users/profile` - Get user profile
- `PUT /api/users/profile` - Update user profile
- `DELETE /api/users/profile` - Delete user account

### Projects
- `GET /api/projects` - List all projects
- `POST /api/projects` - Create new project
- `GET /api/projects/<id>` - Get project details
- `PUT /api/projects/<id>` - Update project
- `DELETE /api/projects/<id>` - Delete project

### AI Features
- `POST /api/ai/suggestions` - Get code suggestions
- `POST /api/ai/analyze` - Analyze code
- `POST /api/ai/generate` - Generate code snippets

## Technology Stack
- Flask/FastAPI
- Python
- PostgreSQL/MongoDB

## License
MIT
