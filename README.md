# uitWiki Chatbot Service 🤖

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://semver.org) [![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com)

## Overview

uitWiki-Chatbot is an intelligent assistant designed to help University of Information Technology (UIT) students access university-related information efficiently. Using natural language processing, it provides accurate responses to queries about academic matters, registration procedures, tuition fees, and other university-related topics.

### Key Features

- 🔍 Natural language query processing
- 📚 Comprehensive university information database
- 💾 Conversation history tracking
- 📄 PDF document management for knowledge base
- 🔄 Real-time response generation
- 🌐 Multi-language support (Vietnamese/English)

## Technology Stack

- **Backend**: FastAPI, Python 3.12
- **Database**: MongoDB, Redis
- **AI/ML**: LangChain, Google Generative AI
- **Vector Store**: Pinecone
- **Container**: Docker
- **Package Manager**: UV

## Prerequisites

- Python 3.12+
- Docker and Docker Compose
- UV Package Manager
- Redis

## Installation

1. **Install UV**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone the Repository**
```bash
git clone https://github.com/yourusername/uitWiki-chatbot-service.git
cd uitWiki-chatbot-service
```

3. **Install Dependencies**
```bash
uv sync
```

4. **Setup Pre-commit Hooks**
```bash
uv run pre-commit install
```

5. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Running the Application
### Using Docker (Recommended)
```bash
docker compose up
```

### Using Local Development Server
1. **Start Redis Server**
```bash
docker run -d -p 6379:6379 redis
```

2. **Start the FastAPI Server**
```bash
uv run fastapi dev
```

## Testing
```bash
uv run pytest tests/test_questions.py
```

## API Documentation
Once running, access the API documentation at:
- Swagger UI: http://localhost:8000/docs

## Project Structure
```bash
uitWiki-chatbot-service/
├── app/
│   ├── api/         # API routes
│   ├── core/        # Core functionality
│   ├── services/    # Business logic
│   └── static/      # Static files
├── tests/           # Test suite
└── docker/          # Docker configuration
```

## Contributing
- Fork the repository
- Create your feature branch (`git checkout -b feature/amazing-feature`)
- Commit your changes (`git commit -m 'Add some amazing feature'`)
- Push to the branch (`git push origin feature/amazing-feature`)
- Open a Pull Request

## Core Team
- Đoàn Anh Hiển - Lead Developer
- Đào Gia Hải - Developer

## Guide
Feel free to insert your documents into the knowledge base. Here are some recommended document types:

### Academic Documents
- 📚 Course syllabi and learning outcomes
- 📝 Exam regulations and procedures
- 🎓 Graduation requirements
- 📅 Academic calendar and important dates
- 🔬 Laboratory safety guidelines

### Administrative Documents
- 📋 Student registration procedures
- 💰 Tuition and fee schedules
- 🏢 Department contact information
- 🎯 Program-specific requirements
- 📑 Forms and templates

### Student Life Documents
- 🏫 Campus facilities guide
- 🎭 Club and organization guidelines
- 🏆 Scholarship information
- 🌟 Internship opportunities
- 🎪 Event planning procedures

### Technical Guidelines
- 💻 IT services documentation
- 🔑 Account management procedures
- 📱 Student portal guides
- 🌐 Online learning resources
- 🛠️ Software installation guides

### Best Practices for Document Preparation
1. Ensure documents are in PDF format
2. Include clear titles and version numbers
3. Add relevant metadata (author, department, date)
4. Structure content with clear headings
5. Update documents regularly to maintain accuracy

For assistance with document uploading or formatting, please contact the development team.

## Appreciation Post – A Journey of 6+ Months
After more than 6 months of countless hours, sleepless nights, and relentless debugging, we’re beyond excited to see uitWiki-chatbot-service finally come to life! A huge shout-out to the core team, Đào Gia Hải & Đoàn Anh Hiển, for their insane dedication, problem-solving wizardry, and unwavering commitment to making this happen. From battling unexpected errors to optimizing retrieval pipelines, every challenge was met with determination. Thanks to everyone who supported us along the journey. Onward and upward!

## License 
This project is licensed under the MIT License - see the LICENSE file for details.