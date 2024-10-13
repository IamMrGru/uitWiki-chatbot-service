# uitWiki Chatbot Service by Hien and Hai 😎

[![Build Status](https://img.shields.io/travis/com/yourusername/projectname.svg)](https://travis-ci.com/yourusername/projectname)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://semver.org)

### Project: uitWiki-Chatbot

uitWiki-Chatbot is designed to provide students with a seamless way to interact with information about the University of Information Technology (UIT). The chatbot can answer questions related to academic matters, registration, tuition, and other university-related topics. It streamlines access to important information by allowing students to query data in natural language and get accurate responses.

**Purpose:**
The uitWiki-Chatbot is aimed at UIT students who need quick access to university-related information. It provides a user-friendly interface for asking questions, and it stores and retrieves conversation history for reference. It also supports an admin portal where lecturers can manage and upload PDF documents to update the system’s knowledge base.

This project is ideal for:

- UIT students seeking quick answers about the university.
- Administrators or lecturers who need to maintain and update relevant documents for student queries.

## How to run the project

Install the dependencies

```console
$ uv sync
```

Install the pre-commit hooks

```console
$ uv run pre-commit install
```

Create .env file and fill in the necessary information

```console
$ touch .env
```

Run the project with docker

```console
$ docker compose up
```

Run the project with uvicorn

1. Start the redis server

```console
$ docker run -d -p 6379:6379 redis
```

2. Run the project

```console
$ uv run fastapi dev
```

## Testing

Run the tests

```console
$ uv run pytest tests/test_questions.py
```
