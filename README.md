# uitWiki Chatbot Service by Hien and Hai 😎

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