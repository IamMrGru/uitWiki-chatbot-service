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
# Linux/macOS
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
```

```console
# Windows (Command Prompt)
echo GOOGLE_API_KEY=your_google_api_key_here > .env
```

```console
# Windows (PowerShell)
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
```

Run the project
```console
$ uv run fastapi dev
```