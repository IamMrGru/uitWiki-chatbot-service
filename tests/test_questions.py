import pytest
import csv
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)


def load_questions_from_csv(file_path):
    questions = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            questions.append(row["question"])
    return questions


questions_data = load_questions_from_csv('tests/questions.csv')

API_ENDPOINT = f"{settings.API_V1_STR}/chat_bot/send_message"
CSV_FILE_PATH = 'tests/results.csv'

results_df = {'Câu hỏi': [], 'Câu trả lời': []}


@pytest.mark.parametrize("question", questions_data)
def test_question_response(question):
    data = {"user_question": question}

    response = client.post(API_ENDPOINT, json=data)

    if response.status_code == 200:
        response_data = response.json()
    else:
        response_data = {"error": f"Error: {response.status_code}"}

    results_df['Câu hỏi'].append(question)
    results_df['Câu trả lời'].append(response_data.get("response", response_data.get(
        "error", "No response")).replace('\n', ' ').replace('\r', ''))

    assert response.status_code == 200
    assert "response" in response.json()


@pytest.fixture(scope='session', autouse=True)
def pytest_sessionfinish(request):
    def create_csv():
        print('Writing results to CSV file...')
        dataFrame = pd.DataFrame(results_df)
        dataFrame.to_csv(CSV_FILE_PATH, sep=";", index=False, encoding='utf-8')
        print(f"Results saved to {CSV_FILE_PATH}")
    request.addfinalizer(create_csv)
