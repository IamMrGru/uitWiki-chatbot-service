import csv

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)

# Đọc dữ liệu từ file CSV mới (gồm các cột user_input, reference_contexts, reference)


def load_questions_from_csv(file_path):
    questions = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='}')
        for row in reader:
            # Lấy cả ba cột cần thiết từ file CSV
            user_input = row["user_input"]
            reference_contexts = row["reference_contexts"]
            reference = row["reference"]
            questions.append((user_input, reference_contexts, reference))
    return questions


# Đọc dữ liệu từ file CSV chứa các câu hỏi
questions_data = load_questions_from_csv('tests/questions252.csv')

API_ENDPOINT = f"{settings.API_V1_STR}/chat_bot/send_message"
CSV_FILE_PATH = 'tests/results.csv'

# Khởi tạo dictionary kết quả
results_df = {'user_input': [], 'reference_contexts': [],
              'reference': [], 'response': [], 'retrieved_contexts': []}

# Sử dụng pytest.mark.parametrize để kiểm tra từng câu hỏi


@pytest.mark.parametrize("user_input, reference_contexts, reference", questions_data)
def test_question_response(user_input, reference_contexts, reference):
    data = {"user_question": user_input}

    response = client.post(API_ENDPOINT, json=data)

    if response.status_code == 200:
        response_data = response.json()
    else:
        response_data = {"error": f"Error: {response.status_code}"}

    # Thêm câu hỏi, câu trả lời và thông tin reference vào kết quả
    results_df['user_input'].append(user_input)
    results_df['reference_contexts'].append(reference_contexts)
    results_df['reference'].append(reference)
    results_df['response'].append(response_data.get("response", response_data.get(
        "error", "No response")).replace('\n', ' ').replace('\r', ''))
    results_df['retrieved_contexts'].append(
        response_data.get("retrieved_contexts", "No response"))

    # Kiểm tra mã phản hồi của API
    assert response.status_code == 200
    assert "response" in response.json()

# Hàm pytest fixture để ghi kết quả vào file CSV sau khi test hoàn thành


@pytest.fixture(scope='session', autouse=True)
def pytest_sessionfinish(request):
    def create_csv():
        print('Writing results to CSV file...')
        # Lưu kết quả vào DataFrame và ghi vào file CSV
        dataFrame = pd.DataFrame(results_df)
        dataFrame.to_csv(CSV_FILE_PATH, sep="}", index=False, encoding='utf-8')
        print(f"Results saved to {CSV_FILE_PATH}")
    request.addfinalizer(create_csv)
