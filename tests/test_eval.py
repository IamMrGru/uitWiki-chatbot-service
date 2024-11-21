import csv
import json

import pandas as pd
import pytest
from bert_score import score
from fastapi.testclient import TestClient
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import settings
from app.main import app

client = TestClient(app)


def load_questions_from_csv(file_path):
    questions = []
    references = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='}')
        for row in reader:
            questions.append(row["question"])
            references.append(row["reference"])
    return questions, references


questions_data, references_data = load_questions_from_csv(
    'tests/questions.csv')

API_ENDPOINT = f"{settings.API_V1_STR}/chat_bot/send_message"
CSV_FILE_PATH = 'tests/results_evaluation.csv'
JSON_FILE_PATH = 'tests/metrics_summary.json'

results_df = {
    'Câu hỏi': [],
    'Câu trả lời': [],
    'Tham chiếu': [],
    'BLEU': [],
    'ROUGE-L': [],
    'BERTScore': []
}
metrics_summary = {'BLEU_avg': 0, 'ROUGE_avg': 0, 'BERTScore_avg': 0}

# Initialize Rouge scorer
rouge = Rouge()


@pytest.mark.parametrize("idx", range(len(questions_data)))
def test_question_response(idx):
    question = questions_data[idx]
    reference = references_data[idx]
    data = {"user_question": question}

    # Call the API
    response = client.post(API_ENDPOINT, json=data)

    # Process API response
    if response.status_code == 200:
        response_data = response.json()
        generated_answer = response_data.get(
            "response", "").replace('\n', ' ').replace('\r', '')
    else:
        generated_answer = f"Error: {response.status_code}"

    # Calculate BLEU
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu(
        [reference.split()], generated_answer.split(), smoothing_function=smoothie)

    # Calculate ROUGE
    rouge_scores = rouge.get_scores(generated_answer, reference, avg=True)
    rouge_l_score = rouge_scores["rouge-l"]["f"]

    # Calculate BERTScore
    # Change "en" to your language if needed
    bert_p, bert_r, bert_f1 = score(
        [generated_answer], [reference], lang="vie")
    bert_f1_avg = bert_f1.mean().item()

    # Store results
    results_df['Câu hỏi'].append(question)
    results_df['Câu trả lời'].append(generated_answer)
    results_df['Tham chiếu'].append(reference)
    results_df['BLEU'].append(bleu_score)
    results_df['ROUGE-L'].append(rouge_l_score)
    results_df['BERTScore'].append(bert_f1_avg)

    # Assert conditions
    assert response.status_code == 200
    assert "response" in response.json()


@pytest.fixture(scope='session', autouse=True)
def pytest_sessionfinish(request):
    def create_csv():
        print('Writing results to CSV file...')
        dataFrame = pd.DataFrame(results_df)
        dataFrame.to_csv(CSV_FILE_PATH, sep="}", index=False, encoding='utf-8')
        print(f"Results saved to {CSV_FILE_PATH}")
        # Tính trung bình các chỉ số
        metrics_summary['BLEU_avg'] = sum(
            results_df['BLEU']) / len(results_df['BLEU'])
        metrics_summary['ROUGE_avg'] = sum(
            results_df['ROUGE-L']) / len(results_df['ROUGE-L'])
        metrics_summary['BERTScore_avg'] = sum(
            results_df['BERTScore']) / len(results_df['BERTScore'])

        # Lưu kết quả trung bình ra file JSON
        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as json_file:
            json.dump(metrics_summary, json_file, ensure_ascii=False, indent=4)

        print(f"Metrics summary saved to {JSON_FILE_PATH}")

    request.addfinalizer(create_csv)
