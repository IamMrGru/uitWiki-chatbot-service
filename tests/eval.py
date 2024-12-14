import pandas 
from ragas_metrics import precision,recall,response_relevance,faithfulness,factualcorrectness,answersimilarity
from ast import literal_eval
from tqdm import tqdm

# Đọc danh sách câu hỏi từ file CSV
questions_df = pandas.read_csv("./results.csv", sep='}',encoding='utf-8')
# Xóa cột reference_contexts
# questions_df = questions_df.drop(columns=['reference_contexts'])

# Biến đổi dạng dữ liệu của retrieved_contexts từ string sang list
questions_df['retrieved_contexts'] = questions_df['retrieved_contexts'].apply(literal_eval)


for index, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Evaluating..."):
    question = row["user_input"]
    reference = row["reference"]
    retrieved_contexts_list = row["retrieved_contexts"]
    top_k_retrieved_contexts = retrieved_contexts_list[:10]
    response=row["response"]

    # In câu hỏi và tham chiếu để theo dõi
    print(f'Question: {question}\nReference: {reference}\nResponse: {response}')
    try:
        # Tính Precision@10 và Recall@10
        precision_score=precision(question, reference, top_k_retrieved_contexts)
        recall_score=recall(question, reference, top_k_retrieved_contexts)
        if precision_score+recall_score==0:
            f1_score=0
        else:
            f1_score=(2*(precision_score*recall_score))/(precision_score+recall_score)
        faithful_score=faithfulness(question,response,top_k_retrieved_contexts)
        response_relevance_score=response_relevance(question,response,top_k_retrieved_contexts)
        factualcorrectness_score=factualcorrectness(response,reference)
        answer_similarity_score=answersimilarity(response,reference)

        questions_df.loc[index, "precision"] = precision_score
        questions_df.loc[index, "recall"] = recall_score
        questions_df.loc[index, "f1-score"] = f1_score
        questions_df.loc[index, "faithfulness"] = faithful_score
        questions_df.loc[index, "answer_relevance"] = response_relevance_score
        questions_df.loc[index, "answer_correctness"] = factualcorrectness_score
        questions_df.loc[index, "answer_similarity"] = answer_similarity_score

        
        print(f"Precision: {precision_score}\nRecall: {recall_score}\nF1-Score: {f1_score}\nFaithfulness: {faithful_score}\nAnswer relevance: {response_relevance_score}\nFactual correctness: {factualcorrectness_score}\nAnswer similarity: {answer_similarity_score}\n")
        

    except Exception as e:
        print(f"Error on index {index}: {e}")
        questions_df.loc[index, "precision"] = None
        questions_df.loc[index, "recall"] = None
        questions_df.loc[index, "f1-score"] = None
        questions_df.loc[index, "faithfulness"] = None
        questions_df.loc[index, "answer_relevance"] = None
        questions_df.loc[index, "factual_correctness"] = None
        questions_df.loc[index, "answer_similarity"] = None


# Lưu kết quả vào file CSV
questions_df.to_csv("./results.csv", index=False, sep='}',encoding='utf-8')
print('DONE')
