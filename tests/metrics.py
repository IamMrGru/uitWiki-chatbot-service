import pandas as pd

# Read CSV file
df = pd.read_csv("./results.csv", sep='}')


# Precision
print('Precision@10: ',df['precision'].mean())
# Recall
print('Recall@10: ',df['recall'].mean())
# F1-Score
print('F1-Score@10: ',df['f1-score'].mean())
# Faithfulness
print('Faithfulness: ',df['faithfulness'].mean())
# Answer Relevance
print('Answer Relevance: ',df['answer_relevance'].mean())
# Factual Metrics
print('Answer Similarity: ',df['answer_similarity'].mean())
