from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference,LLMContextRecall,Faithfulness,ResponseRelevancy,AnswerSimilarity
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

import asyncio

# Print log to remind user to input the open api key
print("Please input your open api key")

open_api_key=''

def precision(user_input="Manchester United là đội bóng đến từ giải nào?",reference="Ngoại hạng Anh",retrieved_contexts=["Manchester đã vô địch UCL năm 2008","Manchester United là đội bóng đến từ Ngoại hạng Anh"]):
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini",api_key=open_api_key))
    async def contextPrecision():
        context_precision = LLMContextPrecisionWithReference(llm=generator_llm)

        sample = SingleTurnSample(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts, 
        )
        score= await context_precision.single_turn_ascore(sample)
        return score
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(contextPrecision())

def recall(user_input="Manchester United là đội bóng đến từ giải nào?",reference="Ngoại hạng Anh",retrieved_contexts=["Manchester đã vô địch UCL năm 2008","Manchester United là đội bóng đến từ Ngoại hạng Anh"]):
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini",api_key=open_api_key))
    async def contextRecall():
        context_recall = LLMContextRecall(llm=generator_llm)
        sample = SingleTurnSample(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts, 
        )
        score = await context_recall.single_turn_ascore(sample)
        return score
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(contextRecall())


def response_relevance(user_input="Manchester United là đội bóng đến từ giải nào?",response="Ngoại hạng Anh",retrieved_contexts=["Manchester đã vô địch UCL năm 2008","Manchester United là đội bóng đến từ Ngoại hạng Anh"]):
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini",api_key=open_api_key))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=open_api_key))
    async def relevancy():
            sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts
        )
            scorer = ResponseRelevancy(llm=generator_llm,embeddings=generator_embeddings)
            relevance_score=await scorer.single_turn_ascore(sample)
            return relevance_score
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(relevancy())
    
def faithfulness(user_input="Manchester United là đội bóng đến từ giải nào ?",response="Ngoại hạng Anh",retrieved_contexts=["Manchester đã vô địch UCL năm 2008","Manchester United là đội bóng đến từ Ngoại hạng Anh"]):
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini",api_key=open_api_key))
    async def faithful():
            sample = SingleTurnSample(
            user_input=user_input,
            response=f"{response}.",
            retrieved_contexts=retrieved_contexts
        )
            scorer = Faithfulness(llm=generator_llm)
            faithful_score=await scorer.single_turn_ascore(sample)
            return faithful_score
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(faithful())

def factualcorrectness(response="MU là đội bóng của giải Ngoại Hạng Anh",reference="Manchester United là đội bóng đến từ Ngoại hạng Anh"):
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini",api_key=open_api_key))
    async def f_correctness():
            sample = SingleTurnSample(
            response=response,
            reference=reference
        )
            scorer = FactualCorrectness(llm=generator_llm,coverage='high',mode='recall')
            factual_score=await scorer.single_turn_ascore(sample)
            return factual_score
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(f_correctness())

def answersimilarity(response="MU là đội bóng của giải Ngoại Hạng Anh",reference="Manchester United là đội bóng đến từ Ngoại hạng Anh"):
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=open_api_key))
    async def a_similarity():
            sample = SingleTurnSample(
            response=response,
            reference=reference
        )
            scorer = AnswerSimilarity(embeddings=generator_embeddings)
            answersimilarity_score=await scorer.single_turn_ascore(sample)
            return answersimilarity_score
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(a_similarity())



