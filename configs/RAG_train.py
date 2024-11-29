import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoModelForCausalLM, AutoTokenizer
tcm_llm = AutoModelForCausalLM.from_pretrained("/path/to/ChineseMedicalAssistant_internlm")
tokenizer = AutoTokenizer.from_pretrained("/path/to/ChineseMedicalAssistant_internlm")
with open("RAG_train.json", "r", encoding="utf-8") as f:
    rag_data = json.load(f)
# 初始化SentenceTransformer模型，您可以选择其他合适的中文预训练模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') 
# 将RAG_train.json文件中的文本数据转换成向量
embeddings = []
for item in rag_data:
    text = item["text"] # 假设您的RAG_train.json文件中每个条目都有一个"text"字段
    embedding = model.encode(text)
    embeddings.append(embedding)
def search(query):
    # 将查询转换成向量
    query_embedding = model.encode(query)

    # 计算查询向量与数据库中所有向量的相似度
    similarities = cosine_similarity([query_embedding], embeddings)

    # 获取相似度最高的top_k个索引
    top_k_indices = similarities.argsort()[-5:][::-1]  # 获取相似度最高的5个索引

    # 返回相似度最高的top_k个结果
    results = []
    for index in top_k_indices:
        results.append(rag_data[index])
    return results
def answer_question(query):
    # 检索相关信息
    results = search(query)

    # 将检索结果作为上下文信息提供给TCM_LLM模型
    context = "\n".join([item["text"] for item in results])
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"

    # 使用TCM_LLM模型生成答案
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = tcm_llm.generate(input_ids)
    answer = tokenizer.decode(output, skip_special_tokens=True)
    return answer