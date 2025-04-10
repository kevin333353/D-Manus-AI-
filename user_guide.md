# 使用指南與示例

本文檔提供了智能文檔處理與問答系統的詳細使用指南和示例，幫助您快速上手和深入使用系統的各個功能。

## 目錄

1. [環境準備](#環境準備)
2. [基本使用流程](#基本使用流程)
3. [文檔處理模組使用指南](#文檔處理模組使用指南)
4. [RAG系統使用指南](#rag系統使用指南)
5. [LLM微調使用指南](#llm微調使用指南)
6. [模型量化使用指南](#模型量化使用指南)
7. [向量資料庫整合使用指南](#向量資料庫整合使用指南)
8. [評估指標系統使用指南](#評估指標系統使用指南)
9. [常見問題與解決方案](#常見問題與解決方案)
10. [進階使用場景](#進階使用場景)

## 環境準備

### 系統要求

- Python 3.8+
- CUDA 11.7+ (用於GPU加速，可選)
- 至少8GB RAM (推薦16GB+)
- 至少20GB磁盤空間

### 安裝步驟

1. 克隆專案

```bash
git clone https://github.com/yourusername/llm-rag-project.git
cd llm-rag-project
```

2. 創建虛擬環境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. 安裝依賴

```bash
pip install -r requirements.txt
```

4. 安裝特定模組依賴（可選）

```bash
# 安裝LLM微調依賴
pip install -r code/fine_tuning/requirements.txt

# 安裝模型量化依賴
pip install -r code/quantization/requirements.txt

# 安裝向量資料庫依賴
pip install -r code/vector_db_integration/requirements.txt
```

5. 設置環境變數

創建`.env`文件，添加必要的API密鑰和配置：

```
# OpenAI API (可選)
OPENAI_API_KEY=your_openai_api_key

# Hugging Face (可選)
HUGGINGFACE_API_KEY=your_huggingface_api_key

# 向量資料庫配置 (可選)
WEAVIATE_URL=your_weaviate_url
MILVUS_URI=your_milvus_uri
QDRANT_URL=your_qdrant_url
```

## 基本使用流程

系統的基本使用流程包括以下步驟：

1. 準備文檔
2. 處理文檔
3. 創建向量存儲
4. 設置RAG系統
5. 進行問答

以下是一個基本的使用示例：

```python
# 導入必要的模組
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_system import RAGSystem

# 1. 初始化文檔處理器
processor = DocumentProcessor(
    chunk_size=1000,
    chunk_overlap=200
)

# 2. 處理文檔
documents = processor.process_documents("../data/sample_docs")
print(f"處理了 {len(documents)} 個文檔片段")

# 3. 創建向量存儲
vector_store = VectorStore(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store.add_documents(documents)

# 4. 設置RAG系統
rag_system = RAGSystem(
    vector_store=vector_store,
    llm_model_name="google/flan-t5-base"
)

# 5. 進行問答
result = rag_system.query("什麼是RAG？")
print(f"問題: {result['query']}")
print(f"回答: {result['answer']}")
print("來源文檔:")
for doc in result['source_documents']:
    print(f"- {doc.page_content[:100]}...")
```

## 文檔處理模組使用指南

文檔處理模組負責文檔的加載、解析和分塊，是系統的入口點。

### 基本使用

```python
from document_processor import DocumentProcessor

# 初始化文檔處理器
processor = DocumentProcessor(
    chunk_size=1000,  # 每個片段的大小
    chunk_overlap=200  # 片段之間的重疊大小
)

# 處理單個文檔
document = processor.process_document("../data/sample_docs/document.pdf")

# 處理目錄中的所有文檔
documents = processor.process_documents("../data/sample_docs")
```

### 自定義分塊策略

```python
from document_processor import DocumentProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 創建自定義分塊器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

# 使用自定義分塊器初始化文檔處理器
processor = DocumentProcessor(
    text_splitter=text_splitter
)
```

### 處理不同類型的文檔

```python
from document_processor import DocumentProcessor

processor = DocumentProcessor()

# 處理PDF文檔
pdf_docs = processor.process_document("../data/sample_docs/document.pdf")

# 處理Word文檔
docx_docs = processor.process_document("../data/sample_docs/document.docx")

# 處理文本文檔
txt_docs = processor.process_document("../data/sample_docs/document.txt")

# 處理HTML文檔
html_docs = processor.process_document("../data/sample_docs/document.html")
```

### 保存處理結果

```python
from document_processor import DocumentProcessor
import json

processor = DocumentProcessor()
documents = processor.process_documents("../data/sample_docs")

# 保存處理結果
processor.save_documents(documents, "../data/processed/processed_documents.json")

# 加載處理結果
loaded_documents = processor.load_documents("../data/processed/processed_documents.json")
```

## RAG系統使用指南

RAG系統是本專案的核心，結合檢索和生成能力，提供準確的問答功能。

### 基本使用

```python
from rag_system import RAGSystem
from vector_store import VectorStore

# 創建向量存儲
vector_store = VectorStore(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store.add_documents(documents)

# 創建RAG系統
rag_system = RAGSystem(
    vector_store=vector_store,
    llm_model_name="google/flan-t5-base"
)

# 單次查詢
result = rag_system.query("什麼是RAG？")
print(result["answer"])

# 批量查詢
queries = ["什麼是RAG？", "RAG有什麼優勢？", "RAG的應用場景有哪些？"]
results = rag_system.batch_query(queries)
for result in results:
    print(f"問題: {result['query']}")
    print(f"回答: {result['answer']}")
    print("---")
```

### 自定義檢索參數

```python
from rag_system import RAGSystem

# 創建RAG系統
rag_system = RAGSystem(
    vector_store=vector_store,
    llm_model_name="google/flan-t5-base",
    search_type="mmr",  # 使用最大邊際相關性
    search_kwargs={
        "k": 10,  # 檢索文檔數量
        "fetch_k": 20,  # 初始檢索數量
        "lambda_mult": 0.7  # 多樣性參數
    }
)
```

### 自定義提示模板

```python
from rag_system import RAGSystem
from langchain.prompts import PromptTemplate

# 創建自定義提示模板
template = """
你是一個專業的助手，請根據以下上下文回答問題。
如果上下文中沒有相關信息，請說"我沒有足夠的信息來回答這個問題"。

上下文:
{context}

問題: {question}

回答:
"""
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 創建RAG系統
rag_system = RAGSystem(
    vector_store=vector_store,
    llm_model_name="google/flan-t5-base",
    prompt_template=prompt
)
```

### 使用不同的LLM

```python
from rag_system import RAGSystem
from langchain.llms import OpenAI

# 使用OpenAI LLM
llm = OpenAI(temperature=0.7)

# 創建RAG系統
rag_system = RAGSystem(
    vector_store=vector_store,
    llm=llm  # 直接傳入LLM實例
)
```

### 保存和加載RAG系統

```python
from rag_system import RAGSystem

# 保存RAG系統
rag_system.save("../models/rag_system")

# 加載RAG系統
loaded_rag_system = RAGSystem.load("../models/rag_system")
```

## LLM微調使用指南

LLM微調模組提供了參數高效微調功能，使LLM能夠適應特定領域需求。

### 準備微調數據

```python
from fine_tuning import DataProcessor

# 初始化數據處理器
data_processor = DataProcessor()

# 準備指令微調數據
instruction_data = [
    {"instruction": "解釋RAG的工作原理", "output": "RAG（檢索增強生成）是一種結合檢索系統和生成模型的方法..."},
    {"instruction": "比較RAG和傳統問答系統", "output": "RAG與傳統問答系統的主要區別在於..."}
]
instruction_dataset = data_processor.prepare_instruction_dataset(instruction_data)

# 準備對話微調數據
conversation_data = [
    {
        "messages": [
            {"role": "user", "content": "什麼是RAG？"},
            {"role": "assistant", "content": "RAG（檢索增強生成）是一種結合檢索系統和生成模型的方法..."}
        ]
    }
]
conversation_dataset = data_processor.prepare_conversation_dataset(conversation_data)

# 保存數據集
data_processor.save_dataset(instruction_dataset, "../data/training_data/instruction_dataset.json")
data_processor.save_dataset(conversation_dataset, "../data/training_data/conversation_dataset.json")
```

### 使用LoRA微調

```python
from fine_tuning import LoRAFineTuner

# 初始化LoRA微調器
fine_tuner = LoRAFineTuner(
    model_name="meta-llama/Llama-3-8b-hf",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)

# 加載數據集
train_dataset = fine_tuner.load_dataset("../data/training_data/instruction_dataset.json")

# 微調模型
fine_tuner.train(
    train_dataset=train_dataset,
    output_dir="../models/fine_tuned/llama3-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100
)
```

### 使用微調後的模型

```python
from fine_tuning import ModelInferenceHelper

# 初始化推理助手
inference_helper = ModelInferenceHelper(
    model_path="../models/fine_tuned/llama3-lora"
)

# 生成回答
response = inference_helper.generate("請解釋RAG的工作原理")
print(response)

# 批量生成
queries = ["什麼是RAG？", "RAG有什麼優勢？", "RAG的應用場景有哪些？"]
responses = inference_helper.batch_generate(queries)
for query, response in zip(queries, responses):
    print(f"問題: {query}")
    print(f"回答: {response}")
    print("---")
```

### 整合微調模型到RAG系統

```python
from rag_system import RAGSystem
from fine_tuning import ModelInferenceHelper

# 初始化推理助手
inference_helper = ModelInferenceHelper(
    model_path="../models/fine_tuned/llama3-lora"
)

# 創建RAG系統
rag_system = RAGSystem(
    vector_store=vector_store,
    llm=inference_helper.model  # 使用微調後的模型
)
```

## 模型量化使用指南

模型量化模組提供了模型量化功能，降低計算資源需求。

### GPTQ量化

```python
from model_quantization import GPTQQuantizer, QuantizationConfig

# 創建量化配置
config = QuantizationConfig(
    model_name_or_path="meta-llama/Llama-3-8b-hf",
    output_dir="../models/quantized/llama3-8b-gptq-4bit",
    bits=4,
    group_size=128,
    desc_act=True,
    sym=True
)

# 初始化量化器
quantizer = GPTQQuantizer(config)

# 生成量化指令
instructions = quantizer.generate_quantization_instructions()
print(instructions)

# 準備校準數據
calibration_samples = quantizer.prepare_calibration_data(
    data_path="../data/calibration_data.txt",
    num_samples=100
)

# 量化模型
quantizer.quantize(calibration_samples)
```

### AWQ量化

```python
from model_quantization import AWQQuantizer, QuantizationConfig

# 創建量化配置
config = QuantizationConfig(
    model_name_or_path="meta-llama/Llama-3-8b-hf",
    output_dir="../models/quantized/llama3-8b-awq-4bit",
    bits=4,
    group_size=128,
    desc_act=True,
    sym=True
)

# 初始化量化器
quantizer = AWQQuantizer(config)

# 生成量化指令
instructions = quantizer.generate_quantization_instructions()
print(instructions)

# 準備校準數據
calibration_samples = quantizer.prepare_calibration_data(
    data_path="../data/calibration_data.txt",
    num_samples=100
)

# 量化模型
quantizer.quantize(calibration_samples)
```

### 使用量化後的模型

```python
from model_quantization import QuantizedModelManager

# 初始化量化模型管理器
model_manager = QuantizedModelManager(
    model_path="../models/quantized/llama3-8b-gptq-4bit",
    quantization_type="gptq"
)

# 加載模型
model, tokenizer = model_manager.load_model()

# 生成回答
response = model_manager.generate_response(
    "請解釋RAG的工作原理",
    max_length=512,
    temperature=0.7
)
print(response)
```

### 整合量化模型到RAG系統

```python
from rag_system import RAGSystem
from model_quantization import QuantizedModelManager

# 初始化量化模型管理器
model_manager = QuantizedModelManager(
    model_path="../models/quantized/llama3-8b-gptq-4bit",
    quantization_type="gptq"
)

# 加載模型
model, tokenizer = model_manager.load_model()

# 創建RAG系統
rag_system = RAGSystem(
    vector_store=vector_store,
    llm=model  # 使用量化後的模型
)
```

## 向量資料庫整合使用指南

向量資料庫整合模組支援多種向量資料庫，提供統一的接口。

### 創建向量資料庫

```python
from vector_db_integration import VectorDBConfig, VectorDBManager

# 創建向量資料庫管理器
vector_db_manager = VectorDBManager()

# 創建FAISS向量資料庫
faiss_config = VectorDBConfig(
    db_type="faiss",
    collection_name="documents",
    persist_directory="../data/vector_store/faiss_documents"
)
faiss_db = vector_db_manager.create_vector_db(faiss_config)

# 創建Chroma向量資料庫
chroma_config = VectorDBConfig(
    db_type="chroma",
    collection_name="documents",
    persist_directory="../data/vector_store/chroma_documents"
)
chroma_db = vector_db_manager.create_vector_db(chroma_config)

# 創建Weaviate向量資料庫
weaviate_config = VectorDBConfig(
    db_type="weaviate",
    collection_name="Documents",
    url=os.getenv("WEAVIATE_URL")
)
weaviate_db = vector_db_manager.create_vector_db(weaviate_config)

# 創建Milvus向量資料庫
milvus_config = VectorDBConfig(
    db_type="milvus",
    collection_name="documents",
    connection_args={"uri": os.getenv("MILVUS_URI")}
)
milvus_db = vector_db_manager.create_vector_db(milvus_config)

# 創建Qdrant向量資料庫
qdrant_config = VectorDBConfig(
    db_type="qdrant",
    collection_name="documents",
    url=os.getenv("QDRANT_URL")
)
qdrant_db = vector_db_manager.create_vector_db(qdrant_config)
```

### 添加文檔到向量資料庫

```python
from vector_db_integration import VectorDBConfig, VectorDBManager
from document_processor import DocumentProcessor

# 處理文檔
processor = DocumentProcessor()
documents = processor.process_documents("../data/sample_docs")

# 創建向量資料庫管理器
vector_db_manager = VectorDBManager()

# 創建FAISS向量資料庫
faiss_config = VectorDBConfig(
    db_type="faiss",
    collection_name="documents",
    persist_directory="../data/vector_store/faiss_documents"
)
faiss_db = vector_db_manager.create_vector_db(faiss_config)

# 添加文檔
faiss_db.add_documents(documents)

# 保存向量資料庫
faiss_db.persist()
```

### 使用多向量資料庫RAG系統

```python
from vector_db_integration import VectorDBConfig, VectorDBManager, RAGSystemWithMultiVectorDB

# 創建向量資料庫管理器
vector_db_manager = VectorDBManager()

# 創建FAISS向量資料庫
faiss_config = VectorDBConfig(
    db_type="faiss",
    collection_name="documents",
    persist_directory="../data/vector_store/faiss_documents"
)
faiss_db = vector_db_manager.create_vector_db(faiss_config)

# 創建Chroma向量資料庫
chroma_config = VectorDBConfig(
    db_type="chroma",
    collection_name="documents",
    persist_directory="../data/vector_store/chroma_documents"
)
chroma_db = vector_db_manager.create_vector_db(chroma_config)

# 添加文檔
documents = processor.process_documents("../data/sample_docs")
faiss_db.add_documents(documents)
chroma_db.add_documents(documents)

# 創建RAG系統
rag_system = RAGSystemWithMultiVectorDB(
    vector_db_manager=vector_db_manager,
    default_db_config=faiss_config
)

# 使用FAISS資料庫查詢
result_faiss = rag_system.query("什麼是向量資料庫？", "faiss", "documents")
print(f"FAISS回答: {result_faiss['answer']}")

# 使用Chroma資料庫查詢
result_chroma = rag_system.query("什麼是向量資料庫？", "chroma", "documents")
print(f"Chroma回答: {result_chroma['answer']}")
```

### 向量資料庫性能比較

```python
from vector_db_integration import VectorDBConfig, VectorDBManager, compare_vector_dbs

# 創建向量資料庫管理器
vector_db_manager = VectorDBManager()

# 創建配置
configs = [
    VectorDBConfig(db_type="faiss", collection_name="documents", persist_directory="../data/vector_store/faiss_documents"),
    VectorDBConfig(db_type="chroma", collection_name="documents", persist_directory="../data/vector_store/chroma_documents"),
    VectorDBConfig(db_type="weaviate", collection_name="Documents", url=os.getenv("WEAVIATE_URL"))
]

# 比較向量資料庫性能
results = compare_vector_dbs(
    vector_db_manager=vector_db_manager,
    configs=configs,
    documents=documents,
    queries=["什麼是向量資料庫？", "RAG的工作原理是什麼？", "如何選擇合適的向量資料庫？"],
    k=5
)

# 顯示結果
for db_type, metrics in results.items():
    print(f"{db_type}:")
    print(f"  平均查詢時間: {metrics['avg_query_time']:.4f}秒")
    print(f"  平均添加時間: {metrics['avg_add_time']:.4f}秒")
    print(f"  內存使用: {metrics['memory_usage']:.2f}MB")
```

## 評估指標系統使用指南

評估指標系統提供全面的評估指標，確保系統質量。

### 基本評估

```python
from evaluation_metrics import EvaluationConfig, EvaluationPipeline

# 創建評估配置
config = EvaluationConfig(
    metrics=["relevance", "factuality", "coherence", "fluency", "rouge", "bleu"],
    retrieval_metrics=["precision", "recall", "ndcg", "mrr"],
    output_dir="../evaluation_results"
)

# 創建評估流水線
pipeline = EvaluationPipeline(config)

# 評估RAG系統
results = pipeline.evaluate_system(rag_system)

# 查看結果
print("檢索評估結果:")
for metric, score in results["retrieval"].items():
    print(f"- {metric}: {score:.4f}")

print("\n生成評估結果:")
for metric, score in results["generation"].items():
    print(f"- {metric}: {score:.4f}")
```

### 生成測試數據

```python
from evaluation_metrics import EvaluationConfig, EvaluationDataGenerator

# 創建評估配置
config = EvaluationConfig(
    output_dir="../evaluation_results"
)

# 創建數據生成器
data_generator = EvaluationDataGenerator(config)

# 生成測試數據
queries, reference_answers, relevant_docs = data_generator.generate_test_data(
    num_samples=100,
    domains=["general", "tech", "science", "finance", "healthcare"]
)

# 保存測試數據
data_generator._save_test_data(queries, reference_answers, relevant_docs)
```

### 單獨評估檢索性能

```python
from evaluation_metrics import EvaluationConfig, RetrievalEvaluator

# 創建評估配置
config = EvaluationConfig(
    retrieval_metrics=["precision", "recall", "ndcg", "mrr"],
    output_dir="../evaluation_results"
)

# 創建檢索評估器
evaluator = RetrievalEvaluator(config)

# 評估檢索性能
retrieval_results = evaluator.evaluate_retrieval(
    queries=queries,
    retrieved_docs=retrieved_docs,
    relevant_docs=relevant_docs
)

# 查看結果
for metric, score in retrieval_results.items():
    print(f"{metric}: {score:.4f}")
```

### 單獨評估生成性能

```python
from evaluation_metrics import EvaluationConfig, GenerationEvaluator

# 創建評估配置
config = EvaluationConfig(
    metrics=["relevance", "factuality", "coherence", "fluency", "rouge", "bleu"],
    output_dir="../evaluation_results"
)

# 創建生成評估器
evaluator = GenerationEvaluator(config)

# 評估生成性能
generation_results = evaluator.evaluate_generation(
    queries=queries,
    generated_answers=generated_answers,
    reference_answers=reference_answers,
    contexts=contexts
)

# 查看結果
for metric, score in generation_results.items():
    print(f"{metric}: {score:.4f}")
```

### 創建評估報告

```python
from evaluation_metrics import EvaluationConfig, EvaluationPipeline

# 創建評估配置
config = EvaluationConfig(
    metrics=["relevance", "factuality", "coherence", "fluency", "rouge", "bleu"],
    retrieval_metrics=["precision", "recall", "ndcg", "mrr"],
    output_dir="../evaluation_results"
)

# 創建評估流水線
pipeline = EvaluationPipeline(config)

# 評估RAG系統
results = pipeline.evaluate_system(rag_system)

# 生成評估報告
report = pipeline._generate_evaluation_report(results)

# 保存評估報告
with open("../evaluation_results/evaluation_report.md", "w", encoding="utf-8") as f:
    f.write(report)
```

## 常見問題與解決方案

### 1. 文檔處理問題

**問題**：處理大型PDF文檔時內存不足

**解決方案**：
- 使用流式處理
- 減小chunk_size
- 分批處理文檔

```python
from document_processor import DocumentProcessor

# 使用流式處理
processor = DocumentProcessor(
    chunk_size=500,  # 減小chunk_size
    chunk_overlap=50
)

# 分批處理文檔
all_documents = []
for i in range(0, len(pdf_files), 5):
    batch = pdf_files[i:i+5]
    documents = processor.process_documents(batch)
    all_documents.extend(documents)
```

### 2. 向量存儲問題

**問題**：向量存儲速度慢

**解決方案**：
- 使用批處理
- 選擇更高效的向量資料庫
- 優化嵌入模型

```python
from vector_store import VectorStore

# 使用批處理
vector_store = VectorStore(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)
for i in range(0, len(documents), 100):
    batch = documents[i:i+100]
    vector_store.add_documents(batch)
```

### 3. RAG系統問題

**問題**：生成的回答不準確或不相關

**解決方案**：
- 優化檢索參數
- 使用更好的LLM
- 優化提示模板

```python
from rag_system import RAGSystem

# 優化檢索參數
rag_system = RAGSystem(
    vector_store=vector_store,
    llm_model_name="google/flan-t5-large",  # 使用更好的LLM
    search_type="mmr",  # 使用最大邊際相關性
    search_kwargs={
        "k": 8,  # 增加檢索文檔數量
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)
```

### 4. LLM微調問題

**問題**：微調過程中顯存不足

**解決方案**：
- 減小batch_size
- 使用梯度累積
- 使用更小的模型

```python
from fine_tuning import LoRAFineTuner

# 優化微調參數
fine_tuner = LoRAFineTuner(
    model_name="meta-llama/Llama-3-8b-hf"
)

fine_tuner.train(
    train_dataset=train_dataset,
    output_dir="../models/fine_tuned/llama3-lora",
    per_device_train_batch_size=2,  # 減小batch_size
    gradient_accumulation_steps=8,  # 使用梯度累積
    fp16=True  # 使用混合精度訓練
)
```

### 5. 模型量化問題

**問題**：量化後模型性能下降

**解決方案**：
- 使用更高的位元數
- 優化校準數據
- 嘗試不同的量化方法

```python
from model_quantization import GPTQQuantizer, QuantizationConfig

# 優化量化參數
config = QuantizationConfig(
    model_name_or_path="meta-llama/Llama-3-8b-hf",
    output_dir="../models/quantized/llama3-8b-gptq-8bit",
    bits=8,  # 使用更高的位元數
    group_size=128,
    desc_act=True,
    sym=True
)

quantizer = GPTQQuantizer(config)

# 使用更多校準數據
calibration_samples = quantizer.prepare_calibration_data(
    data_path="../data/calibration_data.txt",
    num_samples=200  # 增加校準數據數量
)

quantizer.quantize(calibration_samples)
```

## 進階使用場景

### 1. 多領域知識庫

```python
from document_processor import DocumentProcessor
from vector_db_integration import VectorDBConfig, VectorDBManager, RAGSystemWithMultiVectorDB

# 處理不同領域的文檔
processor = DocumentProcessor()
tech_docs = processor.process_documents("../data/tech_docs")
finance_docs = processor.process_documents("../data/finance_docs")
healthcare_docs = processor.process_documents("../data/healthcare_docs")

# 創建向量資料庫管理器
vector_db_manager = VectorDBManager()

# 為每個領域創建向量資料庫
tech_config = VectorDBConfig(
    db_type="faiss",
    collection_name="tech",
    persist_directory="../data/vector_store/tech"
)
tech_db = vector_db_manager.create_vector_db(tech_config)
tech_db.add_documents(tech_docs)

finance_config = VectorDBConfig(
    db_type="faiss",
    collection_name="finance",
    persist_directory="../data/vector_store/finance"
)
finance_db = vector_db_manager.create_vector_db(finance_config)
finance_db.add_documents(finance_docs)

healthcare_config = VectorDBConfig(
    db_type="faiss",
    collection_name="healthcare",
    persist_directory="../data/vector_store/healthcare"
)
healthcare_db = vector_db_manager.create_vector_db(healthcare_config)
healthcare_db.add_documents(healthcare_docs)

# 創建RAG系統
rag_system = RAGSystemWithMultiVectorDB(
    vector_db_manager=vector_db_manager,
    default_db_config=tech_config
)

# 根據查詢領域選擇向量資料庫
def query_with_domain_detection(query):
    # 簡單的領域檢測
    if "股票" in query or "投資" in query or "金融" in query:
        domain = "finance"
        collection = "finance"
    elif "醫療" in query or "疾病" in query or "健康" in query:
        domain = "healthcare"
        collection = "healthcare"
    else:
        domain = "tech"
        collection = "tech"
    
    # 使用對應領域的向量資料庫查詢
    result = rag_system.query(query, domain, collection)
    return result

# 測試
queries = [
    "什麼是機器學習？",  # 技術領域
    "如何進行股票投資？",  # 金融領域
    "常見的心臟疾病有哪些？"  # 醫療領域
]

for query in queries:
    result = query_with_domain_detection(query)
    print(f"問題: {query}")
    print(f"回答: {result['answer']}")
    print("---")
```

### 2. 混合檢索策略

```python
from langchain.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from rag_system import RAGSystem

# 創建BM25檢索器
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# 創建向量檢索器
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 創建多查詢檢索器
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_retriever,
    llm=llm
)

# 創建集成檢索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# 創建RAG系統
rag_system = RAGSystem(
    retriever=ensemble_retriever,  # 使用集成檢索器
    llm_model_name="google/flan-t5-base"
)

# 查詢
result = rag_system.query("什麼是RAG？")
print(result["answer"])
```

### 3. 流式回答生成

```python
from rag_system import RAGSystem
import time

# 創建RAG系統
rag_system = RAGSystem(
    vector_store=vector_store,
    llm_model_name="google/flan-t5-base"
)

# 流式生成回答
def stream_answer(query):
    # 檢索相關文檔
    docs = rag_system.retrieve_documents(query)
    
    # 準備上下文
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 生成回答（模擬流式生成）
    answer = rag_system.generate_answer(query, context)
    
    # 模擬流式輸出
    words = answer.split()
    result = ""
    for word in words:
        result += word + " "
        print(result, end="\r")
        time.sleep(0.1)
    
    print("\n")
    return answer

# 測試
stream_answer("什麼是RAG？")
```

### 4. 自定義評估指標

```python
from evaluation_metrics import EvaluationConfig, GenerationEvaluator
import numpy as np

# 創建自定義評估器
class CustomGenerationEvaluator(GenerationEvaluator):
    def __init__(self, config):
        super().__init__(config)
    
    # 添加新的評估指標：回答長度
    def _calculate_answer_length(self, generated_answers):
        lengths = [len(answer.split()) for answer in generated_answers]
        return np.mean(lengths)
    
    # 添加新的評估指標：多樣性
    def _calculate_diversity(self, generated_answers):
        if not generated_answers:
            return 0.0
        
        # 計算詞彙多樣性
        all_words = []
        for answer in generated_answers:
            words = answer.lower().split()
            all_words.extend(words)
        
        unique_words = set(all_words)
        diversity = len(unique_words) / len(all_words) if all_words else 0.0
        
        return diversity
    
    # 重寫評估方法
    def evaluate_generation(self, queries, generated_answers, reference_answers, contexts=None):
        # 調用父類方法獲取基本評估結果
        results = super().evaluate_generation(queries, generated_answers, reference_answers, contexts)
        
        # 添加自定義評估指標
        results["answer_length"] = self._calculate_answer_length(generated_answers)
        results["diversity"] = self._calculate_diversity(generated_answers)
        
        return results

# 創建評估配置
config = EvaluationConfig(
    metrics=["relevance", "factuality", "coherence", "fluency", "rouge", "bleu"],
    output_dir="../evaluation_results"
)

# 創建自定義評估器
custom_evaluator = CustomGenerationEvaluator(config)

# 評估生成性能
generation_results = custom_evaluator.evaluate_generation(
    queries=queries,
    generated_answers=generated_answers,
    reference_answers=reference_answers,
    contexts=contexts
)

# 查看結果
for metric, score in generation_results.items():
    print(f"{metric}: {score:.4f}")
```

### 5. 多模型集成

```python
from rag_system import RAGSystem
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# 創建多個LLM
llm1 = OpenAI(temperature=0.7)
llm2 = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",
    task="text2text-generation",
    model_kwargs={"temperature": 0.5}
)

# 創建提示模板
template = """
你是一個專業的助手，請根據以下上下文回答問題。
如果上下文中沒有相關信息，請說"我沒有足夠的信息來回答這個問題"。

上下文:
{context}

問題: {question}

回答:
"""
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 創建多個LLM鏈
chain1 = LLMChain(llm=llm1, prompt=prompt)
chain2 = LLMChain(llm=llm2, prompt=prompt)

# 創建RAG系統
rag_system = RAGSystem(
    vector_store=vector_store
)

# 使用多模型集成生成回答
def ensemble_query(query):
    # 檢索相關文檔
    docs = rag_system.retrieve_documents(query)
    
    # 準備上下文
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 使用多個模型生成回答
    answer1 = chain1.run(context=context, question=query)
    answer2 = chain2.run(context=context, question=query)
    
    # 簡單集成：選擇較長的回答
    if len(answer1) > len(answer2):
        final_answer = answer1
    else:
        final_answer = answer2
    
    return {
        "query": query,
        "answer": final_answer,
        "source_documents": docs,
        "model_answers": {
            "model1": answer1,
            "model2": answer2
        }
    }

# 測試
result = ensemble_query("什麼是RAG？")
print(f"問題: {result['query']}")
print(f"最終回答: {result['answer']}")
print(f"模型1回答: {result['model_answers']['model1']}")
print(f"模型2回答: {result['model_answers']['model2']}")
```
