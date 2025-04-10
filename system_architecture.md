# 系統架構與模組說明

## 系統架構概覽

本系統採用模組化設計，由六個核心模組組成，每個模組負責特定功能，共同構成完整的智能文檔處理與問答系統。

![系統架構圖](../images/system_architecture.png)

## 模組詳細說明

### 1. 文檔處理模組 (document_processor.py)

文檔處理模組負責文檔的加載、解析和分塊，是系統的入口點。

#### 核心功能

- **文檔加載**：支援多種格式文檔的加載
- **文檔解析**：提取文本內容和元數據
- **文檔分塊**：根據語義或長度進行智能分塊

#### 核心類

- `DocumentLoader`：負責加載不同格式的文檔
- `DocumentProcessor`：負責處理和分塊文檔

#### 使用示例

```python
from document_processor import DocumentProcessor

# 初始化文檔處理器
processor = DocumentProcessor(
    chunk_size=1000,
    chunk_overlap=200
)

# 處理文檔
documents = processor.process_documents("../data/sample_docs")

# 查看處理結果
print(f"處理了 {len(documents)} 個文檔片段")
```

### 2. 向量存儲模組 (vector_store.py)

向量存儲模組負責文檔嵌入和向量索引，是檢索系統的核心。

#### 核心功能

- **文檔嵌入**：使用預訓練模型將文本轉換為向量
- **向量索引**：建立高效的向量索引
- **相似性搜索**：基於向量相似性進行搜索

#### 核心類

- `VectorStore`：負責管理向量存儲
- `EmbeddingModel`：負責文本嵌入

#### 使用示例

```python
from vector_store import VectorStore

# 初始化向量存儲
vector_store = VectorStore(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 添加文檔
vector_store.add_documents(documents)

# 搜索相關文檔
results = vector_store.similarity_search("什麼是RAG？", k=4)

# 查看搜索結果
for doc, score in results:
    print(f"相似度: {score:.4f}, 內容: {doc.page_content[:100]}...")
```

### 3. 檢索增強生成模組 (rag_system.py)

檢索增強生成模組結合檢索結果和LLM生成回答，是系統的核心。

#### 核心功能

- **查詢處理**：分析用戶查詢意圖
- **相關性檢索**：檢索相關文檔片段
- **上下文增強**：將檢索結果作為上下文提供給LLM
- **回答生成**：生成準確、相關的回答

#### 核心類

- `RAGSystem`：負責整合檢索和生成
- `QueryProcessor`：負責處理查詢

#### 使用示例

```python
from rag_system import RAGSystem

# 初始化RAG系統
rag_system = RAGSystem(
    vector_store=vector_store,
    llm_model_name="google/flan-t5-base"
)

# 處理查詢
result = rag_system.query("什麼是RAG？")

# 查看結果
print(f"問題: {result['query']}")
print(f"回答: {result['answer']}")
print("來源文檔:")
for doc in result['source_documents']:
    print(f"- {doc.page_content[:100]}...")
```

### 4. LLM微調模組 (fine_tuning.py)

LLM微調模組提供LLM微調功能，適應特定領域需求。

#### 核心功能

- **數據處理**：準備和格式化微調數據
- **LoRA微調**：實現參數高效微調
- **模型推理**：使用微調後的模型進行推理

#### 核心類

- `DataProcessor`：負責準備微調數據
- `LoRAFineTuner`：負責使用LoRA技術微調模型
- `ModelInferenceHelper`：負責使用微調後的模型進行推理

#### 使用示例

```python
from fine_tuning import DataProcessor, LoRAFineTuner, ModelInferenceHelper

# 準備數據
data_processor = DataProcessor()
train_dataset = data_processor.prepare_dataset("../data/training_data/dataset.json")

# 微調模型
fine_tuner = LoRAFineTuner(
    model_name="meta-llama/Llama-3-8b-hf",
    lora_r=16,
    lora_alpha=32
)
fine_tuner.train(train_dataset)

# 使用微調後的模型進行推理
inference_helper = ModelInferenceHelper(
    model_path="../models/fine_tuned/llama3-lora"
)
response = inference_helper.generate("請解釋RAG的工作原理")
print(response)
```

### 5. 模型量化模組 (model_quantization.py)

模型量化模組提供模型量化功能，降低計算資源需求。

#### 核心功能

- **GPTQ量化**：實現GPTQ算法的模型量化
- **AWQ量化**：實現AWQ算法的模型量化
- **量化模型管理**：管理和使用量化後的模型

#### 核心類

- `GPTQQuantizer`：負責使用GPTQ算法量化模型
- `AWQQuantizer`：負責使用AWQ算法量化模型
- `QuantizedModelManager`：負責管理和使用量化後的模型

#### 使用示例

```python
from model_quantization import GPTQQuantizer, QuantizedModelManager

# 量化模型
quantizer = GPTQQuantizer(
    config=QuantizationConfig(
        model_name_or_path="meta-llama/Llama-3-8b-hf",
        output_dir="../models/quantized/llama3-8b-gptq-4bit",
        bits=4
    )
)
quantizer.generate_quantization_instructions()

# 使用量化後的模型
model_manager = QuantizedModelManager(
    model_path="../models/quantized/llama3-8b-gptq-4bit",
    quantization_type="gptq"
)
model, tokenizer = model_manager.load_model()
response = model_manager.generate_response("請解釋RAG的工作原理")
print(response)
```

### 6. 向量資料庫整合模組 (vector_db_integration.py)

向量資料庫整合模組支援多種向量資料庫，提供統一的接口。

#### 核心功能

- **多資料庫支援**：支援FAISS、Chroma、Weaviate、Milvus、Qdrant等
- **統一接口**：提供統一的接口操作不同資料庫
- **多資料庫RAG**：支援在不同向量資料庫間切換

#### 核心類

- `VectorDBFactory`：負責創建不同類型的向量資料庫
- `BaseVectorDB`：向量資料庫基類
- `VectorDBManager`：負責管理多個向量資料庫
- `RAGSystemWithMultiVectorDB`：支援多向量資料庫的RAG系統

#### 使用示例

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

# 添加文檔
faiss_db.add_documents(documents)

# 創建RAG系統
rag_system = RAGSystemWithMultiVectorDB(
    vector_db_manager=vector_db_manager,
    default_db_config=faiss_config
)

# 使用FAISS資料庫查詢
result = rag_system.query("什麼是向量資料庫？", "faiss", "documents")
print(result["answer"])

# 創建Chroma向量資料庫
chroma_config = VectorDBConfig(
    db_type="chroma",
    collection_name="documents",
    persist_directory="../data/vector_store/chroma_documents"
)
chroma_db = vector_db_manager.create_vector_db(chroma_config)

# 添加文檔
chroma_db.add_documents(documents)

# 使用Chroma資料庫查詢
result = rag_system.query("什麼是向量資料庫？", "chroma", "documents")
print(result["answer"])
```

### 7. 評估指標系統 (evaluation_metrics.py)

評估指標系統提供全面的評估指標，確保系統質量。

#### 核心功能

- **檢索評估**：評估檢索系統的性能
- **生成評估**：評估生成系統的性能
- **系統評估**：評估整個RAG系統的性能

#### 核心類

- `RetrievalEvaluator`：負責評估檢索性能
- `GenerationEvaluator`：負責評估生成性能
- `RAGEvaluator`：負責評估RAG系統性能
- `EvaluationPipeline`：提供端到端的評估流程

#### 使用示例

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

## 模組間的交互

1. **文檔處理 → 向量存儲**：文檔處理模組處理文檔後，將結果傳遞給向量存儲模組進行向量化和索引。

2. **向量存儲 → 檢索增強生成**：檢索增強生成模組使用向量存儲模組檢索相關文檔，作為生成回答的上下文。

3. **LLM微調 → 檢索增強生成**：檢索增強生成模組可以使用LLM微調模組微調後的模型進行回答生成。

4. **模型量化 → 檢索增強生成**：檢索增強生成模組可以使用模型量化模組量化後的模型進行回答生成。

5. **向量資料庫整合 → 檢索增強生成**：檢索增強生成模組可以使用向量資料庫整合模組提供的多種向量資料庫進行檢索。

6. **評估指標系統 → 所有模組**：評估指標系統可以評估各個模組的性能，提供改進建議。

## 擴展與定制

系統的模組化設計使其具有良好的可擴展性和可定制性。用戶可以根據需求替換或擴展特定模組，例如：

1. **替換嵌入模型**：可以使用不同的嵌入模型，如OpenAI的text-embedding-ada-002或自定義嵌入模型。

2. **替換LLM**：可以使用不同的LLM，如GPT-4、Claude或自定義LLM。

3. **添加新的向量資料庫**：可以添加對新的向量資料庫的支援，如PGVector、Elasticsearch等。

4. **擴展評估指標**：可以添加新的評估指標，如回答長度、多樣性等。

5. **添加新的文檔格式**：可以添加對新的文檔格式的支援，如Markdown、CSV等。

## 最佳實踐

1. **文檔處理**：
   - 根據文檔類型選擇適當的分塊策略
   - 保留文檔的元數據，如來源、日期等
   - 對於長文檔，使用層次化分塊

2. **向量存儲**：
   - 選擇適合數據規模的向量資料庫
   - 定期更新向量索引
   - 使用適當的相似度度量（如餘弦相似度）

3. **檢索增強生成**：
   - 優化提示模板
   - 使用多樣性檢索（如MMR）
   - 實現重排序機制

4. **LLM微調**：
   - 使用高質量的微調數據
   - 選擇適當的微調參數
   - 定期評估微調效果

5. **模型量化**：
   - 根據硬件選擇適當的量化方法
   - 使用代表性的校準數據
   - 評估量化對性能的影響

6. **評估**：
   - 使用多種評估指標
   - 定期評估系統性能
   - 根據評估結果持續改進系統
