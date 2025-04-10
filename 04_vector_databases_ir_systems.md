# 向量資料庫與資訊檢索系統架構

## 向量資料庫基礎概念

向量資料庫是一種專門設計用於存儲、索引和查詢高維向量數據的資料庫系統。在大型語言模型（LLM）和檢索增強生成（RAG）應用中，向量資料庫扮演著至關重要的角色，它們能夠高效地執行相似性搜索，找出與查詢向量最相似的文檔或資訊片段。

### 向量嵌入與相似性搜索

向量資料庫的核心概念是將非結構化數據（如文本、圖像、音頻等）轉換為數值向量表示，這一過程稱為「嵌入」（Embedding）。嵌入模型將語義信息編碼到高維空間中，使得語義相似的內容在向量空間中彼此接近。

常見的相似性度量方法包括：

1. **餘弦相似度（Cosine Similarity）**：測量兩個向量之間的夾角餘弦值，範圍在 -1 到 1 之間，值越大表示越相似。
   
   ```
   cosine_similarity(A, B) = (A·B) / (||A|| × ||B||)
   ```

2. **歐氏距離（Euclidean Distance）**：測量向量空間中兩點之間的直線距離，值越小表示越相似。
   
   ```
   euclidean_distance(A, B) = √(Σ(Aᵢ - Bᵢ)²)
   ```

3. **點積（Dot Product）**：兩個向量對應元素相乘後求和，通常在歸一化向量上使用。
   
   ```
   dot_product(A, B) = Σ(Aᵢ × Bᵢ)
   ```

4. **曼哈頓距離（Manhattan Distance）**：測量兩點在標準坐標系中的絕對距離總和。
   
   ```
   manhattan_distance(A, B) = Σ|Aᵢ - Bᵢ|
   ```

### 向量索引技術

為了高效處理大規模向量數據，向量資料庫使用特殊的索引結構，主要包括：

1. **精確最近鄰搜索（Exact Nearest Neighbor Search）**：返回與查詢向量最相似的確切結果，但在大規模數據集上計算成本高。

2. **近似最近鄰搜索（Approximate Nearest Neighbor Search, ANN）**：犧牲一定的精確度換取更高的搜索效率，適用於大規模數據集。常見的 ANN 算法包括：

   - **基於樹的方法**：如 KD-Tree、Ball Tree 和 VP-Tree 等，通過將空間劃分為多個區域來加速搜索。
   
   - **基於哈希的方法**：如局部敏感哈希（LSH），將相似的向量映射到相同的哈希桶中。
   
   - **基於圖的方法**：如階層可導航小世界圖（HNSW），通過構建多層圖結構實現高效搜索。
   
   - **基於量化的方法**：如乘積量化（PQ），通過壓縮向量表示減少存儲和計算開銷。

## 主流向量資料庫比較

市場上有多種向量資料庫解決方案，各有特點和適用場景：

### 1. Pinecone

Pinecone 是一個全託管的向量資料庫服務，專為機器學習和 AI 應用設計。

**主要特點：**
- 完全託管的雲服務，無需維護基礎設施
- 支持實時更新和查詢
- 提供自動擴展和高可用性
- 內置的向量索引和相似性搜索功能
- 支持元數據過濾

**適用場景：**
- 需要快速部署的企業應用
- 要求低延遲的實時推薦系統
- 需要無縫擴展的大規模 RAG 應用

```python
import pinecone
from sentence_transformers import SentenceTransformer

# 初始化 Pinecone
pinecone.init(api_key="your-api-key", environment="your-environment")

# 創建或連接到索引
index_name = "document-embeddings"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768, metric="cosine")
index = pinecone.Index(index_name)

# 嵌入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 生成嵌入
texts = ["向量資料庫是專門設計用於存儲和查詢向量數據的系統。", 
         "RAG 技術結合了檢索和生成能力。"]
embeddings = model.encode(texts)

# 上傳向量
vectors = [(f"id-{i}", embedding.tolist(), {"text": text}) 
           for i, (embedding, text) in enumerate(zip(embeddings, texts))]
index.upsert(vectors=vectors)

# 查詢
query = "什麼是向量資料庫？"
query_embedding = model.encode(query).tolist()
results = index.query(query_embedding, top_k=1, include_metadata=True)
print(results)
```

### 2. Milvus

Milvus 是一個開源的向量資料庫，專注於大規模相似性搜索和分析。

**主要特點：**
- 開源，可自行部署
- 支持多種索引類型（FLAT、IVF、HNSW 等）
- 提供水平擴展能力
- 支持混合搜索（向量 + 標量過濾）
- 提供雲託管版本 Zilliz Cloud

**適用場景：**
- 需要自定義部署和控制的企業
- 大規模圖像和視頻搜索
- 需要靈活索引選項的複雜應用

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np

# 連接到 Milvus
connections.connect("default", host="localhost", port="19530")

# 定義集合架構
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
]
schema = CollectionSchema(fields, "向量資料庫示例集合")

# 創建集合
collection_name = "document_collection"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
collection = Collection(name=collection_name, schema=schema)

# 創建索引
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64}
}
collection.create_index(field_name="embedding", index_params=index_params)

# 插入數據
ids = [1, 2]
embeddings = np.random.rand(2, 768).tolist()  # 實際應用中使用真實嵌入
texts = ["向量資料庫是專門設計用於存儲和查詢向量數據的系統。", 
         "RAG 技術結合了檢索和生成能力。"]
collection.insert([ids, embeddings, texts])

# 加載集合
collection.load()

# 搜索
search_params = {"metric_type": "COSINE", "params": {"ef": 16}}
results = collection.search(
    data=[np.random.rand(768).tolist()],  # 實際應用中使用查詢嵌入
    anns_field="embedding",
    param=search_params,
    limit=1,
    output_fields=["text"]
)
print(results)
```

### 3. Faiss (Facebook AI Similarity Search)

Faiss 是由 Facebook AI Research 開發的高性能向量相似性搜索庫。

**主要特點：**
- 專注於高效的相似性搜索
- 支持 CPU 和 GPU 加速
- 提供多種索引類型和量化方法
- 輕量級，可嵌入到其他應用中
- 不提供持久化存儲，需要與其他存儲系統結合

**適用場景：**
- 需要極高搜索性能的應用
- 研究和原型開發
- 與現有系統集成的嵌入式解決方案

```python
import faiss
import numpy as np

# 創建示例向量
dimension = 768
num_vectors = 1000
vectors = np.random.random((num_vectors, dimension)).astype('float32')

# 創建索引
index = faiss.IndexFlatL2(dimension)  # L2 距離索引
print(f"索引是否已訓練: {index.is_trained}")
index.add(vectors)  # 添加向量到索引
print(f"索引包含的向量數量: {index.ntotal}")

# 搜索
k = 5  # 返回前 5 個最相似結果
query_vector = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query_vector, k)
print(f"查詢結果 - 距離: {distances}, 索引: {indices}")

# 使用 IVF 索引提高大規模搜索效率
nlist = 100  # 聚類數量
quantizer = faiss.IndexFlatL2(dimension)
index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

# IVF 索引需要訓練
index_ivf.train(vectors)
index_ivf.add(vectors)

# 設置搜索參數
index_ivf.nprobe = 10  # 搜索時檢查的聚類數量
distances, indices = index_ivf.search(query_vector, k)
print(f"IVF 查詢結果 - 距離: {distances}, 索引: {indices}")
```

### 4. Chroma

Chroma 是一個為 RAG 應用專門設計的開源嵌入式向量資料庫。

**主要特點：**
- 專為 LLM 應用設計
- 簡單易用的 API
- 支持多種嵌入模型
- 提供內存和持久化存儲選項
- 與 LangChain 等框架無縫集成

**適用場景：**
- 快速原型開發
- 小型到中型 RAG 應用
- 需要簡單部署的項目

```python
import chromadb
from chromadb.utils import embedding_functions

# 創建客戶端
client = chromadb.Client()

# 定義嵌入函數
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 創建集合
collection = client.create_collection(
    name="documents",
    embedding_function=embedding_function
)

# 添加文檔
collection.add(
    documents=["向量資料庫是專門設計用於存儲和查詢向量數據的系統。", 
               "RAG 技術結合了檢索和生成能力。"],
    metadatas=[{"source": "textbook"}, {"source": "article"}],
    ids=["doc1", "doc2"]
)

# 查詢
results = collection.query(
    query_texts=["什麼是向量資料庫？"],
    n_results=1
)
print(results)
```

### 5. Weaviate

Weaviate 是一個開源的向量搜索引擎，結合了向量搜索和圖數據庫功能。

**主要特點：**
- 結合向量搜索和圖數據庫功能
- 支持 GraphQL 查詢語言
- 提供模塊化架構和插件系統
- 支持多模態數據（文本、圖像等）
- 提供雲託管選項

**適用場景：**
- 需要複雜數據關係的應用
- 多模態搜索系統
- 需要靈活查詢能力的企業應用

```python
import weaviate
from weaviate.auth import AuthApiKey

# 連接到 Weaviate
client = weaviate.Client(
    url="http://localhost:8080",
    # 如果使用雲服務，添加認證
    # auth_client_secret=AuthApiKey(api_key="your-api-key")
)

# 定義類別架構
class_obj = {
    "class": "Document",
    "vectorizer": "text2vec-transformers",  # 使用本地 transformers 模型
    "properties": [
        {
            "name": "content",
            "dataType": ["text"]
        },
        {
            "name": "source",
            "dataType": ["string"]
        }
    ]
}

# 創建類別
if not client.schema.contains({"classes": [class_obj]}):
    client.schema.create_class(class_obj)

# 添加數據
client.batch.configure(batch_size=2)
with client.batch as batch:
    batch.add_data_object(
        {"content": "向量資料庫是專門設計用於存儲和查詢向量數據的系統。", "source": "textbook"},
        "Document"
    )
    batch.add_data_object(
        {"content": "RAG 技術結合了檢索和生成能力。", "source": "article"},
        "Document"
    )

# 查詢
response = (
    client.query
    .get("Document", ["content", "source"])
    .with_near_text({"concepts": ["向量資料庫"]})
    .with_limit(1)
    .do()
)
print(response)
```

## 向量資料庫選擇考量因素

在選擇向量資料庫時，應考慮以下因素：

1. **規模需求**：數據量和查詢負載決定了所需的系統規模。

2. **部署偏好**：自託管 vs. 雲服務，取決於團隊資源和專業知識。

3. **性能要求**：查詢延遲和吞吐量對應用體驗至關重要。

4. **功能需求**：元數據過濾、多模態支持、實時更新等特定功能需求。

5. **集成需求**：與現有技術棧和框架的兼容性。

6. **成本考量**：開源解決方案可能有較低的直接成本，但可能需要更多維護資源。

7. **可擴展性**：隨著數據增長，系統應能夠平滑擴展。

## 資訊檢索系統架構

資訊檢索（Information Retrieval，IR）系統是 RAG 應用的核心組件，負責從大量文檔中找出與用戶查詢最相關的內容。

### 傳統 IR 系統

傳統 IR 系統主要基於關鍵詞匹配和統計模型：

1. **布爾檢索**：使用邏輯運算符（AND、OR、NOT）組合關鍵詞進行精確匹配。

2. **向量空間模型**：使用 TF-IDF（詞頻-逆文檔頻率）將文檔和查詢表示為向量，計算相似度。

3. **概率模型**：如 BM25 算法，基於概率理論計算文檔與查詢的相關性。

### 現代神經 IR 系統

現代 IR 系統利用深度學習模型捕捉語義關係：

1. **雙塔模型（Bi-Encoder）**：使用兩個獨立的編碼器分別編碼查詢和文檔，計算它們的向量表示之間的相似度。適合大規模檢索，但可能損失一些交互信息。

2. **交叉編碼器（Cross-Encoder）**：將查詢和文檔作為一個整體輸入到模型中，直接預測相關性分數。提供更高的精度，但計算成本較高。

3. **混合架構**：結合雙塔模型的效率和交叉編碼器的精度，通常採用多階段檢索策略。

### 多階段檢索架構

現代 RAG 系統通常採用多階段檢索架構，平衡效率和精度：

1. **第一階段：粗檢索**
   - 使用高效的向量搜索或關鍵詞匹配快速縮小候選範圍
   - 通常使用雙塔模型或 BM25 等算法
   - 優化召回率，確保相關文檔被包含在候選集中

2. **第二階段：精排序**
   - 對第一階段檢索的候選文檔進行更精細的排序
   - 通常使用交叉編碼器或更複雜的排序模型
   - 優化精確度，確保最相關的文檔排在前面

3. **第三階段：重排序和融合**
   - 可能結合多種信號（語義相似度、關鍵詞匹配、流行度等）
   - 應用領域知識和業務規則
   - 生成最終的檢索結果列表

```python
# 多階段檢索示例
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 第一階段：向量檢索（雙塔模型）
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')

# 示例文檔集合
documents = [
    "向量資料庫是專門設計用於存儲和查詢向量數據的系統。",
    "RAG 技術結合了檢索和生成能力，提高了 LLM 的準確性。",
    "大型語言模型通過自注意力機制處理序列數據。",
    "資訊檢索系統負責從大量文檔中找出相關內容。",
    "向量嵌入將文本轉換為數值表示，捕捉語義信息。"
]

# 預計算文檔嵌入
doc_embeddings = bi_encoder.encode(documents)

# 用戶查詢
query = "什麼是向量資料庫？"
query_embedding = bi_encoder.encode(query)

# 第一階段：計算相似度並選擇前 3 個候選文檔
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
top_k_indices = np.argsort(similarities)[-3:][::-1]
candidates = [documents[i] for i in top_k_indices]
print("第一階段候選文檔:", candidates)

# 第二階段：使用交叉編碼器進行精排序
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [[query, doc] for doc in candidates]
scores = cross_encoder.predict(pairs)

# 根據交叉編碼器分數重新排序
ranked_results = [candidates[i] for i in np.argsort(scores)[::-1]]
print("第二階段排序結果:", ranked_results)
```

### 檢索增強技術

為了提高檢索效果，現代 IR 系統採用多種增強技術：

1. **查詢擴展**：通過添加同義詞、相關詞或上下文信息擴展原始查詢。

2. **查詢重寫**：使用 LLM 重新表述查詢，生成多個變體以提高檢索覆蓋率。

3. **混合檢索**：結合稀疏檢索（關鍵詞匹配）和密集檢索（向量搜索）的結果。

4. **上下文感知檢索**：考慮用戶的歷史查詢和交互，提供更個性化的結果。

5. **多模態檢索**：整合文本、圖像、音頻等多種模態的信息。

## 向量資料庫與 RAG 系統集成

向量資料庫是 RAG 系統的關鍵組件，負責存儲和檢索文檔嵌入。以下是集成向量資料庫與 RAG 系統的最佳實踐：

### 1. 數據預處理與索引建立

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 加載文檔
loader = DirectoryLoader('./data/', glob="**/*.pdf", show_progress=True)
documents = loader.load()

# 文檔分塊
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)

# 初始化嵌入模型
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'}
)

# 創建向量存儲
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
vectorstore.persist()
```

### 2. RAG 查詢流程

```python
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# 加載持久化的向量存儲
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# 創建檢索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# 定義 RAG 提示模板
template = """
使用以下檢索到的上下文來回答問題。如果你不知道答案，就說你不知道，不要試圖編造答案。

上下文：
{context}

問題：{question}

回答：
"""
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 初始化 LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    task="text-generation",
    model_kwargs={"temperature": 0.1, "max_length": 512}
)

# 創建 RAG 鏈
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# 查詢 RAG 系統
query = "向量資料庫在 RAG 系統中扮演什麼角色？"
result = qa_chain({"query": query})
print(result["result"])
print("\n來源文檔:")
for i, doc in enumerate(result["source_documents"]):
    print(f"文檔 {i+1}:\n{doc.page_content}\n")
```

### 3. 混合檢索策略

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# 創建 BM25 檢索器
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# 創建向量檢索器
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 創建集成檢索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)

# 使用集成檢索器
docs = ensemble_retriever.get_relevant_documents(query)
```

## 結論

向量資料庫和資訊檢索系統是現代 RAG 應用的基石，它們使 LLM 能夠訪問和利用外部知識，生成更準確、更可靠的回應。隨著技術的發展，向量資料庫正變得越來越高效、可擴展和易於使用，為 AI 應用開發者提供了強大的工具。

在選擇和實施向量資料庫解決方案時，開發者應根據具體應用需求、性能要求和資源限制做出權衡。無論選擇哪種解決方案，理解向量索引的基本原理和資訊檢索的最佳實踐都是構建高效 RAG 系統的關鍵。

隨著多模態 AI 的興起，向量資料庫也在擴展其能力，支持圖像、音頻和視頻等非文本數據的索引和檢索，為更豐富的 AI 應用場景鋪平道路。
