# LangChain 與 AI Agent 開發指南

## LangChain 基礎概念

LangChain 是一個專為語言模型應用開發設計的框架，旨在簡化基於大型語言模型 (LLM) 的應用程序構建過程。它提供了一套統一的接口和工具，使開發者能夠輕鬆地將 LLM 與各種外部資源和系統整合，創建功能強大的 AI 應用。

### LangChain 核心組件

LangChain 框架由以下核心組件構成：

1. **模型 (Models)**：提供與各種 LLM 和嵌入模型交互的統一接口，支持 OpenAI、Anthropic、Hugging Face、本地模型等多種選項。

2. **提示 (Prompts)**：管理和優化發送給 LLM 的提示，包括提示模板、提示優化和提示序列化。

3. **記憶 (Memory)**：實現對話歷史的存儲和檢索，支持多種記憶類型，如對話緩衝記憶、摘要記憶和向量存儲記憶等。

4. **索引 (Indexes)**：處理和索引外部數據，使 LLM 能夠訪問和利用這些數據，包括文檔加載器、文本分割器和向量存儲等。

5. **鏈 (Chains)**：將多個組件組合成端到端應用的序列，如問答鏈、摘要鏈和對話鏈等。

6. **工具 (Tools)**：連接 LLM 與外部系統和 API 的接口，如搜索引擎、計算器和數據庫等。

7. **代理 (Agents)**：實現 LLM 作為代理，能夠根據用戶輸入選擇和使用工具，執行複雜任務。

### LangChain 基本用法

以下是 LangChain 的一些基本用法示例：

#### 1. 簡單 LLM 調用

```python
from langchain.llms import OpenAI

# 初始化 LLM
llm = OpenAI(temperature=0.7)

# 直接調用 LLM
response = llm("解釋量子計算的基本原理")
print(response)
```

#### 2. 使用提示模板

```python
from langchain.prompts import PromptTemplate

# 定義提示模板
template = """
請提供關於 {topic} 的詳細解釋，包括以下方面：
1. 基本概念
2. 歷史發展
3. 應用場景
4. 未來趨勢
"""

prompt = PromptTemplate(
    input_variables=["topic"],
    template=template,
)

# 使用模板生成提示
formatted_prompt = prompt.format(topic="人工智能")

# 將格式化的提示發送給 LLM
response = llm(formatted_prompt)
print(response)
```

#### 3. 構建簡單鏈

```python
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# 初始化 LLM
llm = OpenAI(temperature=0.7)

# 定義提示模板
template = "請為一家名為 {company_name} 的 {industry} 公司撰寫一段簡短的宣傳文案。"
prompt = PromptTemplate(
    input_variables=["company_name", "industry"],
    template=template,
)

# 創建 LLM 鏈
chain = LLMChain(llm=llm, prompt=prompt)

# 運行鏈
response = chain.run(company_name="TechInnovate", industry="人工智能")
print(response)
```

## 使用 LangChain 構建 RAG 系統

LangChain 提供了豐富的工具和組件，使構建 RAG (檢索增強生成) 系統變得簡單高效。

### 基本 RAG 系統實現

以下是使用 LangChain 構建基本 RAG 系統的步驟：

#### 1. 加載和處理文檔

```python
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 加載文檔
loader = DirectoryLoader('./data/', glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# 文檔分塊
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
```

#### 2. 創建向量存儲

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 初始化嵌入模型
embeddings = OpenAIEmbeddings()

# 創建向量存儲
vectorstore = Chroma.from_documents(chunks, embeddings)

# 創建檢索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

#### 3. 構建 RAG 鏈

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 初始化 LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 創建 RAG 鏈
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 其他選項: map_reduce, refine
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

# 查詢 RAG 系統
query = "人工智能對醫療行業有哪些影響？"
result = qa_chain({"query": query})
print(result["result"])
```

### 進階 RAG 技術實現

LangChain 支持多種進階 RAG 技術，以下是一些示例：

#### 1. 查詢轉換

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI

# 初始化 LLM 用於查詢轉換
llm = ChatOpenAI(temperature=0)

# 創建多查詢檢索器
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# 使用多查詢檢索
docs = retriever.get_relevant_documents(query)
```

#### 2. 上下文壓縮

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 初始化 LLM
llm = ChatOpenAI(temperature=0)

# 創建文檔壓縮器
compressor = LLMChainExtractor.from_llm(llm)

# 創建上下文壓縮檢索器
compression_retriever = ContextualCompressionRetriever(
    base_retriever=vectorstore.as_retriever(),
    doc_compressor=compressor
)

# 檢索並壓縮文檔
compressed_docs = compression_retriever.get_relevant_documents(query)
```

#### 3. 自定義 RAG 提示

```python
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 自定義 RAG 提示模板
template = """
使用以下檢索到的上下文來回答問題。如果你不知道答案，就說你不知道，不要試圖編造答案。

上下文：
{context}

問題：{question}

回答：
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 使用自定義提示創建 RAG 鏈
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
```

## AI Agent 開發

LangChain 的 Agent 框架允許 LLM 作為代理，能夠根據用戶輸入選擇和使用工具，執行複雜任務。

### Agent 基本概念

Agent 由以下核心組件構成：

1. **LLM**：作為 Agent 的大腦，負責理解用戶輸入並決定使用哪些工具。

2. **工具 (Tools)**：Agent 可以使用的功能集合，如搜索引擎、計算器、API 等。

3. **工具選擇器 (Tool Selector)**：決定使用哪個工具的機制，通常由 LLM 實現。

4. **代理執行器 (Agent Executor)**：協調 LLM 和工具之間的交互，管理整個執行流程。

### 構建基本 Agent

以下是使用 LangChain 構建基本 Agent 的示例：

```python
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

# 初始化 LLM
llm = OpenAI(temperature=0)

# 加載工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 初始化 Agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 運行 Agent
agent.run("2023年世界人口最多的國家是哪個？它的人口數量是多少？這個數字是第二多的國家的多少倍？")
```

### 自定義工具

開發者可以創建自定義工具擴展 Agent 的能力：

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

# 定義工具輸入架構
class WeatherInput(BaseModel):
    location: str = Field(description="城市名稱，如 '北京' 或 '上海'")

# 創建自定義天氣工具
class WeatherTool(BaseTool):
    name = "weather_tool"
    description = "獲取指定城市的當前天氣信息"
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, location: str) -> str:
        # 這裡應該是實際的天氣 API 調用
        # 為了示例，返回模擬數據
        return f"{location}的當前天氣：晴朗，溫度25°C，濕度60%"
        
    def _arun(self, location: str):
        # 異步版本
        raise NotImplementedError("暫不支持異步操作")

# 將自定義工具添加到 Agent
tools = [WeatherTool()]
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 運行 Agent
agent.run("北京今天的天氣怎麼樣？")
```

### 高級 Agent 模式

LangChain 支持多種高級 Agent 模式：

#### 1. ReAct Agent

結合推理 (Reasoning) 和行動 (Acting)，通過思考-行動-觀察循環解決問題。

```python
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

#### 2. 自反式 Agent

能夠反思自己的行動並改進策略的 Agent。

```python
from langchain.experimental.autonomous_agents import AutoGPT
from langchain.tools import DuckDuckGoSearchRun

# 初始化搜索工具
search = DuckDuckGoSearchRun()

# 創建工具列表
tools = [search]

# 初始化 AutoGPT
agent = AutoGPT.from_llm_and_tools(
    ai_name="ResearchGPT",
    ai_role="研究助手",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever()
)

# 運行 Agent
agent.run(["研究量子計算的最新進展"])
```

#### 3. 多 Agent 協作

多個 Agent 協同工作解決複雜問題：

```python
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish

# 創建專家 Agent（示例）
research_agent = initialize_agent(
    [search], 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

writing_agent = initialize_agent(
    [], 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 創建協調 Agent
tools = [
    Tool(
        name="ResearchAgent",
        func=lambda q: research_agent.run(q),
        description="用於研究和收集信息的代理"
    ),
    Tool(
        name="WritingAgent",
        func=lambda q: writing_agent.run(q),
        description="用於撰寫和編輯文本的代理"
    )
]

coordinator_agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 運行協調 Agent
coordinator_agent.run("撰寫一篇關於人工智能最新發展的文章")
```

## LangChain 最佳實踐

### 1. 提示工程

- 使用提示模板而非硬編碼提示
- 包含少量示例（少樣本學習）
- 明確指定輸出格式
- 使用結構化提示進行複雜任務

### 2. 鏈和代理設計

- 從簡單鏈開始，逐步增加複雜性
- 使用 `verbose=True` 調試鏈和代理
- 考慮使用 LangChain 的回調系統進行監控和日誌記錄
- 為長時間運行的任務實現中間結果保存

### 3. 記憶管理

- 選擇適合應用場景的記憶類型
- 對於長對話，考慮使用摘要記憶或向量存儲記憶
- 實現記憶清理機制，避免上下文過長

### 4. 錯誤處理

- 實現重試機制處理 API 限制和臨時錯誤
- 為 Agent 設置最大迭代次數，避免無限循環
- 使用結構化輸出解析器處理 LLM 輸出格式錯誤

### 5. 性能優化

- 使用異步 API 提高並發性能
- 實現結果緩存減少重複 API 調用
- 對於大規模應用，考慮使用分佈式向量存儲

## 結論

LangChain 為開發基於 LLM 的應用提供了強大而靈活的框架，特別適合構建 RAG 系統和 AI Agent。通過掌握 LangChain 的核心組件和最佳實踐，開發者可以快速構建功能豐富、性能優越的 AI 應用，充分發揮 LLM 的潛力。

隨著 LangChain 生態系統的不斷發展，我們可以期待更多創新功能和工具的出現，進一步簡化 AI 應用開發流程，並擴展 LLM 的應用場景。對於希望在 AI 領域保持競爭力的開發者來說，深入學習和掌握 LangChain 是一項值得投資的技能。
