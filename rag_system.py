"""
檢索增強生成模組 - 負責查詢處理、檢索和生成回應
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers.document_compressors import LLMChainExtractor

from vector_store import VectorStore

class RAGSystem:
    """檢索增強生成系統，負責查詢處理、檢索和生成回應"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        model_name: str = "google/flan-t5-base",
        use_local_model: bool = True,
        temperature: float = 0.1,
    ):
        """
        初始化RAG系統
        
        Args:
            vector_store: 向量存儲對象
            model_name: 模型名稱
            use_local_model: 是否使用本地模型
            temperature: 溫度參數
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.use_local_model = use_local_model
        self.temperature = temperature
        self.llm = self._init_llm()
        self.retriever = self._init_retriever()
        self.qa_chain = self._init_qa_chain()
    
    def _init_llm(self):
        """初始化語言模型"""
        if self.use_local_model:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
                
                # 使用本地模型
                print(f"正在加載本地模型: {self.model_name}")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                
                # 創建文本生成管道
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=512,
                    temperature=self.temperature
                )
                
                # 創建HuggingFacePipeline
                return HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                print(f"加載本地模型失敗: {str(e)}")
                print("將使用OpenAI模型作為備選")
                return ChatOpenAI(temperature=self.temperature, model_name="gpt-3.5-turbo")
        else:
            # 使用OpenAI模型
            return ChatOpenAI(temperature=self.temperature, model_name="gpt-3.5-turbo")
    
    def _init_retriever(self):
        """初始化檢索器"""
        base_retriever = self.vector_store.get_retriever(search_kwargs={"k": 5})
        
        try:
            # 創建上下文壓縮器
            compressor = LLMChainExtractor.from_llm(self.llm)
            
            # 創建上下文壓縮檢索器
            return ContextualCompressionRetriever(
                base_retriever=base_retriever,
                doc_compressor=compressor
            )
        except Exception as e:
            print(f"創建上下文壓縮檢索器失敗: {str(e)}")
            print("將使用基本檢索器")
            return base_retriever
    
    def _init_qa_chain(self):
        """初始化問答鏈"""
        # 定義提示模板
        template = """
        你是一個專業的知識助手。請使用以下檢索到的上下文來回答問題。
        
        如果你無法從上下文中找到答案，請直接說"我無法從提供的信息中找到答案"，不要試圖編造答案。
        
        回答時，請引用信息來源（用方括號標註，如 [文檔1]）。
        
        上下文：
        {context}
        
        問題：{question}
        
        回答：
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # 創建 QA 鏈
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        查詢RAG系統
        
        Args:
            question: 問題文本
            
        Returns:
            包含回答和來源文檔的字典
        """
        print(f"處理查詢: '{question}'")
        
        try:
            # 執行查詢
            result = self.qa_chain({"query": question})
            
            # 格式化結果
            formatted_result = {
                "query": question,
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
            
            return formatted_result
        except Exception as e:
            print(f"查詢處理失敗: {str(e)}")
            return {
                "query": question,
                "answer": f"處理查詢時出錯: {str(e)}",
                "source_documents": []
            }
    
    def process_query_with_sources(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        處理查詢並返回帶有來源的回答
        
        Args:
            question: 問題文本
            
        Returns:
            回答文本和來源文檔列表的元組
        """
        result = self.query(question)
        
        # 提取來源文檔信息
        sources = []
        for i, doc in enumerate(result.get("source_documents", [])):
            source_info = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "id": f"doc{i+1}"
            }
            sources.append(source_info)
        
        return result["answer"], sources


if __name__ == "__main__":
    # 測試代碼
    from document_processor import DocumentProcessor
    
    # 初始化文檔處理器
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    sample_dir = "../data/sample_docs"
    
    # 確保示例目錄存在
    os.makedirs(sample_dir, exist_ok=True)
    
    # 創建一個示例文檔用於測試
    sample_text = """
    # 人工智能簡介
    
    人工智能（Artificial Intelligence，簡稱AI）是計算機科學的一個分支，致力於開發能夠模擬人類智能的系統。
    
    ## 機器學習
    
    機器學習是AI的一個子領域，專注於開發能夠從數據中學習的算法。
    
    ### 深度學習
    
    深度學習是機器學習的一個分支，使用多層神經網絡處理複雜問題。
    
    ## 自然語言處理
    
    自然語言處理（NLP）是AI的另一個重要領域，專注於使計算機理解和生成人類語言。
    """
    
    sample_file = os.path.join(sample_dir, "ai_intro.md")
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    # 處理文檔
    chunks = processor.process_documents(sample_dir)
    
    # 初始化向量存儲
    vector_store = VectorStore(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        persist_directory="../data/vector_store"
    )
    
    # 創建向量存儲
    vector_store.create_vector_store(chunks)
    
    # 初始化RAG系統
    rag_system = RAGSystem(
        vector_store=vector_store,
        model_name="google/flan-t5-base",
        use_local_model=True
    )
    
    # 測試查詢
    questions = [
        "什麼是深度學習？",
        "自然語言處理的主要目標是什麼？",
        "機器學習和深度學習有什麼關係？"
    ]
    
    for question in questions:
        print("\n" + "="*50)
        print(f"問題: {question}")
        answer, sources = rag_system.process_query_with_sources(question)
        print(f"\n回答: {answer}")
        print("\n來源文檔:")
        for source in sources:
            print(f"- {source['content'][:100]}...")
