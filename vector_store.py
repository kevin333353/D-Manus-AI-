"""
向量存儲模組 - 負責文檔嵌入和向量存儲
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStore:
    """向量存儲類，負責文檔嵌入和向量索引"""
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "../data/vector_store",
    ):
        """
        初始化向量存儲
        
        Args:
            embedding_model_name: 嵌入模型名稱
            persist_directory: 向量存儲持久化目錄
        """
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        self.embedding_model = self._init_embedding_model()
        self.vector_store = None
    
    def _init_embedding_model(self) -> HuggingFaceEmbeddings:
        """初始化嵌入模型"""
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'}  # 使用CPU以節省資源
        )
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        從文檔創建向量存儲
        
        Args:
            documents: 文檔列表
        """
        print(f"正在為 {len(documents)} 個文檔創建向量存儲...")
        
        # 創建向量存儲
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        
        # 確保目錄存在
        os.makedirs(os.path.dirname(self.persist_directory), exist_ok=True)
        
        # 持久化向量存儲
        self.vector_store.save_local(self.persist_directory)
        print(f"向量存儲已保存到 {self.persist_directory}")
    
    def load_vector_store(self) -> bool:
        """
        加載向量存儲
        
        Returns:
            是否成功加載
        """
        if os.path.exists(self.persist_directory):
            print(f"正在從 {self.persist_directory} 加載向量存儲...")
            self.vector_store = FAISS.load_local(
                self.persist_directory,
                self.embedding_model
            )
            return True
        else:
            print(f"向量存儲 {self.persist_directory} 不存在")
            return False
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        向向量存儲添加文檔
        
        Args:
            documents: 要添加的文檔列表
        """
        if self.vector_store is None:
            # 如果向量存儲不存在，則創建新的
            self.create_vector_store(documents)
        else:
            # 向現有向量存儲添加文檔
            print(f"正在向向量存儲添加 {len(documents)} 個文檔...")
            self.vector_store.add_documents(documents)
            self.vector_store.save_local(self.persist_directory)
            print("文檔已添加並保存")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        執行相似性搜索
        
        Args:
            query: 查詢文本
            k: 返回的結果數量
            filter: 過濾條件
            
        Returns:
            文檔和相似度分數的列表
            
        Raises:
            ValueError: 向量存儲未初始化
        """
        if self.vector_store is None:
            raise ValueError("向量存儲未初始化，請先創建或加載向量存儲")
        
        print(f"正在搜索: '{query}'")
        docs_with_scores = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
        
        return docs_with_scores
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        獲取檢索器
        
        Args:
            search_kwargs: 搜索參數
            
        Returns:
            檢索器對象
            
        Raises:
            ValueError: 向量存儲未初始化
        """
        if self.vector_store is None:
            raise ValueError("向量存儲未初始化，請先創建或加載向量存儲")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)


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
    
    # 測試搜索
    query = "什麼是深度學習？"
    results = vector_store.similarity_search(query, k=2)
    
    # 打印搜索結果
    print("\n搜索結果:")
    for i, (doc, score) in enumerate(results):
        print(f"\n--- 結果 {i+1} (相似度: {score:.4f}) ---")
        print(doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content)
