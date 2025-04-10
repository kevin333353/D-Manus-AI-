"""
向量資料庫整合模組 - 負責整合多種向量資料庫
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VectorDBConfig:
    """向量資料庫配置類"""
    db_type: str = "faiss"                                # 資料庫類型: faiss, chroma, weaviate, milvus, qdrant
    collection_name: str = "document_collection"          # 集合名稱
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # 嵌入模型名稱
    persist_directory: str = "../data/vector_store"       # 持久化目錄
    host: Optional[str] = None                            # 主機地址 (用於遠程資料庫)
    port: Optional[int] = None                            # 端口 (用於遠程資料庫)
    api_key: Optional[str] = None                         # API密鑰 (用於雲服務)
    dimension: int = 384                                  # 向量維度
    distance_metric: str = "cosine"                       # 距離度量: cosine, euclidean, dot
    additional_config: Dict[str, Any] = field(default_factory=dict)  # 額外配置

class VectorDBFactory:
    """向量資料庫工廠類，負責創建不同類型的向量資料庫"""
    
    @staticmethod
    def create_vector_db(config: VectorDBConfig):
        """
        創建向量資料庫
        
        Args:
            config: 向量資料庫配置
            
        Returns:
            向量資料庫實例
        """
        db_type = config.db_type.lower()
        
        if db_type == "faiss":
            return FAISSVectorDB(config)
        elif db_type == "chroma":
            return ChromaVectorDB(config)
        elif db_type == "weaviate":
            return WeaviateVectorDB(config)
        elif db_type == "milvus":
            return MilvusVectorDB(config)
        elif db_type == "qdrant":
            return QdrantVectorDB(config)
        else:
            raise ValueError(f"不支持的向量資料庫類型: {db_type}")

class BaseVectorDB:
    """向量資料庫基類"""
    
    def __init__(self, config: VectorDBConfig):
        """
        初始化向量資料庫
        
        Args:
            config: 向量資料庫配置
        """
        self.config = config
        self.embedding_model = self._init_embedding_model()
        self.vector_store = None
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        return HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_name,
            model_kwargs={'device': 'cpu'}  # 使用CPU以節省資源
        )
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加文檔到向量資料庫
        
        Args:
            documents: 文檔列表
            
        Returns:
            是否成功
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        相似性搜索
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            filter: 過濾條件
            
        Returns:
            文檔和相似度分數的列表
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        獲取檢索器
        
        Args:
            search_kwargs: 搜索參數
            
        Returns:
            檢索器對象
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def persist(self) -> bool:
        """
        持久化向量資料庫
        
        Returns:
            是否成功
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def load(self) -> bool:
        """
        加載向量資料庫
        
        Returns:
            是否成功
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def delete_collection(self) -> bool:
        """
        刪除集合
        
        Returns:
            是否成功
        """
        raise NotImplementedError("子類必須實現此方法")

class FAISSVectorDB(BaseVectorDB):
    """FAISS向量資料庫"""
    
    def __init__(self, config: VectorDBConfig):
        """
        初始化FAISS向量資料庫
        
        Args:
            config: 向量資料庫配置
        """
        super().__init__(config)
        self.persist_directory = config.persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加文檔到向量資料庫
        
        Args:
            documents: 文檔列表
            
        Returns:
            是否成功
        """
        try:
            from langchain_community.vectorstores import FAISS
            
            if self.vector_store is None:
                # 創建新的向量存儲
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embedding_model
                )
                logger.info(f"已創建FAISS向量存儲，包含 {len(documents)} 個文檔")
            else:
                # 向現有向量存儲添加文檔
                self.vector_store.add_documents(documents)
                logger.info(f"已向FAISS向量存儲添加 {len(documents)} 個文檔")
            
            # 持久化
            self.persist()
            return True
        except Exception as e:
            logger.error(f"添加文檔到FAISS向量存儲時出錯: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        相似性搜索
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            filter: 過濾條件
            
        Returns:
            文檔和相似度分數的列表
        """
        if self.vector_store is None:
            if not self.load():
                raise ValueError("向量存儲未初始化，請先添加文檔或加載向量存儲")
        
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            logger.info(f"FAISS搜索 '{query}' 返回 {len(docs_with_scores)} 個結果")
            return docs_with_scores
        except Exception as e:
            logger.error(f"FAISS搜索時出錯: {str(e)}")
            return []
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        獲取檢索器
        
        Args:
            search_kwargs: 搜索參數
            
        Returns:
            檢索器對象
        """
        if self.vector_store is None:
            if not self.load():
                raise ValueError("向量存儲未初始化，請先添加文檔或加載向量存儲")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def persist(self) -> bool:
        """
        持久化向量資料庫
        
        Returns:
            是否成功
        """
        if self.vector_store is None:
            logger.warning("向量存儲為空，無法持久化")
            return False
        
        try:
            self.vector_store.save_local(self.persist_directory)
            logger.info(f"FAISS向量存儲已保存到 {self.persist_directory}")
            return True
        except Exception as e:
            logger.error(f"持久化FAISS向量存儲時出錯: {str(e)}")
            return False
    
    def load(self) -> bool:
        """
        加載向量資料庫
        
        Returns:
            是否成功
        """
        try:
            from langchain_community.vectorstores import FAISS
            
            if os.path.exists(self.persist_directory):
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embedding_model
                )
                logger.info(f"已從 {self.persist_directory} 加載FAISS向量存儲")
                return True
            else:
                logger.warning(f"FAISS向量存儲 {self.persist_directory} 不存在")
                return False
        except Exception as e:
            logger.error(f"加載FAISS向量存儲時出錯: {str(e)}")
            return False
    
    def delete_collection(self) -> bool:
        """
        刪除集合
        
        Returns:
            是否成功
        """
        try:
            import shutil
            
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                logger.info(f"已刪除FAISS向量存儲 {self.persist_directory}")
                self.vector_store = None
                return True
            else:
                logger.warning(f"FAISS向量存儲 {self.persist_directory} 不存在")
                return False
        except Exception as e:
            logger.error(f"刪除FAISS向量存儲時出錯: {str(e)}")
            return False

class ChromaVectorDB(BaseVectorDB):
    """Chroma向量資料庫"""
    
    def __init__(self, config: VectorDBConfig):
        """
        初始化Chroma向量資料庫
        
        Args:
            config: 向量資料庫配置
        """
        super().__init__(config)
        self.persist_directory = config.persist_directory
        self.collection_name = config.collection_name
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加文檔到向量資料庫
        
        Args:
            documents: 文檔列表
            
        Returns:
            是否成功
        """
        try:
            from langchain_community.vectorstores import Chroma
            
            if self.vector_store is None:
                # 創建新的向量存儲
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_model,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
                logger.info(f"已創建Chroma向量存儲，包含 {len(documents)} 個文檔")
            else:
                # 向現有向量存儲添加文檔
                self.vector_store.add_documents(documents)
                logger.info(f"已向Chroma向量存儲添加 {len(documents)} 個文檔")
            
            # 持久化
            self.persist()
            return True
        except Exception as e:
            logger.error(f"添加文檔到Chroma向量存儲時出錯: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        相似性搜索
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            filter: 過濾條件
            
        Returns:
            文檔和相似度分數的列表
        """
        if self.vector_store is None:
            if not self.load():
                raise ValueError("向量存儲未初始化，請先添加文檔或加載向量存儲")
        
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            logger.info(f"Chroma搜索 '{query}' 返回 {len(docs_with_scores)} 個結果")
            return docs_with_scores
        except Exception as e:
            logger.error(f"Chroma搜索時出錯: {str(e)}")
            return []
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        獲取檢索器
        
        Args:
            search_kwargs: 搜索參數
            
        Returns:
            檢索器對象
        """
        if self.vector_store is None:
            if not self.load():
                raise ValueError("向量存儲未初始化，請先添加文檔或加載向量存儲")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def persist(self) -> bool:
        """
        持久化向量資料庫
        
        Returns:
            是否成功
        """
        if self.vector_store is None:
            logger.warning("向量存儲為空，無法持久化")
            return False
        
        try:
            self.vector_store.persist()
            logger.info(f"Chroma向量存儲已持久化到 {self.persist_directory}")
            return True
        except Exception as e:
            logger.error(f"持久化Chroma向量存儲時出錯: {str(e)}")
            return False
    
    def load(self) -> bool:
        """
        加載向量資料庫
        
        Returns:
            是否成功
        """
        try:
            from langchain_community.vectorstores import Chroma
            
            if os.path.exists(self.persist_directory):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model,
                    collection_name=self.collection_name
                )
                logger.info(f"已從 {self.persist_directory} 加載Chroma向量存儲")
                return True
            else:
                logger.warning(f"Chroma向量存儲 {self.persist_directory} 不存在")
                return False
        except Exception as e:
            logger.error(f"加載Chroma向量存儲時出錯: {str(e)}")
            return False
    
    def delete_collection(self) -> bool:
        """
        刪除集合
        
        Returns:
            是否成功
        """
        try:
            if self.vector_store is not None:
                self.vector_store.delete_collection()
                logger.info(f"已刪除Chroma集合 {self.collection_name}")
                self.vector_store = None
                return True
            else:
                logger.warning("向量存儲為空，無法刪除集合")
                return False
        except Exception as e:
            logger.error(f"刪除Chroma集合時出錯: {str(e)}")
            return False

class WeaviateVectorDB(BaseVectorDB):
    """Weaviate向量資料庫"""
    
    def __init__(self, config: VectorDBConfig):
        """
        初始化Weaviate向量資料庫
        
        Args:
            config: 向量資料庫配置
        """
        super().__init__(config)
        self.host = config.host or "localhost"
        self.port = config.port or 8080
        self.collection_name = config.collection_name
        self.api_key = config.api_key
        self.additional_config = config.additional_config
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加文檔到向量資料庫
        
        Args:
            documents: 文檔列表
            
        Returns:
            是否成功
        """
        try:
            from langchain_community.vectorstores import Weaviate
            import weaviate
            
            # 創建Weaviate客戶端
            auth_config = None
            if self.api_key:
                auth_config = weaviate.auth.AuthApiKey(api_key=self.api_key)
            
            client = weaviate.Client(
                url=f"http://{self.host}:{self.port}",
                auth_client_secret=auth_config
            )
            
            # 檢查集合是否存在，如果不存在則創建
            if not client.schema.exists(self.collection_name):
                class_obj = {
                    "class": self.collection_name,
                    "vectorizer": "none",  # 使用自定義向量
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"]
                        },
                        {
                            "name": "metadata",
                            "dataType": ["text"]
                        }
                    ]
                }
                client.schema.create_class(class_obj)
                logger.info(f"已創建Weaviate集合 {self.collection_name}")
            
            # 創建向量存儲
            self.vector_store = Weaviate(
                client=client,
                index_name=self.collection_name,
                text_key="content",
                embedding=self.embedding_model,
                by_text=False
            )
            
            # 添加文檔
            self.vector_store.add_documents(documents)
            logger.info(f"已向Weaviate添加 {len(documents)} 個文檔")
            return True
        except Exception as e:
            logger.error(f"添加文檔到Weaviate時出錯: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        相似性搜索
        
        Args:
            query: 查詢文本
            k: 返回結果數量
        
(Content truncated due to size limit. Use line ranges to read in chunks)