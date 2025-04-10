"""
文檔處理模組 - 負責文檔的加載、解析和分塊
"""

import os
import re
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)

# 文檔加載器映射表
LOADER_MAPPING = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".md": UnstructuredMarkdownLoader,
}

class DocumentProcessor:
    """文檔處理類，負責文檔的加載、解析和分塊"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        初始化文檔處理器
        
        Args:
            chunk_size: 文檔分塊大小
            chunk_overlap: 分塊重疊大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = self._create_text_splitter()
    
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """創建文本分割器"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        根據文件擴展名選擇合適的加載器加載文檔
        
        Args:
            file_path: 文件路徑
            
        Returns:
            Document列表
            
        Raises:
            ValueError: 不支持的文件類型
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext in LOADER_MAPPING:
            loader = LOADER_MAPPING[ext](file_path)
            return loader.load()
        else:
            raise ValueError(f"不支持的文件類型: {ext}")
    
    def load_documents_from_directory(
        self, 
        directory_path: str,
        glob_pattern: str = "**/*.*",
        exclude_patterns: List[str] = None
    ) -> List[Document]:
        """
        從目錄中加載所有支持的文檔
        
        Args:
            directory_path: 目錄路徑
            glob_pattern: 文件匹配模式
            exclude_patterns: 排除的文件模式列表
            
        Returns:
            Document列表
        """
        documents = []
        exclude_patterns = exclude_patterns or []
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # 檢查是否應該排除此文件
                if any(re.search(pattern, file_path) for pattern in exclude_patterns):
                    continue
                
                ext = os.path.splitext(file)[1].lower()
                if ext in LOADER_MAPPING:
                    try:
                        docs = self.load_document(file_path)
                        documents.extend(docs)
                        print(f"已加載文檔: {file_path}")
                    except Exception as e:
                        print(f"加載文檔 {file_path} 時出錯: {str(e)}")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        將文檔分割為適合嵌入的塊
        
        Args:
            documents: 要分割的文檔列表
            
        Returns:
            分割後的Document列表
        """
        return self.text_splitter.split_documents(documents)
    
    def process_documents(
        self, 
        input_dir: str,
        output_dir: Optional[str] = None,
        save_processed: bool = False
    ) -> List[Document]:
        """
        處理目錄中的所有文檔：加載、分塊
        
        Args:
            input_dir: 輸入目錄
            output_dir: 輸出目錄（如果需要保存處理後的文檔）
            save_processed: 是否保存處理後的文檔
            
        Returns:
            處理後的Document列表
        """
        # 加載文檔
        documents = self.load_documents_from_directory(input_dir)
        print(f"共加載了 {len(documents)} 個文檔")
        
        # 分割文檔
        chunks = self.split_documents(documents)
        print(f"分割後共有 {len(chunks)} 個文本塊")
        
        # 保存處理後的文檔（可選）
        if save_processed and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, chunk in enumerate(chunks):
                output_path = os.path.join(output_dir, f"chunk_{i}.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(chunk.page_content)
            print(f"已將處理後的文檔保存到 {output_dir}")
        
        return chunks


if __name__ == "__main__":
    # 測試代碼
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    sample_dir = "../data/sample_docs"
    processed_dir = "../data/processed"
    
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
    chunks = processor.process_documents(sample_dir, processed_dir, save_processed=True)
    
    # 打印處理結果
    for i, chunk in enumerate(chunks):
        print(f"\n--- 文本塊 {i+1} ---")
        print(chunk.page_content[:150] + "..." if len(chunk.page_content) > 150 else chunk.page_content)
