"""
主程式 - 整合RAG系統的所有組件並提供使用介面
"""

import os
import argparse
from typing import List, Dict, Any, Optional

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_system import RAGSystem

def setup_sample_data(sample_dir: str):
    """設置示例數據"""
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
    
    ## 計算機視覺
    
    計算機視覺是AI的一個分支，專注於使計算機能夠理解和解釋視覺信息。
    
    ## 強化學習
    
    強化學習是一種機器學習方法，通過與環境互動並從反饋中學習來優化決策。
    """
    
    sample_file = os.path.join(sample_dir, "ai_intro.md")
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    print(f"已創建示例文檔: {sample_file}")
    
    # 創建第二個示例文檔
    sample_text2 = """
    # 大型語言模型簡介
    
    大型語言模型（Large Language Models，簡稱LLM）是一種基於深度學習的自然語言處理模型，通過大規模預訓練學習語言的統計規律。
    
    ## 預訓練與微調
    
    LLM通常採用預訓練-微調的範式，先在大規模語料庫上進行無監督學習，再在特定任務上進行有監督微調。
    
    ## 主要應用
    
    LLM的應用非常廣泛，包括：
    
    1. 文本生成
    2. 問答系統
    3. 摘要生成
    4. 機器翻譯
    5. 代碼生成
    
    ## 挑戰與局限
    
    LLM面臨的主要挑戰包括：
    
    1. 幻覺問題
    2. 偏見與公平性
    3. 計算資源需求
    4. 上下文長度限制
    """
    
    sample_file2 = os.path.join(sample_dir, "llm_intro.md")
    with open(sample_file2, "w", encoding="utf-8") as f:
        f.write(sample_text2)
    
    print(f"已創建示例文檔: {sample_file2}")

def build_knowledge_base(input_dir: str, vector_store_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """構建知識庫"""
    # 初始化文檔處理器
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # 處理文檔
    chunks = processor.process_documents(input_dir)
    print(f"共處理了 {len(chunks)} 個文本塊")
    
    # 初始化向量存儲
    vector_store = VectorStore(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        persist_directory=vector_store_dir
    )
    
    # 創建向量存儲
    vector_store.create_vector_store(chunks)
    print(f"向量存儲已保存到 {vector_store_dir}")
    
    return vector_store

def initialize_rag_system(vector_store: VectorStore, model_name: str = "google/flan-t5-base", use_local_model: bool = True):
    """初始化RAG系統"""
    # 創建RAG系統
    rag_system = RAGSystem(
        vector_store=vector_store,
        model_name=model_name,
        use_local_model=use_local_model
    )
    
    return rag_system

def interactive_mode(rag_system: RAGSystem):
    """交互模式"""
    print("\n" + "="*50)
    print("歡迎使用RAG問答系統！輸入'exit'或'quit'退出。")
    print("="*50)
    
    while True:
        # 獲取用戶輸入
        query = input("\n請輸入您的問題: ")
        
        # 檢查是否退出
        if query.lower() in ['exit', 'quit']:
            print("謝謝使用，再見！")
            break
        
        # 處理查詢
        answer, sources = rag_system.process_query_with_sources(query)
        
        # 顯示結果
        print("\n回答:")
        print(answer)
        
        # 顯示來源
        if sources:
            print("\n參考來源:")
            for i, source in enumerate(sources):
                print(f"[{i+1}] {source['content'][:150]}..." if len(source['content']) > 150 else source['content'])
        else:
            print("\n沒有找到相關來源")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="RAG系統")
    parser.add_argument("--build", action="store_true", help="構建知識庫")
    parser.add_argument("--query", type=str, help="查詢模式")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--sample", action="store_true", help="創建示例數據")
    parser.add_argument("--input_dir", type=str, default="../data/sample_docs", help="輸入目錄")
    parser.add_argument("--vector_store_dir", type=str, default="../data/vector_store", help="向量存儲目錄")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base", help="模型名稱")
    parser.add_argument("--use_local_model", action="store_true", help="使用本地模型")
    
    args = parser.parse_args()
    
    # 創建示例數據
    if args.sample:
        setup_sample_data(args.input_dir)
    
    # 構建知識庫
    if args.build or args.query or args.interactive:
        # 檢查向量存儲是否存在
        vector_store = VectorStore(
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            persist_directory=args.vector_store_dir
        )
        
        if not vector_store.load_vector_store() or args.build:
            print("構建知識庫...")
            vector_store = build_knowledge_base(args.input_dir, args.vector_store_dir)
    
    # 查詢模式
    if args.query:
        rag_system = initialize_rag_system(vector_store, args.model_name, args.use_local_model)
        answer, sources = rag_system.process_query_with_sources(args.query)
        
        print("\n問題:", args.query)
        print("\n回答:", answer)
        
        if sources:
            print("\n參考來源:")
            for i, source in enumerate(sources):
                print(f"[{i+1}] {source['content'][:150]}..." if len(source['content']) > 150 else source['content'])
    
    # 交互模式
    if args.interactive:
        rag_system = initialize_rag_system(vector_store, args.model_name, args.use_local_model)
        interactive_mode(rag_system)

if __name__ == "__main__":
    main()
