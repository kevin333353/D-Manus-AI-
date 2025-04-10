"""
評估指標系統 - 負責評估RAG系統和LLM的性能
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# 確保nltk資源已下載
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """評估配置類"""
    metrics: List[str] = field(default_factory=lambda: ["relevance", "factuality", "coherence", "fluency"])  # 評估指標
    retrieval_metrics: List[str] = field(default_factory=lambda: ["precision", "recall", "ndcg", "mrr"])  # 檢索評估指標
    output_dir: str = "../evaluation_results"  # 輸出目錄
    test_set_path: Optional[str] = None  # 測試集路徑
    reference_answers_path: Optional[str] = None  # 參考答案路徑
    embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # 嵌入模型名稱
    num_samples: int = 100  # 樣本數量
    batch_size: int = 16  # 批次大小
    save_results: bool = True  # 是否保存結果
    verbose: bool = True  # 是否顯示詳細信息

class RetrievalEvaluator:
    """檢索評估器，負責評估檢索系統的性能"""
    
    def __init__(self, config: EvaluationConfig):
        """
        初始化檢索評估器
        
        Args:
            config: 評估配置
        """
        self.config = config
        self.metrics = config.retrieval_metrics
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化嵌入模型
        self.embedding_model = self._init_embedding_model()
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            return HuggingFaceEmbeddings(
                model_name=self.config.embeddings_model_name,
                model_kwargs={'device': 'cpu'}  # 使用CPU以節省資源
            )
        except Exception as e:
            logger.error(f"初始化嵌入模型時出錯: {str(e)}")
            return None
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        retrieved_docs: List[List[Document]],
        relevant_docs: List[List[Document]]
    ) -> Dict[str, float]:
        """
        評估檢索性能
        
        Args:
            queries: 查詢列表
            retrieved_docs: 檢索到的文檔列表的列表
            relevant_docs: 相關文檔列表的列表
            
        Returns:
            評估結果字典
        """
        results = {}
        
        # 計算各項指標
        if "precision" in self.metrics:
            results["precision"] = self._calculate_precision(retrieved_docs, relevant_docs)
        
        if "recall" in self.metrics:
            results["recall"] = self._calculate_recall(retrieved_docs, relevant_docs)
        
        if "ndcg" in self.metrics:
            results["ndcg"] = self._calculate_ndcg(retrieved_docs, relevant_docs)
        
        if "mrr" in self.metrics:
            results["mrr"] = self._calculate_mrr(retrieved_docs, relevant_docs)
        
        # 保存結果
        if self.config.save_results:
            self._save_retrieval_results(queries, retrieved_docs, relevant_docs, results)
        
        return results
    
    def _calculate_precision(
        self,
        retrieved_docs: List[List[Document]],
        relevant_docs: List[List[Document]]
    ) -> float:
        """
        計算精確率
        
        Args:
            retrieved_docs: 檢索到的文檔列表的列表
            relevant_docs: 相關文檔列表的列表
            
        Returns:
            精確率
        """
        precisions = []
        
        for i in range(len(retrieved_docs)):
            retrieved = retrieved_docs[i]
            relevant = relevant_docs[i]
            
            # 獲取檢索文檔和相關文檔的ID
            retrieved_ids = [self._get_doc_id(doc) for doc in retrieved]
            relevant_ids = [self._get_doc_id(doc) for doc in relevant]
            
            # 計算精確率
            if len(retrieved_ids) == 0:
                precisions.append(0.0)
            else:
                relevant_retrieved = [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]
                precisions.append(len(relevant_retrieved) / len(retrieved_ids))
        
        # 計算平均精確率
        return sum(precisions) / len(precisions) if precisions else 0.0
    
    def _calculate_recall(
        self,
        retrieved_docs: List[List[Document]],
        relevant_docs: List[List[Document]]
    ) -> float:
        """
        計算召回率
        
        Args:
            retrieved_docs: 檢索到的文檔列表的列表
            relevant_docs: 相關文檔列表的列表
            
        Returns:
            召回率
        """
        recalls = []
        
        for i in range(len(retrieved_docs)):
            retrieved = retrieved_docs[i]
            relevant = relevant_docs[i]
            
            # 獲取檢索文檔和相關文檔的ID
            retrieved_ids = [self._get_doc_id(doc) for doc in retrieved]
            relevant_ids = [self._get_doc_id(doc) for doc in relevant]
            
            # 計算召回率
            if len(relevant_ids) == 0:
                recalls.append(1.0)  # 如果沒有相關文檔，則召回率為1
            else:
                relevant_retrieved = [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]
                recalls.append(len(relevant_retrieved) / len(relevant_ids))
        
        # 計算平均召回率
        return sum(recalls) / len(recalls) if recalls else 0.0
    
    def _calculate_ndcg(
        self,
        retrieved_docs: List[List[Document]],
        relevant_docs: List[List[Document]],
        k: int = 10
    ) -> float:
        """
        計算NDCG (Normalized Discounted Cumulative Gain)
        
        Args:
            retrieved_docs: 檢索到的文檔列表的列表
            relevant_docs: 相關文檔列表的列表
            k: 計算NDCG的文檔數量
            
        Returns:
            NDCG
        """
        ndcg_scores = []
        
        for i in range(len(retrieved_docs)):
            retrieved = retrieved_docs[i][:k]  # 只考慮前k個文檔
            relevant = relevant_docs[i]
            
            # 獲取檢索文檔和相關文檔的ID
            retrieved_ids = [self._get_doc_id(doc) for doc in retrieved]
            relevant_ids = [self._get_doc_id(doc) for doc in relevant]
            
            # 計算DCG
            dcg = 0.0
            for j, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_ids:
                    # 使用二元相關性（0或1）
                    relevance = 1
                    # 計算DCG
                    dcg += relevance / np.log2(j + 2)  # j+2是因為log2(1)=0，我們從位置1開始
            
            # 計算IDCG（理想DCG）
            idcg = 0.0
            for j in range(min(len(relevant_ids), k)):
                # 使用二元相關性（0或1）
                relevance = 1
                # 計算IDCG
                idcg += relevance / np.log2(j + 2)
            
            # 計算NDCG
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(1.0)  # 如果沒有相關文檔，則NDCG為1
        
        # 計算平均NDCG
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    
    def _calculate_mrr(
        self,
        retrieved_docs: List[List[Document]],
        relevant_docs: List[List[Document]]
    ) -> float:
        """
        計算MRR (Mean Reciprocal Rank)
        
        Args:
            retrieved_docs: 檢索到的文檔列表的列表
            relevant_docs: 相關文檔列表的列表
            
        Returns:
            MRR
        """
        reciprocal_ranks = []
        
        for i in range(len(retrieved_docs)):
            retrieved = retrieved_docs[i]
            relevant = relevant_docs[i]
            
            # 獲取檢索文檔和相關文檔的ID
            retrieved_ids = [self._get_doc_id(doc) for doc in retrieved]
            relevant_ids = [self._get_doc_id(doc) for doc in relevant]
            
            # 計算倒數排名
            rank = 0
            for j, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_ids:
                    rank = j + 1  # 排名從1開始
                    break
            
            if rank > 0:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        # 計算平均倒數排名
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def _get_doc_id(self, doc: Document) -> str:
        """
        獲取文檔ID
        
        Args:
            doc: 文檔
            
        Returns:
            文檔ID
        """
        # 如果文檔有ID，則使用ID
        if hasattr(doc, 'id') and doc.id:
            return doc.id
        
        # 如果文檔的metadata中有ID，則使用metadata中的ID
        if hasattr(doc, 'metadata') and doc.metadata and 'id' in doc.metadata:
            return doc.metadata['id']
        
        # 如果文檔的metadata中有source，則使用source作為ID
        if hasattr(doc, 'metadata') and doc.metadata and 'source' in doc.metadata:
            return doc.metadata['source']
        
        # 如果以上都沒有，則使用文檔內容的哈希值作為ID
        return str(hash(doc.page_content))
    
    def _save_retrieval_results(
        self,
        queries: List[str],
        retrieved_docs: List[List[Document]],
        relevant_docs: List[List[Document]],
        results: Dict[str, float]
    ) -> None:
        """
        保存檢索評估結果
        
        Args:
            queries: 查詢列表
            retrieved_docs: 檢索到的文檔列表的列表
            relevant_docs: 相關文檔列表的列表
            results: 評估結果字典
        """
        # 創建結果目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.output_dir, f"retrieval_evaluation_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存評估結果
        with open(os.path.join(result_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存詳細結果
        detailed_results = []
        
        for i in range(len(queries)):
            query = queries[i]
            retrieved = retrieved_docs[i]
            relevant = relevant_docs[i]
            
            # 獲取檢索文檔和相關文檔的ID
            retrieved_ids = [self._get_doc_id(doc) for doc in retrieved]
            relevant_ids = [self._get_doc_id(doc) for doc in relevant]
            
            # 計算精確率和召回率
            if len(retrieved_ids) == 0:
                precision = 0.0
            else:
                relevant_retrieved = [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]
                precision = len(relevant_retrieved) / len(retrieved_ids)
            
            if len(relevant_ids) == 0:
                recall = 1.0
            else:
                relevant_retrieved = [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]
                recall = len(relevant_retrieved) / len(relevant_ids)
            
            # 添加到詳細結果
            detailed_results.append({
                "query": query,
                "retrieved_docs": [doc.page_content for doc in retrieved],
                "relevant_docs": [doc.page_content for doc in relevant],
                "precision": precision,
                "recall": recall
            })
        
        # 保存詳細結果
        with open(os.path.join(result_dir, "detailed_results.json"), "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"檢索評估結果已保存到 {result_dir}")

class GenerationEvaluator:
    """生成評估器，負責評估生成系統的性能"""
    
    def __init__(self, config: EvaluationConfig):
        """
        初始化生成評估器
        
        Args:
            config: 評估配置
        """
        self.config = config
        self.metrics = config.metrics
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化嵌入模型
        self.embedding_model = self._init_embedding_model()
        
        # 初始化ROUGE評分器
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # 初始化BLEU平滑函數
        self.smoothing = SmoothingFunction().method1
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            return HuggingFaceEmbeddings(
                model_name=self.config.embeddings_model_name,
                model_kwargs={'device': 'cpu'}  # 使用CPU以節省資源
            )
        except Exception as e:
            logger.error(f"初始化嵌入模型時出錯: {str(e)}")
            return None
    
    def evaluate_generation(
        self,
        queries: List[str],
        generated_answers: List[str],
        reference_answers: List[str],
        contexts: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """
        評估生成性能
        
        Args:
            queries: 查詢列表
            generated_answers: 生成的答案列表
            reference_answers: 參考答案列表
            contexts: 上下文列表的列表
            
        Returns:
            評估結果字典
        """
        results = {}
        
        # 計算各項指標
        if "relevance" in self.metrics:
            results["relevance"] = self._calculate_relevance(queries, generated_answers)
        
        if "factuality" in self.metrics:
            results["factuality"] = self._calculate_factuality(generated_answers, reference_answers, contexts)
        
        if "coherence" in self.metrics:
            results["coherence"] = self._calculate_coherence(generated_answers)
        
        if "fluency" in self.metrics:
            results["fluency"] = self._calculate_fluency(generated_answers)
        
        if "rouge" in self.metrics:
            rouge_scores = self._calculate_rouge(generated_answers, reference_answers)
            results.update(rouge_scores)
        
        if "bleu" in self.metrics:
            results["bleu"] = self._calculate_bleu(generated_answers, reference_answers)
        
        # 保存結果
        if self.config.save_results:
            self._save_generation_results(queries, generated_answers, reference_answers, contexts, results)
        
        return results
    
    def _calculate_relevance(
        self,
        queries: List[str],
        generated_answers: List[str]
    ) -> float:
        """
        計算相關性
        
        Args:
            queries: 查詢列表
            generated_answers: 生成的答案列表
            
        Returns:
            相關性分數
        """
        if self.embedding_model is None:
            logger.warning("嵌入模型未初始化，無法計算相關性")
            return 0.0
        
        relevance_scores = []
        
        for i in range(len(queries)):
            query = queries[i]
            answer = generated_answers[i]
            
            # 獲取查詢和答案的嵌入
            query_embedding = self.embedding_model.embed_query(query)
            answer_embedding = self.embedding_model.embed_query(answer)
            
            # 計算餘弦相似度
            similarity = cosine_similarity([query_embedding], [answer_embedding])[0][0]
            relevance_scores.append(similarity)
        
        # 計算平均相關性
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    
    def _calculate_factuality(
        self,
        generated_answers: List[str],
        reference_answers: List[str],
        contexts: Optional[List[List[str]]] = None
    ) -> float:
        """
        計算事實性
        
        Args:
            generated_answers: 生成的答案列表
            reference_answers: 參考答案列表
            contexts: 上下文列表的列表
            
        Returns:
            事實性分數
        """
        if self.embedding_model is None:
            logger.warning("嵌入模型未初始化，無法計算事實性")
            return 0.0
        
        factuality_scores = []
        
        for i in range(len(generated_answers)):
            generated = generated_answers[i]
            reference = 
(Content truncated due to size limit. Use line ranges to read in chunks)