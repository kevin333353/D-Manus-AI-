"""
LLM微調模組 - 負責大型語言模型的參數高效微調
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datasets import Dataset, load_dataset

@dataclass
class FineTuningConfig:
    """微調配置類"""
    model_name: str = "meta-llama/Llama-3-8b-hf"  # 基礎模型名稱
    output_dir: str = "../models/fine_tuned"      # 輸出目錄
    lora_r: int = 16                              # LoRA秩
    lora_alpha: int = 32                          # LoRA alpha參數
    lora_dropout: float = 0.05                    # LoRA dropout
    learning_rate: float = 2e-4                   # 學習率
    batch_size: int = 4                           # 批次大小
    gradient_accumulation_steps: int = 4          # 梯度累積步數
    num_train_epochs: int = 3                     # 訓練輪數
    max_seq_length: int = 2048                    # 最大序列長度
    save_strategy: str = "epoch"                  # 保存策略
    fp16: bool = True                             # 是否使用半精度浮點數
    target_modules: List[str] = None              # 目標模組
    
    def __post_init__(self):
        """初始化後處理"""
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

class DataProcessor:
    """數據處理類，負責準備微調數據"""
    
    def __init__(self, data_dir: str = "../data/training_data"):
        """
        初始化數據處理器
        
        Args:
            data_dir: 數據目錄
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def create_instruction_dataset(
        self, 
        instructions: List[Dict[str, str]],
        output_file: str = "instruction_dataset.json"
    ) -> str:
        """
        創建指令數據集
        
        Args:
            instructions: 指令列表，每個指令是一個字典，包含'instruction'和'response'
            output_file: 輸出文件名
            
        Returns:
            輸出文件路徑
        """
        output_path = os.path.join(self.data_dir, output_file)
        
        # 確保數據格式正確
        formatted_instructions = []
        for item in instructions:
            if 'instruction' in item and 'response' in item:
                formatted_instructions.append({
                    'instruction': item['instruction'],
                    'response': item['response']
                })
        
        # 寫入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_instructions, f, ensure_ascii=False, indent=2)
        
        print(f"已創建指令數據集: {output_path}")
        return output_path
    
    def create_conversation_dataset(
        self, 
        conversations: List[Dict[str, Any]],
        output_file: str = "conversation_dataset.json"
    ) -> str:
        """
        創建對話數據集
        
        Args:
            conversations: 對話列表，每個對話是一個字典，包含'messages'列表
            output_file: 輸出文件名
            
        Returns:
            輸出文件路徑
        """
        output_path = os.path.join(self.data_dir, output_file)
        
        # 確保數據格式正確
        formatted_conversations = []
        for conv in conversations:
            if 'messages' in conv and isinstance(conv['messages'], list):
                formatted_conversations.append({
                    'messages': conv['messages']
                })
        
        # 寫入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_conversations, f, ensure_ascii=False, indent=2)
        
        print(f"已創建對話數據集: {output_path}")
        return output_path
    
    def load_dataset_from_json(self, file_path: str) -> Dataset:
        """
        從JSON文件加載數據集
        
        Args:
            file_path: JSON文件路徑
            
        Returns:
            Dataset對象
        """
        # 檢查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 加載數據集
        dataset = load_dataset('json', data_files=file_path)
        print(f"已加載數據集: {file_path}")
        print(f"數據集大小: {len(dataset['train'])}")
        
        return dataset
    
    def format_instruction_dataset(self, dataset: Dataset) -> Dataset:
        """
        格式化指令數據集為模型輸入格式
        
        Args:
            dataset: 原始數據集
            
        Returns:
            格式化後的數據集
        """
        def format_instruction(example):
            """格式化單個指令樣本"""
            return {
                "text": f"### 指令:\n{example['instruction']}\n\n### 回應:\n{example['response']}"
            }
        
        # 應用格式化函數
        formatted_dataset = dataset.map(format_instruction)
        return formatted_dataset
    
    def format_conversation_dataset(self, dataset: Dataset) -> Dataset:
        """
        格式化對話數據集為模型輸入格式
        
        Args:
            dataset: 原始數據集
            
        Returns:
            格式化後的數據集
        """
        def format_conversation(example):
            """格式化單個對話樣本"""
            conversation = ""
            for message in example['messages']:
                role = message.get('role', 'unknown')
                content = message.get('content', '')
                conversation += f"### {role}:\n{content}\n\n"
            
            return {"text": conversation.strip()}
        
        # 應用格式化函數
        formatted_dataset = dataset.map(format_conversation)
        return formatted_dataset
    
    def create_sample_instruction_dataset(self) -> str:
        """
        創建示例指令數據集
        
        Returns:
            數據集文件路徑
        """
        # 示例指令數據
        instructions = [
            {
                "instruction": "解釋什麼是檢索增強生成（RAG）技術？",
                "response": "檢索增強生成（Retrieval-Augmented Generation，RAG）是一種結合檢索系統和生成模型的技術。它首先從知識庫中檢索與查詢相關的文檔，然後將這些文檔作為上下文提供給生成模型，使模型能夠基於檢索到的信息生成更準確、更可靠的回應。RAG技術可以有效解決大型語言模型的幻覺問題，並使模型能夠訪問最新信息，而無需持續重新訓練。"
            },
            {
                "instruction": "什麼是參數高效微調（PEFT）？請詳細解釋。",
                "response": "參數高效微調（Parameter-Efficient Fine-Tuning，PEFT）是一種優化大型語言模型微調過程的技術，旨在減少計算資源需求並提高微調效率。傳統的微調方法需要更新模型的所有參數，這對於現代大型模型（可能有數十億或數千億參數）來說計算成本極高。PEFT方法只更新模型參數的一小部分，同時保持性能接近全參數微調。\n\n常見的PEFT技術包括：\n\n1. LoRA（Low-Rank Adaptation）：通過添加低秩矩陣來適應預訓練權重，只訓練這些低秩矩陣而凍結原始權重。\n\n2. Prefix Tuning：在模型輸入添加可訓練的前綴向量，只訓練這些前綴。\n\n3. Prompt Tuning：類似於Prefix Tuning，但僅在輸入層添加可訓練的提示嵌入。\n\n4. Adapter：在Transformer層之間插入小型可訓練模塊，凍結原始模型參數。\n\n這些方法大大減少了微調所需的計算資源和存儲空間，使得在消費級硬件上微調大型模型成為可能。"
            },
            {
                "instruction": "如何評估RAG系統的性能？",
                "response": "評估RAG系統性能需要考慮多個維度：\n\n1. 回應質量評估：\n   - 準確性：回應中的事實是否正確\n   - 相關性：回應是否與查詢相關\n   - 完整性：回應是否完整回答了問題\n   - 一致性：回應是否存在內部矛盾\n\n2. 檢索效果評估：\n   - 召回率：系統檢索到的相關文檔比例\n   - 精確度：檢索結果中相關文檔的比例\n   - MRR（Mean Reciprocal Rank）：第一個相關文檔排名的倒數平均值\n   - NDCG（Normalized Discounted Cumulative Gain）：考慮排序的相關性度量\n\n3. 系統性能評估：\n   - 延遲：系統響應時間\n   - 吞吐量：單位時間內處理的查詢數\n   - 資源使用率：CPU、GPU、內存使用情況\n\n4. 用戶體驗評估：\n   - 滿意度：用戶對回應的滿意程度\n   - 任務完成率：用戶成功完成任務的比例\n   - 交互次數：完成任務所需的交互次數\n\n可以使用RAGAS等專門工具評估RAG系統，它提供了忠實度、上下文相關性、回答相關性等指標。此外，人工評估仍然是評估系統質量的重要手段，特別是對於複雜查詢的處理能力。"
            },
            {
                "instruction": "解釋向量資料庫在RAG系統中的作用。",
                "response": "向量資料庫在RAG（檢索增強生成）系統中扮演著核心角色，主要負責高效存儲和檢索文檔的向量表示。其作用包括：\n\n1. 向量存儲：將文檔轉換為高維向量（嵌入）後存儲，保留文檔的語義信息。\n\n2. 相似性搜索：基於向量相似性（通常使用餘弦相似度或歐氏距離）快速檢索與查詢最相關的文檔。\n\n3. 索引優化：使用各種索引技術（如HNSW、IVF等）加速搜索過程，實現大規模數據集的高效檢索。\n\n4. 元數據過濾：支持基於文檔元數據（如日期、來源、類型等）的過濾，提高檢索精度。\n\n5. 增量更新：支持動態添加、更新和刪除文檔，保持知識庫的最新狀態。\n\n常見的向量資料庫包括Faiss、Milvus、Weaviate、Pinecone、Chroma等，它們各有特點，適用於不同規模和需求的RAG系統。選擇合適的向量資料庫對RAG系統的性能和可擴展性至關重要。"
            },
            {
                "instruction": "如何處理RAG系統中的長文檔？",
                "response": "處理RAG系統中的長文檔需要採取特殊策略，主要包括：\n\n1. 智能分塊：\n   - 基於語義邊界（段落、章節）進行分塊，保持上下文完整性\n   - 使用重疊分塊，確保跨塊信息不丟失\n   - 根據文檔類型和內容複雜度動態調整分塊大小\n\n2. 層次化檢索：\n   - 實現多級檢索架構，先檢索相關文檔，再檢索文檔內的相關段落\n   - 使用文檔摘要作為一級索引，詳細內容作為二級索引\n\n3. 密集通道檢索：\n   - 將長文檔分解為多個通道（如標題、摘要、正文）\n   - 對每個通道單獨編碼和檢索，然後合併結果\n\n4. 遞歸檢索：\n   - 首先檢索相關文檔，然後在檢索到的文檔中進一步檢索\n   - 逐步縮小搜索範圍，提高精度\n\n5. 長文本嵌入技術：\n   - 使用支持長文本的嵌入模型（如BGE-M3）\n   - 採用滑動窗口方法生成多個嵌入，然後聚合\n\n6. 文檔壓縮：\n   - 使用LLM提取文檔的關鍵信息，減少噪音\n   - 在檢索後應用上下文壓縮，只保留最相關部分\n\n這些策略可以單獨使用或組合使用，根據具體應用場景和資源限制選擇最適合的方法。"
            }
        ]
        
        return self.create_instruction_dataset(instructions, "sample_instructions.json")
    
    def create_sample_conversation_dataset(self) -> str:
        """
        創建示例對話數據集
        
        Returns:
            數據集文件路徑
        """
        # 示例對話數據
        conversations = [
            {
                "messages": [
                    {"role": "user", "content": "你能解釋一下什麼是大型語言模型嗎？"},
                    {"role": "assistant", "content": "大型語言模型（Large Language Model，簡稱LLM）是一種基於深度學習的自然語言處理模型，通過在大規模文本數據上訓練，學習語言的統計規律和模式。這些模型通常包含數十億到數千億個參數，能夠理解和生成人類語言，執行各種語言任務，如文本生成、翻譯、摘要、問答等。\n\n現代LLM大多基於Transformer架構，採用自監督學習方法訓練，先進行大規模預訓練，再針對特定任務進行微調。代表性的LLM包括GPT系列、LLaMA系列、Claude等。這些模型展現出了強大的語言理解和生成能力，甚至表現出一定程度的推理能力和創造性。"}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "RAG和傳統問答系統有什麼區別？"},
                    {"role": "assistant", "content": "RAG（檢索增強生成）和傳統問答系統的主要區別在於：\n\n1. 架構方面：\n   - 傳統問答：通常採用管道式架構，包括問題分析、檢索、答案提取等獨立模塊\n   - RAG：結合檢索和生成為一體，檢索結果直接作為生成模型的上下文\n\n2. 回答生成：\n   - 傳統問答：從檢索到的文檔中提取或選擇答案片段\n   - RAG：基於檢索內容生成流暢、連貫的回答，而非簡單提取\n\n3. 知識來源：\n   - 傳統問答：主要依賴結構化知識庫或預定義的問答對\n   - RAG：可利用非結構化文本作為知識來源，更加靈活\n\n4. 處理能力：\n   - 傳統問答：對複雜問題、推理和綜合能力有限\n   - RAG：能處理複雜問題，進行推理和知識綜合\n\n5. 可解釋性：\n   - 傳統問答：通常能明確指出答案來源\n   - RAG：可以提供參考來源，但生成過程較為複雜\n\n6. 更新能力：\n   - 傳統問答：知識更新需要重建索引或更新知識庫\n   - RAG：只需更新檢索庫，無需重新訓練生成模型\n\nRAG結合了傳統問答系統的可控性和現代生成模型的靈活性，是當前智能問答系統的主流方向。"},
                    {"role": "user", "content": "RAG系統中可能出現哪些常見問題？"},
                    {"role": "assistant", "content": "RAG系統中的常見問題包括：\n\n1. 檢索相關問題：\n   - 檢索不足：未能找到相關文檔，導致回答缺乏必要信息\n   - 檢索過度：檢索過多無關文檔，干擾生成模型\n   - 語義不匹配：查詢與文檔表達方式不同，影響檢索效果\n\n2. 生成相關問題：\n   - 幻覺：生成不在檢索結果中的虛假信息\n   - 忽略上下文：未充分利用檢索到的信息\n   - 信息衝突：檢索結果包含矛盾信息時無法正確處理\n\n3. 系統集成問題：\n   - 延遲高：檢索和生成過程耗時，影響用戶體驗\n   - 資源消耗大：需要維護大型向量數據庫和生成模型\n   - 擴展性挑戰：隨知識庫增長，檢索效率可能下降\n\n4. 數據質量問題：\n   - 知識庫質量：低質量或過時的文檔影響回答準確性\n   - 分塊策略：不當的文檔分塊導致上下文丟失\n   - 嵌入質量：嵌入模型不佳導致語義捕捉不準確\n\n5. 評估挑戰：\n   - 多維度評估：難以全面評估系統性能\n   - 主觀性：回答質量評估存在主觀因素\n\n解決這些問題需要優化檢索策略、改進提示工程、增強生成模型對上下文的利用、實施嚴格的知識庫管理以及建立全面的評估框架。"}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "如何使用LoRA技術微調大型語言模型？"},
                    {"role": "assistant", "content": "使用LoRA（Low-Rank Adaptation）技術微調大型語言模型的步驟如下：\n\n1. 環境準備：\n   - 安裝必要的庫：transformers, peft, accelerate, bitsandbytes等\n   - 準備適當的硬件（GPU/TPU）\n\n2. 數據準備：\n   - 收集領域特定數據\n   - 格式化為指令微調格式（instruction-response對）\n   - 分割為訓練集和驗證集\n\n3. 模型加載：\n```python\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n\n# 加載基礎模型\nmodel_id = \"meta-llama/Llama-3-8b-hf\"\ntokenizer = AutoTokenizer.from_pretrained(model_id)\nmodel = AutoModelForCausalLM.from_pretrained(\n    model_id,\n    torch_dtype=torch.bfloat16,\n    device_map=\"auto\"\n)\n```\n\n4. LoRA配置：\n```python\nfrom peft import LoraConfig, get_peft_model\n\n# 配置LoRA參數\nlora_config = LoraConfig(\n    r=16,                  # LoRA秩\n    lora_alpha=32,         # LoRA alpha參數\n    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\"],  # 要微調的模塊\n    lora_dropout=0.05,     # LoRA dropout\n    bias=\"none\",           # 是否包含偏置項\n    task_type=\"CAUSAL_LM\"  # 任務類型\n)\n\n# 應用LoRA配置\nmodel = get_peft_model(model, lora_config)\n```\n\n5. 訓練配置：\n```python\nfrom transformers import TrainingArguments\n\ntraining_args = TrainingArguments(\n    output_dir=\"./lora_llama3_output\",\n    per_device_train_batch_size=4,\n    gradient_accumulation_steps=4,\n    learning_rate=2e-4,\n    num_train_epochs=3,\n    save_strategy=\"epoch\",\n    fp16=True,\n)\n```\n\n6. 訓練執行：\n```python\nfrom trl import SFTTrainer\n\ntrainer = SFTTrainer(\n    model=model,\n    args=training_args,\n    train_dataset=train_dataset,\n    tokenizer=tokenizer,\n    dataset_text_field=\"text\",\n    max_seq_length=2048,\n)\n\ntrainer.train()\n```\n\n7. 模型保存：\n```python\n# 保存LoRA權重\nmodel.save_pretrained(\"./lora_llama3_trained\")\n```\n\n8. 模型合併（可選）：\n```python\nfrom peft import PeftModel\n\n# 加載基礎模型\nbase_model = AutoModelForCausalLM.from_pretrained(model_id)\n\n# 加載LoRA模型\nlora_model = PeftModel.from_pretrained(base_model, \"./lora_llama3_trained\")\n\n# 合併模型\nmerged_model = lora_model.merge_and_unload()\n\n# 保存合併後的模型\nmerged_model.save_pretrained(\"./merged_llama3_model\")\n```\n\nLoRA的主要優勢在於顯著減少了可訓練參數的數量，使得在消費級硬件上微調大型模型成為可能，同時保持接近全參數微調的性能。"}
                ]
            }
        ]
        
        return self.create_conversation_dataset(conversations, "sample_conversations.json")

class LoRAFineTuner:
    """LoRA微調器，負責使用LoRA技術微調大型語言模型"""
    
    def __init__(self, config: FineTuningConfig):
        """
        初始化LoRA微調器
        
        Args:
            config: 微調配置
        """
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
    
    def setup_training_code(self) -> str:
        """
        設置訓練代碼
        
        Returns:
            訓練腳本路徑
        """
        # 創建訓練腳本
        script_content = f"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from trl import SFTTrainer

# 配置參數
model_id = "{self.config.model_name}"
output_dir = "{self.config.output_dir}"
dataset_path = "DATASET_PATH_PLACEHOLDER"  # 將在運行時替換

# 加載數據集
dataset = load_dataset('json', data_files=dataset_path)

# 加載tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 加載模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 配置LoRA
lora_config = LoraConfig(
    r={self.config.lora_r},
    lora_alpha={self.config.lora_alpha},
    target_modules={self.config.target_modules},
    lora_dropout={self.config.lora_dropout},
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 應用LoRA配置
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 配置訓練參數
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size={self.config.batch_size},
    gradient_accumulation_steps={self.config.gradient_accumulation_steps},
    learning_rate={self.config.learning_rate},
    num_train_epochs={self.config.num_train_epochs},
    save_strategy="{self.config.save_strategy}",
    fp16={str(self.config.fp16).lower()},
)

# 初始化SFT Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length={self.config.max_seq_length},
)

# 開始訓練
trainer.train()

# 保存模型
trainer.save_model()
print(f"模型已保存到 {output_dir}")
"""
        
        # 保存訓練腳本
        script_path = os.path.join(os.path.dirname(self.config.output_dir), "train_lora.py")
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        
        print(f"訓練腳本已保存到 {script_path}")
        return script_path
    
    def prepare_training_command(self, dataset_path: str, script_path: str) -> str:
        """
        準備訓練命令
        
        Args:
            dataset_path: 數據集路徑
            script_path: 訓練腳本路徑
            
        Returns:
            訓練命令
        """
        # 創建一個臨時腳本，替換數據集路徑
        temp_script_path = script_path.replace(".py", "_temp.py")
        
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        content = content.replace("DATASET_PATH_PLACEHOLDER", dataset_path)
        
        with open(temp_script_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # 構建訓練命令
        command = f"python {temp_script_path}"
        
        return command, temp_script_path
    
    def generate_training_instructions(self, dataset_path: 
(Content truncated due to size limit. Use line ranges to read in chunks)