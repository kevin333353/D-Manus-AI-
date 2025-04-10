"""
模型量化模組 - 負責大型語言模型的量化和優化
"""

import os
import torch
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    """量化配置類"""
    model_name_or_path: str = "meta-llama/Llama-3-8b-hf"  # 模型名稱或路徑
    output_dir: str = "../models/quantized"               # 輸出目錄
    bits: int = 4                                         # 量化位元數
    group_size: int = 128                                 # 量化分組大小
    damp_percent: float = 0.01                            # 阻尼係數百分比
    desc_act: bool = True                                 # 是否使用描述符激活
    sym: bool = True                                      # 是否使用對稱量化
    true_sequential: bool = True                          # 是否使用真正的順序處理
    calibration_samples: List[str] = field(default_factory=list)  # 校準樣本
    calibration_dataset_path: Optional[str] = None        # 校準數據集路徑
    calibration_seq_length: int = 2048                    # 校準序列長度
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 設備

class GPTQQuantizer:
    """GPTQ量化器，負責使用GPTQ算法量化模型"""
    
    def __init__(self, config: QuantizationConfig):
        """
        初始化GPTQ量化器
        
        Args:
            config: 量化配置
        """
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 檢查是否安裝了必要的庫
        try:
            import optimum
            from optimum.gptq import GPTQQuantizer as OptimumGPTQQuantizer
            self.optimum_available = True
        except ImportError:
            logger.warning("未安裝optimum-gptq，將使用模擬量化。要使用實際GPTQ量化，請安裝：pip install optimum[gptq]")
            self.optimum_available = False
    
    def prepare_calibration_data(self) -> List[str]:
        """
        準備校準數據
        
        Returns:
            校準樣本列表
        """
        # 如果已提供校準樣本，直接使用
        if self.config.calibration_samples:
            return self.config.calibration_samples
        
        # 如果提供了校準數據集路徑，從中加載
        if self.config.calibration_dataset_path and os.path.exists(self.config.calibration_dataset_path):
            try:
                from datasets import load_dataset
                
                # 加載數據集
                dataset = load_dataset('json', data_files=self.config.calibration_dataset_path)
                
                # 提取文本樣本
                if 'train' in dataset and 'text' in dataset['train'].features:
                    samples = [sample['text'] for sample in dataset['train'] if 'text' in sample]
                    if samples:
                        logger.info(f"從數據集加載了 {len(samples)} 個校準樣本")
                        return samples[:100]  # 限制樣本數量
            except Exception as e:
                logger.error(f"加載校準數據集時出錯: {str(e)}")
        
        # 如果沒有提供校準數據，使用默認樣本
        logger.info("使用默認校準樣本")
        return [
            "人工智能（Artificial Intelligence，簡稱AI）是計算機科學的一個分支，致力於開發能夠模擬人類智能的系統。",
            "大型語言模型（Large Language Models，簡稱LLM）是一種基於深度學習的自然語言處理模型，通過大規模預訓練學習語言的統計規律。",
            "檢索增強生成（Retrieval-Augmented Generation，RAG）是一種結合檢索系統和生成模型的技術。它首先從知識庫中檢索與查詢相關的文檔，然後將這些文檔作為上下文提供給生成模型。",
            "模型量化是一種通過降低模型參數精度來減少模型大小和計算需求的技術，同時盡量保持模型性能。常見的量化方法包括GPTQ、AWQ、SmoothQuant等。",
            "向量資料庫在RAG系統中扮演著核心角色，主要負責高效存儲和檢索文檔的向量表示。常見的向量資料庫包括Faiss、Milvus、Weaviate、Pinecone、Chroma等。"
        ]
    
    def quantize_model_with_optimum(self) -> Tuple[str, str]:
        """
        使用Optimum庫進行GPTQ量化
        
        Returns:
            量化模型路徑和量化腳本路徑的元組
        """
        if not self.optimum_available:
            raise ImportError("未安裝optimum-gptq，無法執行實際量化")
        
        # 創建量化腳本
        script_content = f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model

# 加載模型和tokenizer
model_name = "{self.config.model_name_or_path}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 準備校準數據
calibration_samples = {self.prepare_calibration_data()}

# 初始化量化器
quantizer = GPTQQuantizer(
    bits={self.config.bits},
    group_size={self.config.group_size},
    desc_act={str(self.config.desc_act).lower()},
    sym={str(self.config.sym).lower()}
)

# 量化模型
quantized_model = quantizer.quantize_model(
    model=model,
    tokenizer=tokenizer,
    calibration_samples=calibration_samples,
    calibration_max_seq_length={self.config.calibration_seq_length}
)

# 保存量化模型
output_dir = "{self.config.output_dir}"
quantized_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"量化模型已保存到 {{output_dir}}")
"""
        
        # 保存量化腳本
        script_path = os.path.join(os.path.dirname(self.config.output_dir), "quantize_gptq.py")
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        
        logger.info(f"量化腳本已保存到 {script_path}")
        
        # 返回量化模型路徑和腳本路徑
        return self.config.output_dir, script_path
    
    def generate_quantization_instructions(self) -> str:
        """
        生成量化指令
        
        Returns:
            量化指令文本
        """
        # 獲取量化模型路徑和腳本路徑
        try:
            model_path, script_path = self.quantize_model_with_optimum()
        except ImportError:
            model_path = self.config.output_dir
            script_path = os.path.join(os.path.dirname(self.config.output_dir), "quantize_gptq.py")
            
            # 創建模擬量化腳本
            script_content = f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加載模型和tokenizer
model_name = "{self.config.model_name_or_path}"
print(f"加載模型: {{model_name}}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 模擬量化過程
print("執行{self.config.bits}位GPTQ量化（模擬）...")
print(f"使用分組大小: {self.config.group_size}")
print(f"使用對稱量化: {self.config.sym}")

# 保存模型（實際上是原始模型）
output_dir = "{self.config.output_dir}"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"模型已保存到 {{output_dir}}")
print("注意：這是模擬量化，實際上保存的是原始模型。要執行實際量化，請安裝optimum[gptq]")
"""
            
            # 保存模擬量化腳本
            os.makedirs(os.path.dirname(script_path), exist_ok=True)
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)
            
            logger.info(f"模擬量化腳本已保存到 {script_path}")
        
        # 生成量化指令
        instructions = f"""
# 模型量化指令

## 環境準備

首先，確保已安裝所需的依賴庫：

```bash
pip install transformers torch
pip install optimum[gptq]  # 用於GPTQ量化
```

## 量化腳本

量化腳本已準備好：
- 腳本路徑: {script_path}

## 執行量化

使用以下命令執行量化：

```bash
python {script_path}
```

## 量化配置

- 基礎模型: {self.config.model_name_or_path}
- 輸出目錄: {self.config.output_dir}
- 量化位元數: {self.config.bits}
- 分組大小: {self.config.group_size}
- 對稱量化: {self.config.sym}
- 校準序列長度: {self.config.calibration_seq_length}

## 注意事項

- 量化過程可能需要較長時間，取決於模型大小和硬件配置
- 量化需要足夠的GPU內存，對於大型模型可能需要高端GPU
- 量化後的模型大小將顯著減小，但可能會有輕微的性能下降

## 使用量化後的模型

量化完成後，可以使用以下代碼加載量化模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加載量化模型
model = AutoModelForCausalLM.from_pretrained(
    "{self.config.output_dir}",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{self.config.output_dir}")

# 使用模型
inputs = tokenizer("請輸入您的問題：", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""
        
        # 保存指令到文件
        instructions_path = os.path.join(os.path.dirname(self.config.output_dir), "quantization_instructions.md")
        with open(instructions_path, "w", encoding="utf-8") as f:
            f.write(instructions)
        
        logger.info(f"量化指令已保存到 {instructions_path}")
        return instructions_path

class AWQQuantizer:
    """AWQ量化器，負責使用AWQ算法量化模型"""
    
    def __init__(self, config: QuantizationConfig):
        """
        初始化AWQ量化器
        
        Args:
            config: 量化配置
        """
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 檢查是否安裝了必要的庫
        try:
            import awq
            self.awq_available = True
        except ImportError:
            logger.warning("未安裝AWQ，將使用模擬量化。要使用實際AWQ量化，請安裝：pip install autoawq")
            self.awq_available = False
    
    def generate_quantization_script(self) -> str:
        """
        生成AWQ量化腳本
        
        Returns:
            量化腳本路徑
        """
        # 創建量化腳本
        script_content = f"""
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 加載模型和tokenizer
model_name = "{self.config.model_name_or_path}"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加載模型用於量化
model = AutoAWQForCausalLM.from_pretrained(model_name)

# 準備校準數據
calibration_samples = {self.prepare_calibration_data()}

# 量化模型
model.quantize(
    tokenizer=tokenizer,
    quant_config={{
        "bits": {self.config.bits},
        "group_size": {self.config.group_size},
        "sym": {str(self.config.sym).lower()},
        "desc_act": {str(self.config.desc_act).lower()}
    }},
    calib_data=calibration_samples
)

# 保存量化模型
output_dir = "{self.config.output_dir}"
model.save_quantized(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"AWQ量化模型已保存到 {{output_dir}}")
"""
        
        # 保存量化腳本
        script_path = os.path.join(os.path.dirname(self.config.output_dir), "quantize_awq.py")
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        
        logger.info(f"AWQ量化腳本已保存到 {script_path}")
        return script_path
    
    def prepare_calibration_data(self) -> List[str]:
        """
        準備校準數據
        
        Returns:
            校準樣本列表
        """
        # 與GPTQ量化器使用相同的校準數據準備邏輯
        # 如果已提供校準樣本，直接使用
        if self.config.calibration_samples:
            return self.config.calibration_samples
        
        # 如果提供了校準數據集路徑，從中加載
        if self.config.calibration_dataset_path and os.path.exists(self.config.calibration_dataset_path):
            try:
                from datasets import load_dataset
                
                # 加載數據集
                dataset = load_dataset('json', data_files=self.config.calibration_dataset_path)
                
                # 提取文本樣本
                if 'train' in dataset and 'text' in dataset['train'].features:
                    samples = [sample['text'] for sample in dataset['train'] if 'text' in sample]
                    if samples:
                        logger.info(f"從數據集加載了 {len(samples)} 個校準樣本")
                        return samples[:100]  # 限制樣本數量
            except Exception as e:
                logger.error(f"加載校準數據集時出錯: {str(e)}")
        
        # 如果沒有提供校準數據，使用默認樣本
        logger.info("使用默認校準樣本")
        return [
            "人工智能（Artificial Intelligence，簡稱AI）是計算機科學的一個分支，致力於開發能夠模擬人類智能的系統。",
            "大型語言模型（Large Language Models，簡稱LLM）是一種基於深度學習的自然語言處理模型，通過大規模預訓練學習語言的統計規律。",
            "檢索增強生成（Retrieval-Augmented Generation，RAG）是一種結合檢索系統和生成模型的技術。它首先從知識庫中檢索與查詢相關的文檔，然後將這些文檔作為上下文提供給生成模型。",
            "模型量化是一種通過降低模型參數精度來減少模型大小和計算需求的技術，同時盡量保持模型性能。常見的量化方法包括GPTQ、AWQ、SmoothQuant等。",
            "向量資料庫在RAG系統中扮演著核心角色，主要負責高效存儲和檢索文檔的向量表示。常見的向量資料庫包括Faiss、Milvus、Weaviate、Pinecone、Chroma等。"
        ]
    
    def generate_quantization_instructions(self) -> str:
        """
        生成量化指令
        
        Returns:
            量化指令文本
        """
        # 生成量化腳本
        script_path = self.generate_quantization_script()
        
        # 生成量化指令
        instructions = f"""
# AWQ模型量化指令

## 環境準備

首先，確保已安裝所需的依賴庫：

```bash
pip install transformers torch
pip install autoawq  # 用於AWQ量化
```

## 量化腳本

AWQ量化腳本已準備好：
- 腳本路徑: {script_path}

## 執行量化

使用以下命令執行量化：

```bash
python {script_path}
```

## 量化配置

- 基礎模型: {self.config.model_name_or_path}
- 輸出目錄: {self.config.output_dir}
- 量化位元數: {self.config.bits}
- 分組大小: {self.config.group_size}
- 對稱量化: {self.config.sym}

## 注意事項

- AWQ量化通常比GPTQ提供更好的性能-大小權衡
- 量化過程可能需要較長時間，取決於模型大小和硬件配置
- 量化需要足夠的GPU內存，對於大型模型可能需要高端GPU

## 使用量化後的模型

量化完成後，可以使用以下代碼加載AWQ量化模型：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 加載量化模型
model = AutoAWQForCausalLM.from_pretrained("{self.config.output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{self.config.output_dir}")

# 使用模型
inputs = tokenizer("請輸入您的問題：", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 與vLLM集成

AWQ量化模型可以與vLLM一起使用，以獲得更快的推理速度：

```python
from vllm import LLM, SamplingParams

# 加載AWQ量化模型
llm = LLM(
    model="{self.config.output_dir}",
    quantization="awq",
    dtype="auto"
)

# 設置採樣參數
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)

# 生成回應
outputs = llm.generate("請輸入您的問題：", sampling_params)
print(outputs[0].outputs[0].text)
```
"""
        
        # 保存指令到文件
        instructions_path = os.path.join(os.path.dirname(self.config.output_dir), "awq_quantization_instructions.md")
        with open(instructions_path, "w", encoding="utf-8") as f:
            f.write(instructions)
        
        logger.info(f"AWQ量化指令已保存到 {instructions_path}")
        return instructions_path

class QuantizedModelManager:
    """量化模型管理器，負責加載和使用量化模型"""
    
    def __init__(
        self,
        model_path: str,
        quantization_type: str = "gptq",  # 'gptq', 'awq', 'none'
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化量化模型管理器
        
        Args:
            model_path: 模型路徑
            quantization_type: 量化類型
            device: 設備
        """
        self.model_path = model_path
        self.quantization_type = quantization_type
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def load_model(self) -> Tuple[Any, Any]:
        """
        加載模型
        
        Returns:
            模型和tokenizer的元組
        """
        logger.info(f"加載{self.quantization_type}量化模型: {self.model_path}")
        
        if self.quantization_type == "gptq":
            # 加載GPTQ量化模型
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto"
                )
                logger.info("成功加載GPTQ量化模型")
            except Exception as e:
                logger.error(f"加載GPTQ量化模型時出錯: {str(e)}")
                raise
        
        elif self.quantization_type == "awq":
            # 加載AWQ量化模型
            try:
                from awq import AutoAWQForCausalLM
                from transformers import AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoAWQForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto"
                )
                logger.info("成功加載AWQ量化模型")
            except ImportError:
                logger.error("未安裝AWQ，無法加載AWQ量化模型")
                raise
            except Exception as e:
                logger.error(f"加載AWQ量化模型時出錯: {str(e)}")
                raise
        
        else:
            # 加載普通模型
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # 使用4位量化加載模型
            bnb_config = BitsAndBytesConfig(
          
(Content truncated due to size limit. Use line ranges to read in chunks)