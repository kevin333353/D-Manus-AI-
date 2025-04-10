# Transformer 與大型語言模型 (LLM) 基礎理論

## Transformer 架構基礎

Transformer 架構自 2017 年由 Google 團隊在論文《Attention Is All You Need》中提出後，徹底改變了自然語言處理領域。這個架構捨棄了傳統的循環神經網路 (RNN) 和卷積神經網路 (CNN)，完全基於注意力機制 (Attention Mechanism) 構建，使模型能夠平行處理輸入序列，大幅提高訓練效率並捕捉長距離依賴關係。

### 核心組件

Transformer 架構主要由以下核心組件構成：

1. **自注意力機制 (Self-Attention)**：允許模型在處理一個詞時能夠「關注」輸入序列中的其他所有詞，並根據相關性賦予不同權重。這使模型能夠理解詞與詞之間的複雜關係，無論它們在句子中的距離有多遠。自注意力機制通過計算查詢 (Query)、鍵 (Key) 和值 (Value) 之間的相互關係來實現，這三者都是輸入向量的線性變換。

2. **多頭注意力 (Multi-Head Attention)**：將自注意力機制擴展為多個「頭」，每個頭可以關注輸入的不同方面，然後將這些結果合併，豐富了模型的表達能力。

3. **位置編碼 (Positional Encoding)**：由於 Transformer 不像 RNN 那樣天然具有處理序列順序的能力，位置編碼被添加到輸入嵌入中，為模型提供詞在序列中位置的信息。

4. **前饋神經網路 (Feed-Forward Network)**：在每個 Transformer 層中，注意力子層之後是一個包含兩個線性變換和一個非線性激活函數的前饋網路，用於進一步處理注意力機制的輸出。

5. **層正規化 (Layer Normalization) 與殘差連接 (Residual Connection)**：這些技術用於穩定訓練過程並幫助梯度流動，使得非常深的網絡能夠有效訓練。

### 編碼器-解碼器架構

原始 Transformer 採用編碼器-解碼器 (Encoder-Decoder) 架構，適用於機器翻譯等序列到序列任務：

- **編碼器 (Encoder)**：處理輸入序列，將其轉換為連續表示 (表示為注意力向量)。編碼器由多個相同層堆疊而成，每層包含自注意力機制和前饋神經網路。

- **解碼器 (Decoder)**：基於編碼器的輸出和之前生成的輸出，逐步生成目標序列。解碼器也由多個相同層堆疊而成，但除了自注意力和前饋網路外，還包含一個對編碼器輸出進行注意力計算的層。

## 大型語言模型 (LLM) 演進

大型語言模型是基於 Transformer 架構發展而來的，通過增加模型參數量和訓練數據規模，實現了前所未有的語言理解和生成能力。

### 從 BERT 到 GPT 系列

1. **BERT (Bidirectional Encoder Representations from Transformers)**：由 Google 於 2018 年提出，使用 Transformer 的編碼器部分，通過雙向上下文預訓練，在多種 NLP 任務上取得了突破性進展。BERT 的創新之處在於其預訓練目標：掩碼語言模型 (MLM) 和下一句預測 (NSP)，使模型能夠學習深層的雙向表示。

2. **GPT (Generative Pre-trained Transformer)**：由 OpenAI 開發的系列模型，使用 Transformer 的解碼器部分，採用自迴歸方式預訓練。GPT 系列的演進展示了擴展模型規模的威力：
   - GPT-1 (2018)：1.17 億參數
   - GPT-2 (2019)：15 億參數
   - GPT-3 (2020)：1750 億參數
   - GPT-4 (2023)：參數量未公開，但估計超過 1 萬億

3. **LLaMA 系列**：由 Meta AI 開發的開源大型語言模型系列，提供了從 70 億到 700 億參數的多種規模版本。LLaMA 2 在訓練數據和方法上進行了改進，特別是在指令微調和對齊方面。LLaMA 3 進一步提升了性能，並提供了更多開源選項。

### 預訓練與微調範式

現代 LLM 開發通常遵循「預訓練-微調」範式：

1. **預訓練 (Pre-training)**：在大規模文本語料庫上訓練模型，使其學習語言的統計規律和世界知識。預訓練通常採用自監督學習方法，如下一個詞預測任務。

2. **微調 (Fine-tuning)**：在特定任務的標註數據上進一步訓練預訓練模型，使其適應特定領域或任務需求。微調可以是傳統的監督式微調，也可以是更現代的方法如指令微調 (Instruction Tuning) 或 RLHF (Reinforcement Learning from Human Feedback)。

### 指令微調與對齊技術

為了使 LLM 更好地遵循人類指令並產生有用、無害的回應，研究者開發了多種技術：

1. **指令微調 (Instruction Tuning)**：通過在指令-回應對上微調模型，使其能夠理解並執行各種自然語言指令。這種方法顯著提高了模型的可用性和通用性。

2. **RLHF (Reinforcement Learning from Human Feedback)**：使用人類反饋來訓練獎勵模型，然後使用強化學習優化語言模型。這種方法在 ChatGPT 和 Claude 等模型中得到了廣泛應用。

3. **憲法 AI (Constitutional AI)**：通過定義一系列原則或「憲法」來引導模型生成，減少對大量人類反饋的依賴。

4. **PEFT (Parameter-Efficient Fine-Tuning)**：如 LoRA (Low-Rank Adaptation)、Prefix Tuning 和 P-Tuning 等技術，允許在有限計算資源下高效微調大型模型。

## 實作 Transformer 和 LLM 的框架

### PyTorch 實現

PyTorch 是實現 Transformer 和 LLM 最流行的框架之一，提供了靈活的動態計算圖和豐富的工具：

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out
```

### Hugging Face Transformers

Hugging Face Transformers 庫提供了預訓練模型和工具，大大簡化了 Transformer 模型的使用：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 載入預訓練模型和分詞器
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 生成文本
input_text = "請解釋什麼是 Transformer 架構："
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## LLM 的評估與挑戰

### 評估指標

評估 LLM 性能的常用指標包括：

1. **困惑度 (Perplexity)**：測量模型對測試集的預測能力，較低的困惑度表示更好的語言模型。

2. **BLEU, ROUGE, METEOR**：用於評估生成文本與參考文本的相似度，常用於翻譯和摘要任務。

3. **人類評估**：通過人類評判模型輸出的質量、相關性、有用性和安全性。

4. **基準測試**：如 MMLU (Massive Multitask Language Understanding)、HumanEval (程式碼生成)、GSM8K (數學推理) 等，測試模型在不同任務上的能力。

### 主要挑戰

LLM 面臨的主要挑戰包括：

1. **幻覺 (Hallucination)**：模型生成看似合理但實際上不正確或虛構的信息。

2. **偏見與有害內容**：模型可能反映或放大訓練數據中的偏見，或生成有害內容。

3. **可解釋性**：理解模型為何做出特定預測或生成特定內容仍然具有挑戰性。

4. **計算資源需求**：訓練和部署大型模型需要大量計算資源。

5. **上下文長度限制**：雖然新模型不斷擴展上下文窗口，但處理非常長的文檔仍然具有挑戰性。

## 結論

Transformer 架構和大型語言模型代表了 NLP 領域的重大突破，為各種應用開啟了新的可能性。隨著研究的進展，我們可以期待更高效、更可靠、更安全的模型出現，進一步擴展 AI 在語言理解和生成方面的能力。

在實際應用中，理解這些模型的基本原理、優勢和局限性至關重要，這有助於開發者設計出更有效的解決方案，並應對可能出現的挑戰。
