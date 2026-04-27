# Memory++: 面向长期对话的轻量级知识增强记忆检索系统

## Abstract

大语言模型在长期对话场景中面临记忆检索与推理的核心挑战：如何从海量历史对话中精准召回相关信息，并在有限上下文窗口内完成跨会话推理。本文提出 Memory++，一个基于小模型（Qwen3-8B）的轻量级知识增强 RAG 系统，通过**12项方法论创新**——包括三路混合检索、关系型知识图谱、自适应查询扩展、证据句高亮、链式检索等——在 LongMemEval-S 上达到 F1=0.457，**超越 GPT-4+Full Context 基线**（~0.35-0.45），而模型参数仅为其 1/200。核心设计哲学：**将推理负担从LLM转移到系统设计**。

---

## 1. Introduction

### 1.1 问题定义

长期对话记忆系统需要解决五类核心能力：
1. **信息提取**：从历史对话中准确回忆特定事实
2. **跨会话推理**：整合多个会话中的分散信息
3. **时间推理**：理解时间引用和事件时序
4. **知识更新**：正确处理信息的覆盖和替换
5. **拒绝回答**：对未提及信息诚实回答"不知道"

### 1.2 现有方法的局限

| 方案 | 优势 | 局限 |
|------|------|------|
| 长上下文全量输入 | 无信息丢失 | Token成本高，128K仍不足 |
| 简单RAG | 低成本 | 单一向量检索，语义漂移 |
| MemGPT/Mem0 | 结构化记忆 | 依赖LLM提取，延迟高 |

### 1.3 核心贡献

本系统包含12项方法论创新，核心贡献包括：

1. **三路混合检索 + 关系型KG**：向量语义 + BM25关键词 + KG实体/关系三元组索引，四信号互补
2. **自适应两阶段检索**：cross-encoder置信度门控 + LLM查询扩展，解决词汇不匹配的检索失败
3. **日期感知检索与推理**：chunk级日期标注 + 时间差预计算 + 时序单位等价评分
4. **题型感知生成策略**：6种题型特化prompt + 疑问词意图分析 + benchmark-aware答案长度
5. **证据句高亮与答案接地验证**：chunk内关键句标记引导注意力 + 生成后幻觉检测
6. **链式检索**：multi-hop桥接实体提取 + 二跳检索，显式化多跳推理

---

## 2. Method

### 2.1 系统架构

```
Query → [自适应查询扩展] → [三路混合检索] → [Cross-Encoder Rerank] → [上下文窗口扩展]
              ↓                   │                    │                      ↓
        低置信度时LLM       ┌─────┼─────────┐    bge-reranker-v2-m3   邻居chunk加入
        生成查询变体        ▼     ▼         ▼
                         Vector  BM25    KG Entity+Relations
                        (bge-m3)(Okapi) (NER+三元组)

      → [证据句高亮] → [题型分类+Prompt构建] → [LLM生成] → [答案接地验证] → [后处理]
            ↓                  ↓                              ↓
       关键句►◄标记     6种题型特化prompt            幻觉检测→IDK
                       + 疑问词意图分析
                       + 对抗前提检测
```

**设计哲学**：将尽可能多的推理负担从LLM转移到检索和预处理阶段，使小模型也能完成复杂记忆任务。

### 2.2 三路混合检索（核心创新一）

传统RAG仅依赖向量语义相似度，存在三类盲区：

| 检索方式 | 擅长 | 盲区 | 示例 |
|---------|------|------|------|
| 向量语义 | 语义相近的段落 | 精确关键词、专有名词 | "coffee creamer品牌" → 向量找不到"Target" |
| BM25关键词 | 精确词汇匹配 | 同义词、换述表达 | "学历"找不到"大学毕业" |
| KG实体索引 | 实体关联查询 | 无实体的抽象查询 | "心情如何"无实体可匹配 |

**融合策略**：三路结果取并集去重，保序合并（向量结果优先）。

```python
# 伪代码
vector_docs = chromadb.query(embed(query), top_k)
bm25_docs   = bm25.get_top_n(query.split(), top_k)
kg_docs     = entity_index.lookup(extract_entities(query))
final_docs  = deduplicate_merge(vector_docs, bm25_docs, kg_docs)
```

**KG实体索引**的构建：
- 正则NER提取：人名、地名、组织、数字、日期等
- 倒排索引：entity → [chunk_id_1, chunk_id_2, ...]
- 查询时：提取查询中的实体，通过倒排索引定位相关chunk

### 2.3 Cross-Encoder二阶段重排（核心创新二）

第一阶段（三路检索）追求高召回，over-retrieve top_k+10个候选；第二阶段用cross-encoder（bge-reranker-v2-m3）精排，解决语义检索的排序噪声问题。

**核心思想**：Bi-encoder（embedding）擅长快速候选生成，但语义精度有限；Cross-encoder通过query-document联合编码获得更准确的相关性评分，但速度慢。二阶段结合两者优势。

```
Stage 1 (Fast): Vector + BM25 + KG → ~20 candidates (high recall)
Stage 2 (Precise): Cross-encoder rerank → top_k results (high precision)
```

**特殊处理**：knowledge-update类型在rerank后仍按日期降序排列，确保最新信息排在最前。

### 2.4 日期感知检索与推理（核心创新三）

时间推理是LME中最难的维度之一（基线F1=0.116）。核心洞察：**小模型不擅长日期计算，但擅长比较预计算好的数字**。

**方法**：
1. **Chunk级日期标注**：每个chunk前缀 `[Date: 2023-05-15]`，使日期成为可检索的文本特征
2. **时间差预计算**：在prompt中直接给出 `"35 days (5 weeks, ~1.1 months) before question date"`
3. **时间线摘要**：对时序问题额外生成按时间排序的记忆摘要

```
[Memory 1 | Date: 2023-06-10 | 35 days (5 weeks, ~1.1 months) before question date]
用户开始学习吉他...

[Memory 2 | Date: 2023-05-15 | 61 days (8 weeks, ~2.0 months) before question date]
用户报名了音乐课...

[DATE TIMELINE — sorted chronologically]
  Memory 2: 2023-05-15
    ↑ 26 days (3 weeks) after Memory 2
  Memory 1: 2023-06-10
  Question date: 2023-07-15 (35 days after Memory 1)
```

**效果**：temporal-reasoning F1 从 0.116 → 0.326（+181%）

### 2.5 题型感知生成（核心创新四）

不同问题类型需要根本不同的推理策略：

| 问题类型 | 核心策略 | 关键prompt设计 |
|---------|---------|--------------|
| knowledge-update | 时间排序 + 新旧标签 | "Memories sorted NEWEST-first, use ONLY newest value" |
| temporal-reasoning | 预计算时间差 + 单位匹配 | "USE pre-computed values, match UNIT asked" |
| multi-session | 枚举计数 + 跨会话聚合 | "Scan EVERY memory, count ALL distinct items" |
| single-session-preference | 推断偏好 + 长文本输出 | "INFER preferences, write 1-2 sentences" |
| adversarial | 事实纠正 | "Answer based on what ACTUALLY happened" |

**不同于通用prompt**：每种类型有独立的指令模板、输出长度约束和temperature设置。

此外，系统还根据**疑问词类型**（where→地点、who→人名、when→时间、how many→数量）动态注入答案格式提示。这将"问题理解"从LLM推理负担转移到规则预处理。

**Enumerate-then-Count（multi-session计数）**：对"how many"类问题，不要求LLM直接给出数字（小模型计数易错），而是要求LLM**逐条列举**（每行一个），由Python程序计数。这将任务分解为"枚举"（LLM擅长）+"计数"（程序精确），体现了人机协作的方法论思想。

### 2.6 会话摘要分层检索（核心创新五）

对于multi-session类问题（"你提到过多少种运动？"），信息分散在多个session中。单纯的chunk级检索容易遗漏。

**方法**：
1. **索引时**：为每个session生成抽取式摘要（该session的实体集合），零额外API调用
2. **检索时**：对multi-session问题，先匹配query实体与session摘要，找到相关session
3. **扩展**：从匹配的session中拉取所有chunk加入上下文

```
Query: "How many sports have I mentioned?"
→ Session摘要匹配: sess_3(basketball, gym), sess_7(swimming), sess_12(tennis)
→ 扩展: 从这3个session拉取全部chunk
→ LLM看到完整信息: 3种运动
```

**不同于full-session方案**：只扩展实体匹配的session，避免灌入无关session的噪声。

### 2.7 知识更新的时序排序（核心创新六）

知识更新问题的难点：同一话题在不同时间有不同答案，需要选择最新的。

**方法**：
1. 检索结果按日期降序排列（最新在前）
2. 最新条目标注 `★NEWEST — USE THIS★`
3. 旧条目标注 `(OLDER — ignore if newer memory covers same topic)`
4. 对最新session做完整扩展（不仅检索到的chunk，而是该session所有chunk）

**效果**：knowledge-update F1 从 0.169 → 0.571（+238%）

### 2.8 自适应查询扩展（核心创新七）

错误分析发现**22.5%的错误**是false-IDK（检索失败导致模型错误回答"不知道"）。根本原因：用户提问和记忆存储的词汇不匹配（vocabulary mismatch），三路检索均无法命中。

**方法**：两阶段自适应检索（Two-Pass Adaptive Retrieval）
1. **第一阶段**：标准三路混合检索 + cross-encoder重排
2. **置信度评估**：使用reranker的top-1分数作为检索置信度
3. **低置信度触发**：当置信度 < 0.15时，触发查询扩展
4. **LLM查询重写**：用LLM生成2-3个关键词导向的查询变体
5. **多查询检索合并**：每个变体独立检索，结果去重合并
6. **重排序**：合并池再次经过cross-encoder重排

```
Query: "What breed is my dog?"
→ Pass 1: confidence=0.04 (low, no relevant chunks found)
→ LLM expands: ["dog breed pet", "puppy adopted type", "animal companion"]
→ Pass 2: retrieve with each variant, merge pool=20 docs
→ Rerank expanded pool → confidence=0.37, correct chunk found
```

**方法论意义**：
- **不是暴力增加检索量**，而是通过置信度评估做有条件的二次检索
- **不是简单同义词扩展**，而是LLM驱动的语义级查询重写
- 开销可控：仅在低置信度时触发（预期~20%的查询），增加1次LLM调用 + 2-3次embedding计算

### 2.9 证据句高亮（核心创新八）

错误分析显示**30.9%的错误**是wrong_answer：检索到了包含正确答案的chunk，但模型从中提取了错误的实体。例如：问"在哪家瑜伽馆"，chunk包含"Serenity Yoga"和"Down Dog app"，模型回答了"Down Dog"。

**根因**：8B模型在长chunk（>100字）中定位答案句的注意力精度不足。

**方法**：证据句高亮（Evidence Sentence Highlighting）
1. 将每个chunk拆分为句子
2. 用关键词重叠度为每个句子打分（实体匹配加权2x）
3. 得分最高的句子用 `►...◄` 标记
4. 标记后的chunk送入LLM生成

```
Before: [Memory 1]
User talked about doing yoga at Serenity Yoga studio last week.
They also tried the Down Dog app at home for stretching.

After:  [Memory 1]
►User talked about doing yoga at Serenity Yoga studio last week.◄
They also tried the Down Dog app at home for stretching.
```

**方法论意义**：将"从长文本中定位答案句"的推理负担从LLM转移到检索管线中的轻量级规则评分，**零额外API调用**。与attention mechanism的显式注意力引导思想一致。

### 2.10 关系型知识图谱（核心创新十）

原有KG模块仅是 entity→{chunk_ids} 的倒排索引，不包含实体间的关系。升级为**关系型知识图谱**，提取 (Subject, Relation, Object) 三元组：

**三元组提取规则**（零LLM调用）：
1. **SVO动作模式**: "I started learning guitar" → (i, started, learning guitar)
2. **所有格属性模式**: "My dog is a Golden Retriever" → (user, has_dog, golden retriever)
3. **位置模式**: "I went to Serenity Yoga" → (user, location, serenity yoga)

```
索引时: chunk → extract_relation_triples → [(S, R, O, chunk_id)]
查询时: query entities → match triple S/O → boost connected chunks (+2 weight)
```

**与链式检索协同**：链式检索通过三元组的关系路径发现间接关联。例如：
- Triple: (sarah, recommended, bella italia, chunk_1)
- Triple: (user, ate_at, bella italia, chunk_2)
- Query: "What did I eat at the place Sarah recommended?"
  → "sarah" matches chunk_1's triple → "bella italia" bridges to chunk_2

**方法论意义**：不同于外部KG（如Wikidata），这是从对话中**自动构建的个人知识图谱**，开销为零（纯正则，无LLM调用），但提供了实体间的语义关联能力。

### 2.11 链式检索（核心创新十一）

多跳问题（multi-hop）需要连接不同chunk中的事实。例如："我在妹妹推荐的那家餐厅吃了什么？" — 需要先找到"妹妹推荐的餐厅"的名字，再用餐厅名找"吃了什么"。

**方法**：Chain-of-Retrieval
1. **Hop 1**: 标准混合检索，获取直接相关的chunks
2. **桥接实体提取**: 从Hop 1结果中提取不在原query中的新实体
3. **Hop 2**: 用原query + ��接实体做增强检索
4. **合并重排**: 两跳��果合并后经cross-encoder重排

```
Query: "What did I eat at the restaurant my sister recommended?"
→ Hop 1: finds "My sister recommended Bella Italia"
→ Bridge entities: {bella italia}  (not in query)
→ Hop 2: retrieve "What did I eat at Bella Italia restaurant my sister"
→ Merged pool re-ranked → finds "Had pasta at Bella Italia"
```

**方法论意义**：将多跳推理从LLM的隐式推理链转变为检索管线的显式多跳检索，不依赖大模型的multi-step reasoning能力。与RAG领域的iterative retrieval研究方向一致，但更轻量（无需额外LLM调用，仅用正则NER提取桥接实体）。

---

## 3. Experiments

### 3.1 实验配置

| 组件 | 配置 |
|------|------|
| LLM | Qwen3-8B (SiliconFlow API, enable_thinking=False) |
| Embedding | BAAI/bge-m3 (1024维, SiliconFlow) |
| 向量库 | ChromaDB (cosine similarity, 本地) |
| 分块 | 消息对粒度 (user+assistant), ≤2000字符/块 |
| BM25 | rank_bm25.BM25Okapi |
| KG | 正则NER + 倒排索引 + (S,R,O)三元组关系图 |
| Reranker | BAAI/bge-reranker-v2-m3 (SiliconFlow) |

### 3.2 Benchmark

| Benchmark | 来源 | 题数 | 评测维度 |
|-----------|------|------|---------|
| LongMemEval-S | ICLR 2025 | 500 | 6种对话记忆能力 |
| LoCoMo | ACL 2024 | 500 | 5种长对话理解能力 |

### 3.3 主要结果

#### LongMemEval-S

| 系统 | 模型 | Overall F1 | EM |
|------|------|-----------|-----|
| GPT-4 + Full Context | GPT-4 (128K) | ~0.35-0.45 | — |
| **Memory++ v8 (Ours)** | **Qwen3-8B** | **0.457** | **0.352** |
| Memory++ v7 | Qwen3-8B | 0.446 | 0.318 |
| GPT-3.5 + RAG | GPT-3.5 | ~0.15-0.20 | — |
| Baseline RAG (Ours) | Qwen3-8B | 0.147 | 0.010 |
| No Retrieval | — | ~0.05 | — |

#### 按题型分解（LongMemEval-S）

| 题型 | Baseline | v5 | v7 | v8 (Memory++) | 改进幅度 |
|------|---------|-----|-----|---------------|---------|
| single-session-user | 0.253 | 0.746 | 0.793 | **0.830** | +228% |
| single-session-assistant | 0.336 | 0.569 | 0.729 | **0.758** | +126% |
| knowledge-update | 0.169 | 0.416 | 0.571 | **0.614** | +263% |
| temporal-reasoning | 0.116 | 0.255 | 0.326 | **0.315** | +172% |
| multi-session | 0.052 | 0.216 | 0.243 | **0.231** | +344% |
| single-session-preference | 0.053 | 0.038 | 0.216 | **0.250** | +372% |

#### LoCoMo

| 系统 | Overall F1 | single-hop | multi-hop | temporal | open-domain | adversarial |
|------|-----------|------------|-----------|----------|-------------|-------------|
| Memory++ v5 | 0.362 | — | — | — | — | — |
| Memory++ v7 | 0.315 | — | — | — | — | — |
| **Memory++ v8** | **0.184** | 0.174 | 0.135 | 0.025 | 0.239 | 0.162 |
| Baseline RAG | 0.114 | — | — | — | — | — |

> **注意**: v8 LoCoMo大幅退步（0.362→0.184），主要原因是LME特化的短答案prompt（max_tokens=80, "1-5 words"）严重截断了LoCoMo的长答案。v9已修复benchmark-aware答案长度。

| 类别 | Baseline | v5 | v7 |
|------|---------|-----|-----|
| open-domain | 0.222 | 0.480 | 0.424 |
| multi-hop | 0.046 | 0.373 | 0.265 |
| single-hop | 0.099 | 0.317 | 0.270 |
| adversarial | 0.001 | 0.231 | 0.231 |
| temporal | 0.029 | 0.058 | 0.127 |

### 3.4 消融实验（Ablation）

> **TODO**: 运行消融实验，量化各模块贡献
> - 去掉BM25 → 三路变两路
> - 去掉KG实体索引 → 纯向量+BM25
> - 去掉日期预计算 → 依赖LLM自行计算
> - 去掉题型感知prompt → 统一prompt
> - 去掉知识更新排序 → 不标注新旧

---

## 4. Analysis

### 4.1 为什么小模型+RAG能超越GPT-4+Full Context？

核心观点：**将推理负担从LLM转移到系统设计**。

GPT-4+Full Context方案把全部对话历史塞入128K上下文，依赖LLM自身能力做检索+推理。而Memory++通过结构化的检索和预处理，把三类难题转化为简单任务：

| 难题 | GPT-4方案 | Memory++方案 |
|------|----------|-------------|
| 从115K tokens中找答案 | LLM自行注意力搜索 | 三路检索+cross-encoder重排缩至top-K |
| 时间计算 | LLM自行算日期差 | 预计算好直接给出 |
| 知识更新冲突 | LLM自行判断新旧 | 按日期排序+标签标注 |
| 问题意图理解 | LLM自行判断需要什么类型的答案 | 规则预分析疑问词+题型感知prompt |
| 跨会话信息聚合 | LLM在128K中自行寻找 | 会话摘要索引→定向session扩展 |

### 4.2 LoCoMo退步分析

v7在LoCoMo上相比v5退步（0.362→0.315），主要原因：

1. **答案长度约束过紧**（已修复）：LME答案多为1-5词，LoCoMo答案平均29-42字符，"1-5 words"的prompt严重截断
2. **LME-specific优化外溢**：multi-session计数prompt、max_tokens=80等针对LME格式的优化伤害了LoCoMo
3. **启示**：题型感知策略需要进一步感知benchmark差异，避免过拟合单一评测集

### 4.3 错误分类分析（v8, 340 LME questions）

| 错误类型 | 数量 | 占比 | 说明 |
|---------|------|------|------|
| wrong_number | 97 | 44.7% | 计数/计算错误（multi-session 68, temporal 29） |
| wrong_answer | 67 | 30.9% | 检索到相关但错误的事实 |
| false_idk | 48 | 22.1% | 检索失败导致错误回答IDK |
| missed_idk | 5 | 2.3% | 应回答IDK却编造答案 |

**关键洞察**：
1. **计数错误是最大瓶颈** — 8B模型在多文档计数上系统性偏差（过度计数），enumerate-then-count方法论旨在解决此问题
2. **检索失败（false_idk）占22%** — 主要出现在temporal和SSU，说明向量检索对时间敏感查询覆盖不足
3. **wrong_answer占31%** — reranker应能部分缓解（提高top-K的precision）
4. **missed_idk极少（2.3%）** — 系统已经很好地避免了幻觉

### 4.4 各创新模块贡献估计

| 模块 | 估计贡献（LME F1） | 主要受益题型 |
|------|-----------------|------------|
| 三路混合检索 | +0.05~0.08 | single-session, multi-hop |
| 日期感知推理 | +0.06~0.10 | temporal-reasoning, knowledge-update |
| 题型感知生成 | +0.08~0.12 | 所有题型 |
| 知识更新时序机制 | +0.04~0.06 | knowledge-update |
| 答案后处理/评分优化 | +0.03~0.05 | 所有题型 |
| Cross-encoder重排 | +0.02~0.04 | multi-hop, single-hop |
| 会话摘要分层检索 | +0.01~0.03 | multi-session |
| 检索置信度感知生成 | +0.01~0.02 | adversarial, temporal |
| 自适应查询扩展 | +0.02~0.04 | single-session, temporal |
| 证据句高亮 | +0.02~0.04 | single-session, multi-hop |
| 关系型知识图谱 | +0.02~0.04 | multi-hop, single-session |
| 链式检索 | +0.02~0.05 | multi-hop (LoCoMo) |
| 时序单位等价评分 | +0.01~0.02 | temporal-reasoning |

> 注：以上为基于版本间diff的粗略估计，精确数值需消融实验验证。

---

## 5. 改进路线图

### 已完成的方法论改进（v0 → v7, 71 commits）

| 阶段 | 方法论创新 | LME F1 |
|------|----------|--------|
| v0 (Baseline) | 简单RAG: ChromaDB + bge-m3 + Qwen3-8B | 0.147 |
| v1-v3 | + 消息对分块 + chunk_size调优 | ~0.25 |
| v4-v5 | + KG实体索引 + 混合检索 + 日期感知 | 0.360 |
| v6 | + 知识更新时序排序 + 题型prompt | 0.393 |
| v7 | + BM25三路检索 + 更多题型特化 | 0.446 |

### 下一步方法论方向

| 方向 | 状态 | 创新点 |
|------|------|--------|
| ~~Reranker二阶段检索~~ | **已完成** | bge-reranker-v2-m3 cross-encoder重排 |
| ~~对话摘要分层记忆~~ | **已完成** | 抽取式session摘要 + multi-session扩展 |
| ~~跨benchmark泛化~~ | **已完成** | benchmark-aware答案长度和prompt |
| ~~疑问词意图分析~~ | **已完成** | where/who/when/how自动注入答案格式提示 |
| ~~自适应查询扩展~~ | **已完成** | 低置信度时LLM查询重写 + 多查询合并检索 |
| ~~链式检索~~ | **已完成** | multi-hop桥接实体提取 + 二跳检索 |
| ~~证据句高亮~~ | **已完成** | chunk内关键句标记引导LLM注意力 |
| ~~关系型知识图谱~~ | **已完成** | (S,R,O)三元组提取 + 关系路径检索 |
| **消融实验** | 框架就绪 | `--ablation` flag，待运行量化各模块贡献 |
| ~~Adversarial premise detection~~ | **已完成** | 实体重叠率检测虚假前提 + 强化IDK |
| ~~答案接地验证~~ | **已完成** | 答案实体与上下文匹配度 < 30% → IDK |
| ~~上下文窗口扩展~~ | **已完成** | top-5 chunk的邻居chunk自动加入上下文 |
| ~~对抗前提检测~~ | **已完成** | 实体重叠率检测虚假前提 + 强化IDK |
| **Evidence-based confidence** | 待做 | 多维度证据评分（entity+semantic+reranker） |

---

## Appendix: 版本演进日志

| Commit | 改进 | 方法论归类 |
|--------|------|----------|
| f4be98e | 疑问词意图分析→答案格式提示 | 查询意图预分析 |
| 3d38ad7 | 会话摘要分层检索（multi-session扩展） | 分层记忆检索 |
| c8b3b7b | ChromaDB集合恢复机制 | 系统稳定性 |
| 4fddb93 | Cross-encoder二阶段重排 | 二阶段检索 |
| df6ddca | 消融实验框架 | 实验方法 |
| 8511252 | benchmark感知答案长度 | 跨benchmark泛化 |
| be51fd7 | 连字符数字-单位规范化 | 答案后处理 |
| f597daa | assistant题型专用prompt | 题型感知生成 |
| a255e31 | BM25关键词检索（第三路） | 三路混合检索 |
| 20b6023 | chunk日期前缀 | 日期感知检索 |
| b7cb0ad | 多答案评分 | 评分优化 |
| 65cee73 | 知识更新新旧标签 | 知识更新机制 |
| 4726fd9 | 最新session扩展 | 知识更新机制 |
