# 安全培训智能 Agent —— 严格评测报告

- 评测时间：2026-10-03T10:25:53
- 后端 / 模型：`openrouter` / `apodex/apodex-1.1-mini:free`
- 测试集：`eval/eval_questions_strict.json`（检索 80 条、出题 8 工种、判卷 14 条用例）
- 重复轮数：出题 3 轮 / 判卷 3 轮（随机性指标报告均值 ± 标准差）
- LLM 事实核查：开启
- LLM 调用：共 0 次（成功 — / 失败 —），总耗时 0.0 秒

## 一、检索层（80 条对抗 query）

| 指标 | 严格评测 | 原始基线（12 条 query） |
|---|---|---|
| 样本数 | 80 | 12 |
| Hit@1 | 71.2% | 未统计 |
| Hit@3 | 81.2% | 未统计 |
| Hit@6 | 92.5% | 100.0% |
| MRR | 0.776 | 1.000 |
| 关键词覆盖（top-k 口径，宽） | 98.8% | 100.0% |
| 关键词覆盖（命中来源口径，严） | 88.1% | 未统计 |

### 按 query 类别拆分（暴露「开卷式满分」背后的短板）

| 类别 | 条数 | Hit@1 | Hit@3 | Hit@k | MRR | 术语覆盖(严) |
|---|---|---|---|---|---|---|
| 口语化/现场土话 | 16 | 56.2% | 81.2% | 81.2% | 0.656 | 68.8% |
| 跨工种易混淆 | 12 | 83.3% | 83.3% | 100.0% | 0.867 | 91.7% |
| 直述式（含工种词，基线同类） | 12 | 100.0% | 100.0% | 100.0% | 1.000 | 95.8% |
| 数值条款式 | 8 | 75.0% | 87.5% | 100.0% | 0.823 | 100.0% |
| 制度流程式 | 6 | 83.3% | 100.0% | 100.0% | 0.917 | 100.0% |
| 场景症状式（不给工种名） | 14 | 64.3% | 71.4% | 92.9% | 0.720 | 92.9% |
| 错别字/变体 | 12 | 50.0% | 58.3% | 83.3% | 0.583 | 83.3% |

### 完全未命中的 query（6 条）

| id | 类别 | query | 期望来源 | 实际 Top1 |
|---|---|---|---|---|
| 13 | colloquial | 爬高上低的活儿要注意啥 | `高空|高处|四知卡` | `操作规程/集团操作规程/关于印发《中铁十一局集团有限公司安全生产操作规程》的通知.pdf` |
| 17 | colloquial | 焊东西的时候火星子乱飞咋办 | `电焊|焊` | `三级教育、四知卡/2、一线作业人员岗位安全“四知卡”（对应工种）.docx` |
| 22 | colloquial | 复工第一天要做什么安全动作 | `节后复工|复工` | `三级教育、四知卡/1、岗前“三级安全教育”组合文件 （2025年11月）.docx` |
| 33 | typo | 遂道施工安全要求 | `隧道` | `三级教育、四知卡/2、一线作业人员岗位安全“四知卡”（对应工种）.docx` |
| 34 | typo | 刚筋绑扎安全要求 | `钢筋` | `安全教育培训试题和课件/培训课件及培训材料/桥涵施工安全培训.docx` |
| 58 | scenario | 有人触电了怎么救人 | `电工|各工种` | `安全交底/安全技术交底（水波纹工班）2022.11.11.doc` |

### 命中来源但术语覆盖不全的 query（4 条）

| id | 类别 | query | 术语覆盖(严) |
|---|---|---|---|
| 2 | literal | 电工施工现场临时用电安全操作规程 | 50.0% |
| 16 | colloquial | 挖机干活的时候旁边有人行不行 | 0.0% |
| 18 | colloquial | 工地上的电焊机漏电了咋整 | 0.0% |
| 51 | cross_craft | 装载机和挖掘机作业半径内的人员管控 | 0.0% |

## 二、出题层（8 个工种）

_本轮未运行出题层。_

## 三、判卷层（3 客观题 + 2 简答题 × 14 条用例）

_本轮未运行判卷层。_

## 四、跨后端对照与外部依赖说明

| 报告 | 后端 | 已完成阶段 | 检索 Hit@6 | MRR | 出题在线率 | 一致性校验 | 知识点覆盖 | 简答 MAE | 总分 MAE | Pearson r | LLM 评分占比 | LLM 成功/总调用 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `eval_report_strict.json` | openrouter | retrieval | 92.5% | 0.776 | — | — | — | — | — | — | — | —/— |
| `eval_report_strict_offline.json` | offline | grading | — | — | — | — | — | 5.0 | 10.0 | 0.773 | 0.0% | 0/0 |
| `eval_report_strict_ollama.json` | ollama | retrieval/exam/grading | 92.5% | 0.776 | 25.0% | 0.0% | 70.3% | 7.29 | 12.14 | 0.590 | 100.0% | 16/16 |

> 注：检索层为本地向量 + BM25 + BGE 重排，与 LLM 后端无关，各报告数值应一致；不一致说明知识库或检索配置被改动过。

### 外部依赖限制（本轮实测）

1. **OpenRouter 免费额度按日限流**：`apodex/apodex-1.1-mini:free` 属免费模型，免费档上限 50 次/日（响应头 `X-RateLimit-Limit: 50`）。本轮出题/判卷阶段触发 HTTP 429 `openrouter_free_tier_daily`，因此**线上模型（openrouter）的出题与判卷指标本日无法测量**；报告中相应指标改由本地 ollama 与离线引擎给出，并已明确标注后端。
2. **本地 ollama 可用模型**：仅 `deepseek-r1:1.5b`（1.8B, Q4_K_M）可用于生成与评分，`qwen3-embedding:0.6b` 仅用于嵌入。
3. 实测本地 1.5B 模型**结构化出题能力不足**：8 工种 × 2 轮中仅 4 次返回可解析 JSON，且这 4 次都只有 4 道题、结构与一致性校验均不通过；其余均由离线引擎兜底。
4. 实测本地 1.5B 模型**无法担任事实核查裁判**：要求输出逐题 JSON 数组时，模型把试卷 JSON 原样回吐（解析失败），故 `fact_ok_rate` 标记为「不可用」，未以 0% 冒充。

**额度恢复后补测命令**（直接重跑线上层并合并进同一报告）：
```bash
python scripts/run_eval_strict.py --backend openrouter --stage exam --runs-exam 3 --tag openrouter
python scripts/run_eval_strict.py --backend openrouter --stage grading --runs-grading 3 --tag openrouter
python scripts/render_strict_report.py eval_report_strict_openrouter
```

## 五、口径与局限

1. **术语覆盖严口径**要求关键词出现在「命中的期望来源文件」分块内，而宽口径只要求出现在 top-k 任意文本中；两者差值即「被通用大文件摊薄」的程度。
2. **事实核查由同款模型兼任裁判**（LLM-as-judge），存在同源偏差与批次噪声，故按多轮报告均值 ± 标准差；结论应结合 `failed_examples` 人工复核。
3. **判卷主观题的人工期望分**按标准答案的 6 个得分点等权折算，属规则化人工标注；相关系数反映的是「与规则化人工评分的一致性」，不等同于与多位资深阅卷人的组内一致性。
4. 检索层为确定性指标（固定向量 + BM25 + 重排模型），单轮即可复现；出题与主观题判分涉及模型采样，故多轮统计。
5. 标注可信度：`expected_source` / `expected_terms` 已用脚本逐条校验（来源可匹配、术语确实出现在该来源的分块文本中），避免臆造标注。

## 六、如何复现

```bash
# 全量严格评测（检索 + 出题 + 判卷）
python scripts/run_eval_strict.py --backend openrouter --runs-exam 3 --runs-grading 3

# 分阶段执行（推荐：长任务可分段，各阶段结果自动合并落盘，可断点续跑）
python scripts/run_eval_strict.py --stage retrieval            # 纯本地，约 10 分钟
python scripts/run_eval_strict.py --stage exam                 # 出题层
python scripts/run_eval_strict.py --stage grading --tag ollama # 判卷层（可指定后端与报告名）

# 本地/离线对照
python scripts/run_eval_strict.py --backend ollama --stage grading --tag ollama
python scripts/run_eval_strict.py --backend offline --stage grading --tag offline --no-fact-check

# 渲染报告
python scripts/render_strict_report.py eval_report_strict
```

改动清单（原 `app/evaluate.py` 与 `eval/eval_questions.json` 未改动，基线仍可复现）：
- 新增测试集 `eval/eval_questions_strict.json`：检索 80 条 / 出题 8 工种 / 判卷 14 用例；
- 新增 `app/evaluate_strict.py`（严格评估器）、`scripts/run_eval_strict.py`（分阶段运行）、`scripts/render_strict_report.py`（渲染本报告）；
- 产物：`eval/eval_report_strict*.json`（机器可读）与同名 `.md`（人读）；检索明细缓存 `eval/eval_retrieval_details.json` 供换后端时复用。
