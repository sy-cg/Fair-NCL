# KBS 精简补充实验流程

> 2026-11-11更新：新增10条SASRec MCE已全部完成，有效训练且零次非有限更新跳过。按当前已完成数据报告S10的五个配置比较：ML-1M/SASRec、BERT4Rec、GRU4Rec及Taobao/SASRec、GRU4Rec；包含50条方法记录、70条配对统计，Holm按每个指标/截断跨五组校正，最小p=0.3125。偏斜资源诊断保留在来源审计中，不作为本轮结果报告的排除条件；两数据集SASRec结果均已纳入。无需新增ML-1M参考或重跑MCE。详见[MCE_MATCHED_EXPERIMENT_RUNBOOK.md](MCE_MATCHED_EXPERIMENT_RUNBOOK.md)。下文保留此前CoSeRec/RQ5批次的原始流程与64次训练计数；原600条析因与120条MCE调参仍不作为本次补充清单。

## 1. 实验边界

精简方案只新增两类训练：

1. CoSeRec：18 个验证集调参任务和 10 个五种子正式任务；
2. RQ5 核心参数敏感性：36 个三种子任务。

完整析因、随机替换和 FairAug+MCE 不再重新运行。RQ2 使用
`results_revision_kbs_v2` 中已有的 SASRec 五种子定向消融；RQ4 和敏感属性
分析使用已有检查点和逐用户预测结果。RQ1/RQ3 继续使用已有的两数据集、
四 backbone、五种子主实验。

精简方案共新增 64 次训练。若论文不保留 RQ5 敏感性实验，只执行 CoSeRec，
新增训练数为 28。

精简计划只要求 ML-1M/SASRec 和 Taobao/SASRec 各有五个完整的 Fair-NCL
参考结果；BERT4Rec、GRU4Rec 或 Caser 的历史结果缺失不会阻止计划生成。
完整模式仍要求两数据集、四 backbone 的五种子矩阵完整。

不要执行 `run_revision_completion.sh`，该脚本对应原始完整方案，会运行析因、
随机替换和 MCE 计划。

## 2. 设置目录

在仓库根目录执行：

```bash
export RESULTS_ROOT=results_revision_kbs_v2
export PROCESSED_ROOT=processed_data
export PLAN_DIR=experiments/revision_lean_plans
export OUTPUT_DIR=tables/supplementary/revision_lean
export REFERENCE_PLAN=experiments/revision_supplementary_final_valid.jsonl
mkdir -p "$PLAN_DIR" "$OUTPUT_DIR"
```

精简计划使用独立目录，避免覆盖
`experiments/revision_completion_plans` 中的原始完整计划。

## 3. 运行前检查

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m unittest tests.test_revision_completion -v
python -m py_compile experiments/revision_completion.py analysis/report_revision_completion.py
ls processed_data/ml-1m/processed_data.pkl
ls processed_data/taobao/processed_data.pkl
```

新增实验必须继续使用现有结果采用的预处理、序列表示提取逻辑和新偏斜定义。
不要在同一个结果目录中混入修改了数据协议或表示索引后的运行。

## 4. 生成精简计划

```bash
python -m experiments.revision_completion plan \
  --profile lean \
  --results-root "$RESULTS_ROOT" \
  --reference-plan "$REFERENCE_PLAN" \
  --output-dir "$PLAN_DIR"
```

`--reference-plan`只读取已经冻结的 Fair-NCL 任务参数，不读取其中的测试指标，
因此生成计划的实验服务器不需要存放旧 Fair-NCL 检查点。

检查任务量：

```bash
python -m experiments.plan_status summary \
  --plan "$PLAN_DIR/coserec_tune.jsonl" \
  --results-root "$RESULTS_ROOT"

python -m experiments.plan_status summary \
  --plan "$PLAN_DIR/sensitivity.jsonl" \
  --results-root "$RESULTS_ROOT"
```

预期：

- `coserec_tune.jsonl`：18 个任务；
- `sensitivity.jsonl`：36 个任务；
- 不生成 `factorial.jsonl`、`random_pool.jsonl` 或 `mce_tune.jsonl`。

敏感性计划只包含 SASRec、种子 `42/2025/2026` 和三个参数轴：
`epsilon`、`augment_ratio`、`utility_beta`。每个参数使用中心点两侧各一个值，
中心点直接复用已有 Fair-NCL 结果。

## 5. CoSeRec：验证集调参

```bash
PLAN="$PLAN_DIR/coserec_tune.jsonl" \
RESULTS_ROOT="$RESULTS_ROOT" \
PROCESSED_ROOT="$PROCESSED_ROOT" \
bash resume_plan_jobs.sh
```

调参阶段只允许保存验证集指标。检查完成状态：

```bash
python -m experiments.plan_status summary \
  --plan "$PLAN_DIR/coserec_tune.jsonl" \
  --results-root "$RESULTS_ROOT"
```

结果应为 `done=18`。每个调参目录中的 `test_results.json` 应标记
`status=not_run_by_protocol`。

## 6. 固定CoSeRec参数并生成正式计划

```bash
python -m experiments.revision_completion finalize \
  --profile lean \
  --results-root "$RESULTS_ROOT" \
  --reference-plan "$REFERENCE_PLAN" \
  --output-dir "$PLAN_DIR"
```

检查选择结果和任务量：

```bash
cat "$PLAN_DIR/selection_manifest.json"
wc -l "$PLAN_DIR/coserec_final.jsonl"
```

预期生成 10 个正式任务：两个数据集、SASRec、五个种子。如果某个数据集的
三个候选都没有完整的三种子有效验证结果，该数据集会被排除，并在
`selection_manifest.json` 中记录原因。

## 7. CoSeRec：五种子正式实验

```bash
PLAN="$PLAN_DIR/coserec_final.jsonl" \
RESULTS_ROOT="$RESULTS_ROOT" \
PROCESSED_ROOT="$PROCESSED_ROOT" \
bash resume_plan_jobs.sh
```

检查：

```bash
python -m experiments.plan_status summary \
  --plan "$PLAN_DIR/coserec_final.jsonl" \
  --results-root "$RESULTS_ROOT"
```

预期为 `done=10`，除非选择清单明确记录了排除的数据集。

## 8. RQ5核心参数敏感性

仅在论文或补充材料保留 RQ5 时执行：

```bash
PLAN="$PLAN_DIR/sensitivity.jsonl" \
RESULTS_ROOT="$RESULTS_ROOT" \
PROCESSED_ROOT="$PROCESSED_ROOT" \
bash resume_plan_jobs.sh
```

检查：

```bash
python -m experiments.plan_status summary \
  --plan "$PLAN_DIR/sensitivity.jsonl" \
  --results-root "$RESULTS_ROOT"
```

预期为 `done=36`。三种子结果用于趋势和稳定性分析，不作显著性结论。

## 9. RQ4和偏斜诊断

RQ4使用已有模型检查点，不重新训练：

```bash
python -m analysis.export_revision_mechanisms \
  --results-root "$RESULTS_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --methods fair_ncl baseline

python run_research_pipeline.py summarize-mechanism \
  --jobs experiments/revision_supplementary_final_valid.jsonl \
  --results-root "$RESULTS_ROOT" \
  --phase supplementary \
  --datasets ml-1m taobao \
  --backbones sasrec \
  --output "$OUTPUT_DIR/rq4_mechanism_sasrec.xlsx"

python tables/make_rq4_mechanism_figures.py \
  --input "$OUTPUT_DIR/rq4_mechanism_sasrec.xlsx" \
  --sheet rq4_paper_table \
  --output-dir figures/rq4_sasrec \
  --seed-count 5
```

数据层偏斜诊断同样不训练模型：

```bash
python -m analysis.skew_diagnostics \
  --processed-root "$PROCESSED_ROOT" \
  --datasets ml-1m taobao \
  --output-root tables/supplementary/skew \
  --permutations 20
```

性别、年龄和交叉群体结果从已有逐用户预测中聚合，不分别训练属性专用模型。

## 10. 汇总新增结果

完成CoSeRec和敏感性任务后执行：

结果汇总需要读取 ML-1M/SASRec 和 Taobao/SASRec 的既有五种子 Fair-NCL
测试结果。若训练服务器没有这些历史目录，应将新增结果下载到保存完整论文
结果的机器上汇总，或者先把这10个Fair-NCL结果目录复制到同一个
`$RESULTS_ROOT`。

```bash
python -m analysis.report_revision_completion \
  --results-root "$RESULTS_ROOT" \
  --plans \
    "$PLAN_DIR/coserec_final.jsonl" \
    "$PLAN_DIR/sensitivity.jsonl" \
  --output-dir "$OUTPUT_DIR"
```

如果不运行RQ5，只传入CoSeRec正式计划：

```bash
python -m analysis.report_revision_completion \
  --results-root "$RESULTS_ROOT" \
  --plans "$PLAN_DIR/coserec_final.jsonl" \
  --output-dir "$OUTPUT_DIR"
```

主要输出：

- `direct_method_comparison.csv`：已有Baseline、NCL-SR-style、A-FSR、
  Fair-NCL和新增CoSeRec；
- `paired_summary.csv`：CoSeRec五种子对照，以及执行RQ5时的三种子敏感性；
- `selection_manifest.json`：CoSeRec验证选择及排除原因。

## 11. 论文数据使用

- RQ1/RQ3：继续使用现有四backbone五种子主结果；
- RQ2：继续使用现有SASRec五种子定向消融，不运行完整析因；
- 方法定位：SASRec上报告CoSeRec、Similarity-only、NCL-SR-style MCE、
  FairAug+MCE、A-FSR和Fair-NCL；
- RQ4：主文只展示SASRec，敏感属性和交叉群体使用已有逐用户结果；
- RQ5：如保留，只报告三个核心增强参数的三种子趋势。

