# KBS 补充实验代码审计与运行手册

## 1. 本轮实验边界

本轮只补充审稿意见需要的实验，不重新运行已有的 `ncl_sr_mce` 和
`a_fsr`，也不重新运行成本实验。统一使用：

- 数据集：ML-1M、Taobao；
- 最终种子：`42, 2025, 2026, 2027, 2028`；
- 偏斜指标：`bias_metric=prior_jsd`；
- 偏斜事件来源：`bias_event_source=unique_train_events`；
- CoSeRec：仅 SASRec；
- Caser：继续保留在 RQ1、RQ3 和跨 backbone 消融中；
- ML-1M/Caser 的 `ncl_sr_mce`：作为有诊断证据的训练失败报告，不补造测试结果。

官方可复现实验入口是 `run_research_pipeline.py`、`resume_plan_jobs.sh`
和 `run_revision_completion.sh`。根目录中的旧训练脚本仅保留用于历史兼容，
不属于本轮复现实验入口。

## 2. 审计结论

### 已确认

1. `ncl_sr_mce` 和 `a_fsr` 不会出现在新增任务计划中。
2. CoSeRec 任务只生成 SASRec 组合；注册器也会拒绝其他 backbone。
3. CoSeRec 使用开源代码中的 `Random` 增强、`NCELoss` 和 ItemCF-IUF 公式。
4. CoSeRec 的增强默认值已与开源入口对齐，包括
   `max_insert_num_per_pos=1`。
   这里复用官方增强和 NCE 损失，但保留本文统一的 prefix-sample 推荐损失、
   50 epoch 预算和全排序评估。官方默认在线相似度切换在第 160 epoch 之后，
   因而本实验应表述为“统一协议下适配的 CoSeRec”，不表述为逐行复现。
5. 所有新增任务都从已完成的五种子 Fair-NCL 任务继承数据集、backbone、
   种子和已选参数，并强制使用新偏斜定义。
6. 调参任务只保存验证集结果；最终任务才计算测试集。
7. 恢复逻辑会同时检查 `validation_results.json`、`test_results.json` 和
   `best.pt`，失败或残缺目录不会再被误判为成功结果。
8. MCE/CoSeRec 选择器会拒绝读取过测试集的调参结果、非有限指标，以及
   完全没有有限训练更新的候选。若一个数据集/backbone 的全部候选失败，
   最终计划会排除该组合并写入 `selection_manifest.json`。

### 必须在大规模运行前决定的问题

数据加载器将序列左侧补零，但 SASRec、BERT4Rec、GRU4Rec 及共享方法封装
当前使用 `有效长度 - 1` 作为最后位置。该索引只适用于右侧补零。例如
`[0, 0, 8, 9]` 的有效长度为 2，当前代码会读取位置 1，而真实最后物品在
位置 3。Taobao 的中位序列长度只有 7，因此该问题会影响大量样本。

该问题已经存在于既有五种子结果中。只修复新增实验会使新增结果与既有
Fair-NCL、baseline、NCL-SR 和 A-FSR 结果不再采用同一表示提取逻辑；修复后
进行科学上完整的比较，需要重新运行受影响的既有方法。当前代码没有修改
这一行为，以避免在未决定重跑范围时混合两套实验协议。

## 3. 新增任务矩阵

| 阶段 | 方法/目的 | 数据集与 backbone | 种子/候选 | 任务数 |
|---|---|---|---|---:|
| 验证选择 | FairAug+MCE 权重 | 2 数据集 × 4 backbone | 3 种子 × 5 候选 | 120 |
| 验证选择 | CoSeRec 对比权重 | 2 数据集 × SASRec | 3 种子 × 3 候选 | 18 |
| RQ2 | 四因素完整析因的 15 个新增单元 | 2 × 4 backbone | 5 种子 | 600 |
| RQ2 | 随机候选池控制 | 2 × 4 backbone | 5 种子 | 40 |
| RQ5 | SASRec 单因素敏感性 | 2 数据集 × 6 参数 × 2 扰动 | 5 种子 | 120 |
| 目标函数控制 | 验证选出的 FairAug+MCE | 最多 2 × 4 backbone | 5 种子 | 最多 40 |
| 方法比较 | 验证选出的 CoSeRec | 2 数据集 × SASRec | 5 种子 | 10 |

验证前固定任务共 898 个；验证选择后最多增加 50 个最终任务。若 MCE 某个
数据集/backbone 组合在三个选择种子上均无合格候选，每个排除组合减少 5 个
最终任务，并在选择清单中留下原因可追踪的空缺。

## 4. 服务器预检

在仓库根目录执行：

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import pandas, openpyxl, matplotlib; print('report dependencies ok')"
python -m unittest tests.test_plan_status tests.test_revision_completion -v
python -m unittest discover -s tests -v
bash -n run_revision_completion.sh
bash -n resume_plan_jobs.sh
python -m py_compile \
  config.py experiments/job_utils.py experiments/plan_status.py \
  experiments/revision_completion.py experiments/runner.py \
  methods/coserec.py methods/mce.py methods/registry.py methods/trainer.py \
  analysis/audit_revision_completion.py analysis/report_revision_completion.py
```

确认处理数据存在：

```bash
ls processed_data/ml-1m/processed_data.pkl
ls processed_data/taobao/processed_data.pkl
cat processed_data/ml-1m/summary.json
cat processed_data/taobao/summary.json
```

不要在同一个 `results_revision_kbs_v2` 中混入修改过预处理或表示提取逻辑后
得到的结果。

## 5. 分阶段运行命令

先设置统一目录：

```bash
export RESULTS_ROOT=results_revision_kbs_v2
export PROCESSED_ROOT=processed_data
export PLAN_DIR=experiments/revision_completion_plans
export OUTPUT_DIR=tables/supplementary/revision_completion
```

### 5.1 生成任务并审计已有数据

```bash
python -m experiments.revision_completion plan \
  --results-root "$RESULTS_ROOT" \
  --output-dir "$PLAN_DIR"

python -m analysis.audit_revision_completion \
  --results-root "$RESULTS_ROOT" \
  --plan-dir "$PLAN_DIR" \
  --output "$OUTPUT_DIR/coverage_audit.json"

python -m experiments.plan_status summary \
  --plan "$PLAN_DIR/mce_tune.jsonl" \
  --results-root "$RESULTS_ROOT"

python -m experiments.plan_status summary \
  --plan "$PLAN_DIR/coserec_tune.jsonl" \
  --results-root "$RESULTS_ROOT"
```

审计文件必须将 ML-1M/Caser/`ncl_sr_mce` 标记为
`documented_training_failure`，而不是普通缺失结果。

### 5.2 运行验证集调参

```bash
PLAN="$PLAN_DIR/mce_tune.jsonl" \
RESULTS_ROOT="$RESULTS_ROOT" \
PROCESSED_ROOT="$PROCESSED_ROOT" \
bash resume_plan_jobs.sh

PLAN="$PLAN_DIR/coserec_tune.jsonl" \
RESULTS_ROOT="$RESULTS_ROOT" \
PROCESSED_ROOT="$PROCESSED_ROOT" \
bash resume_plan_jobs.sh
```

调参目录中的 `test_results.json` 应只有
`status=not_run_by_protocol`，不能含 `selected` 测试指标。

### 5.3 固定验证选择并生成最终计划

```bash
python -m experiments.revision_completion finalize \
  --results-root "$RESULTS_ROOT" \
  --output-dir "$PLAN_DIR"

cat "$PLAN_DIR/selection_manifest.json"
wc -l "$PLAN_DIR/mce_final.jsonl"
wc -l "$PLAN_DIR/coserec_final.jsonl"
```

预期 CoSeRec 为 10 个最终任务。MCE 不超过 40 个；少于 40 时必须与
`excluded_mce_pairs` 对应，不得人工补值。

### 5.4 运行最终新增实验

```bash
for plan in \
  factorial.jsonl \
  random_pool.jsonl \
  sensitivity.jsonl \
  mce_final.jsonl \
  coserec_final.jsonl
do
  PLAN="$PLAN_DIR/$plan" \
  RESULTS_ROOT="$RESULTS_ROOT" \
  PROCESSED_ROOT="$PROCESSED_ROOT" \
  bash resume_plan_jobs.sh
done
```

每次中断后重复相同命令即可继续。任务索引和任务 ID 都是确定性的。

### 5.5 生成 RQ4、偏斜诊断和最终统计表

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
  --backbones all \
  --output "$OUTPUT_DIR/rq4_mechanism.xlsx"

python tables/make_rq4_mechanism_figures.py \
  --input "$OUTPUT_DIR/rq4_mechanism.xlsx" \
  --sheet rq4_paper_table \
  --output-dir figures/rq4 \
  --seed-count 5

# 人工核对数值与标题后，再同步论文图和可公开源数据。
cp figures/rq4/rq4_main_mechanism_figure.pdf els-cas-templates/
cp figures/rq4/rq4_attribute_specific_heatmaps.pdf els-cas-templates/
cp figures/rq4/rq4_transform_diagnostics.pdf els-cas-templates/
mkdir -p els-cas-templates/source_data/figures/rq4
cp figures/rq4/*.csv els-cas-templates/source_data/figures/rq4/

python -m analysis.skew_diagnostics \
  --processed-root "$PROCESSED_ROOT" \
  --datasets ml-1m taobao \
  --output-root tables/supplementary/skew \
  --permutations 20

python -m analysis.report_revision_completion \
  --results-root "$RESULTS_ROOT" \
  --plans \
    "$PLAN_DIR/factorial.jsonl" \
    "$PLAN_DIR/random_pool.jsonl" \
    "$PLAN_DIR/sensitivity.jsonl" \
    "$PLAN_DIR/mce_final.jsonl" \
    "$PLAN_DIR/coserec_final.jsonl" \
  --output-dir "$OUTPUT_DIR"

python -m analysis.audit_revision_completion \
  --results-root "$RESULTS_ROOT" \
  --plan-dir "$PLAN_DIR" \
  --output "$OUTPUT_DIR/coverage_audit.json" \
  --require-complete-plans
```

## 6. 一键运行

在完成第 2 节中的表示索引协议决定并通过服务器预检后，可执行：

```bash
bash run_revision_completion.sh
```

该脚本按“生成计划 → 验证调参 → 固定参数 → 最终实验 → 诊断与统计”顺序
运行，并可根据现有完整结果断点续跑。

## 7. 交付前检查

1. `coverage_audit.json` 中既有 baseline、Fair-NCL、A-FSR 都应为五种子。
2. ML-1M/Caser/`ncl_sr_mce` 应显示诊断性训练失败。
3. `selection_manifest.json` 中 CoSeRec 只能出现 SASRec。
4. 所有最终计划都使用五个种子。
5. 所有新增任务均为 `prior_jsd + unique_train_events`。
6. `direct_method_comparison.csv` 中不把效用崩溃产生的小 gap 解释为公平改进。
7. 论文主文的方法直接比较采用 SASRec；跨 backbone 结果继续保留 Caser。
8. `coverage_audit.json` 中所有新增最终计划的 `complete` 必须为 `true`，
   且 `incomplete_job_ids` 必须为空。
