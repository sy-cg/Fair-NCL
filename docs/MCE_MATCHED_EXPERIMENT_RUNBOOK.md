# 保持公平增强配置的表示目标比较

## 当前报告范围（2026-11-11更新）

新增10条SASRec MCE运行已全部完成：两数据集各五种子，全部存在有效训练损失，非有限更新跳过次数均为0。RQ2使用现有结果报告两个SASRec设置，并合并ML-1M/BERT4Rec及两数据集/GRU4Rec的历史五种子结果，共五个设置。各设置保留记录中的Fair-NCL增强参数、增强视图CE和当前规则解析的有效batch；偏斜资源诊断仍可追溯，但不作为本次结果报告的筛除条件。

S10比较包含50条方法运行记录（五个设置×两个目标×五种子）、10条方法汇总和70条配对指标比较。每个指标及截断的Holm族跨这五个设置，最小校正p为0.3125，没有校正后显著结果。区间使用10000次配对种子bootstrap，不重采样用户或配置选择。S10另审计全部50条正式MCE运行（旧40条＋新增10条），并单独报告10条Caser零有效更新失败；S6另将Taobao/Caser的五条NCL-SR-ID标为TF。Caser其他主比较保留。

报告依据是当前完成且有效训练的实验数据。新SASRec结果在两数据集均进入目标比较，不再要求额外补跑ML-1M参考后才报告。原ML-1M/SASRec与Taobao/SASRec的旧MCE配置分别改变了epsilon或alpha/beta，因此使用新增配置一致的SASRec结果；Taobao/BERT4Rec旧MCE也改变alpha/beta，不纳入这五个配置比较。选择依据不使用最终测试性能。

这些对照比较的是包含数值适配的实现：MCE的矩阵对数有稳定化处理，且MCE路径禁用autocast。记录增强参数一致不等于历史软件或全部资源内容相同，结果不应写成仅由抽象损失公式引起的因果效应。

## 已完成的SASRec计划：10条

冻结计划：`experiments/revision_mce_matched_plans/sasrec_mce_matched.jsonl`。这是已完成实验的来源记录。分析生成器读取该计划和结果，不再生成或覆盖计划；10条新增结果均纳入本轮五设置比较，manifest状态为`completed_and_reported`。

| 设置 | 记录的公平增强配置 | MCE目标 | 运行数 |
|---|---|---|---|
| ML-1M/SASRec | ε=1、α=0.6、β=0.4、r=0.1；semantic-hybrid | uniformity=0.5，alignment=1，γ=1，矩阵floor=1e-4 | 5/5完成并报告 |
| Taobao/SASRec | ε=0.5、α=0.6、β=0.4、r=0.1；semantic-hybrid | 同上 | 5/5完成并报告 |

两组种子均为42、2025、2026、2027、2028。学习率、隐藏维度、dropout、原始/增强CE、偏斜定义、候选构造参数、最大序列长度、划分及评估协议继承原Fair-NCL配置；batch显式固定为其当前规则解析值192/64。新增实验方法为`fair_aug_mce`。

0.5/1来自既有SASRec MCE配置，在运行之前冻结。它是固定权重的目标比较，不应称为“重新调参所得最优MCE”。GRU4Rec使用历史MCE uniformity/alignment权重1/1；ML-1M/BERT4Rec使用1/2。各epoch仍按验证HR@10选择checkpoint，测试结果不得用于改权重或挑种子。

## 完成状态检查

在保存完整结果的机器上、仓库根目录的Git Bash中执行：

```bash
export RESULTS_ROOT=results_revision_kbs_v2
export MCE_PLAN=experiments/revision_mce_matched_plans/sasrec_mce_matched.jsonl

python -m experiments.plan_status summary \
  --plan "$MCE_PLAN" --results-root "$RESULTS_ROOT"
```

当前应显示done=10/10。十条新增运行已核查有效训练且零跳过更新；失败运行须单独报告，不能把零更新的输出当成学得性能结果。这份计划不需要再次调用`revision_completion finalize`或重新训练。

## 汇总与入文

十个`revision_mce_matched_*`结果目录现已位于`results_revision_kbs_v2/<dataset>/`。在保存完整旧结果及新增结果的机器上执行：

```bash
python -m analysis.prepare_mce_objective_comparison --include-sasrec
```

`--include-sasrec`要求十条新增结果文件齐全。结果报告按记录视图参数、有效batch、数据尺寸、训练有效性及完整五种子组选择，并核对同种子测试用户集合。实际偏斜资源差异保留在配置审计中，不再据此筛除ML-1M结果。当前使用五设置、50条方法运行、70条配对比较，全部重新计算Holm校正；不沿用上轮两设置或更早三设置的p值。

S10的`_mce_new_runs.csv`、`_mce_new_summary.csv`、`_mce_new_environment.csv`分别保留新增SASRec的10条运行、2条数据集汇总和10条环境记录。配置审计覆盖50条MCE，来源任务文件`S10_objective_comparison_jobs.jsonl`含90条：50条MCE和40条唯一Fair-NCL参考。正文、附录、结论与回复应使用相同报告集合和校正族；生成脚本不会自动撰写科学结论。

## 结果解释与实验边界

所有结论同时检查NDCG、性别差距、年龄差距和最差组效用。差距降低伴随效用下降时报告为取舍，不将未显著解释为目标等效。五个种子的精确双侧检验最小p为0.0625，均值获胜或未校正bootstrap区间不含零不能写成校正后显著。

本轮使用已经完成的数据更新论文，不新增训练任务，不恢复600条析因或120条全主干MCE调参。`ncl_sr_mce`同时改变视图和增强监督，不能代替这一目标比较。BYOL、Barlow Twins属于审稿人举出的替代目标示例，本轮不额外实现这些方法。历史commit及资源内容指纹未保存、MCE与VICReg的数值路径不同、目标各自调参预算不完全可重建等事实，限制对纯目标效应和普遍优越性的解释。
