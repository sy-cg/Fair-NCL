#!/usr/bin/env python
"""
包含 FairRec 和 FFVAE 的公平性推荐系统对比实验主模块
比较: Baseline、AFRL、SM、FFVAE、FairRec
"""

import os
import sys
import pickle
import torch
import pandas as pd
import torch.backends.cudnn as cudnn
import time
import argparse
import warnings
import numpy as np
from collections import defaultdict

# 引入模型注册机制
from models.base_model import create_model

# 导入具体模型类以触发注册 (Registry)
from models.sasrec import OptimizedSASRec
from models.bert4rec import OptimizedBERT4Rec
from models.caser import Caser
from models.gru4rec import GRU4Rec
from models.sm_models import SM_SASRec, SM_BERT4Rec

# === 核心修复：在这里显式导入 AFRL 模块，触发注册机制 ===
import models.afrl_model
# 注册 FFVAE 和 FairRec
try:
    from models.ffvae_model import FFVAEForRecommendation
    from models.fairrec_model import FairRecForRecommendation
except ImportError as e:
    print(f"Warning: Model import failed: {e}")

# 添加路径
sys.path.append('.')

# 导入配置和工具
from config import Config
from utils.utils import set_seed, create_directory_structure, print_gpu_utilization

# 导入数据处理
from data.dataset import create_optimized_data_loaders
from data.afrl_adapter import create_afrl_compatible_loaders, verify_data_compatibility

# 导入训练函数
from train import train_model_optimized, load_best_model_optimized
from train_afrl import train_afrl_model_optimized, load_afrl_checkpoint
from train_sm import train_sm_model_optimized, load_sm_checkpoint
from train_ffvae import train_ffvae_model, load_ffvae_checkpoint
from train_fairrec import train_fairrec_model
from train_fair_ncl import train_fair_ncl_model

# 导入评估器
from fairness.evaluation import OptimizedFairnessEvaluator
from fairness.afrl_eval_adapter import AFRLEvaluationAdapter
from models.sm_models import SMEvaluationAdapter
from fairness.ffvae_eval_adapter import FFVAEEvaluationAdapter
from fairness.dp_augmenter import (
    OptimizedDifferentialPrivacyAugmenter,
    build_genre_similarity_candidates,
    compute_movie_bias_scores_optimized
)

warnings.filterwarnings('ignore')


def save_all_metrics_to_csv(results, save_path, method_name=""):
    """统一的函数，将所有计算的指标保存到CSV"""
    # 准备一个扁平化的字典来存储所有指标
    flat_metrics = {}

    # 添加方法名称
    if method_name:
        flat_metrics['method'] = method_name

    # 1. 保存性能指标
    if 'performance' in results:
        for metric, value in results['performance'].items():
            flat_metrics[f'perf_{metric}'] = value

    # 2. 保存公平性指标
    if 'fairness' in results:
        for attr, attr_metrics in results['fairness'].items():
            for metric, value in attr_metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_metrics[f'fair_{attr}_{metric}_{sub_key}'] = sub_value
                elif isinstance(value, (int, float, np.number)):
                    if pd.isna(value):
                        flat_metrics[f'fair_{attr}_{metric}'] = None
                    else:
                        flat_metrics[f'fair_{attr}_{metric}'] = float(value)
                else:
                    flat_metrics[f'fair_{attr}_{metric}'] = str(value)

    # 3. 特定方法的额外信息
    if 'afrl_info' in results:
        flat_metrics['afrl_mask_name'] = results['afrl_info'].get('mask_name', '')
    if 'ffvae_info' in results:
        flat_metrics['ffvae_config_name'] = results['ffvae_info'].get('config_name', '')
    if 'sm_info' in results:
        flat_metrics['sm_mask_name'] = results['sm_info'].get('mask_name', '')

    # 创建DataFrame并保存
    df = pd.DataFrame([flat_metrics])

    # 验证关键列是否存在
    spd_cols = [col for col in df.columns if 'statistical_parity_difference' in col]
    if spd_cols:
        print(f"  ✅ Data saved with {len(spd_cols)} fairness metrics columns.")

    df.to_csv(save_path, index=False)
    print(f"✅ All metrics saved to: {save_path}")

    return df


def load_processed_data_optimized(config):
    """GPU优化的数据加载"""
    data_path = _resolve_processed_data_path(config)
    print(f"Loading processed data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    if data.get('item_id_start') != 1 or _processed_data_has_zero_item(data):
        raise ValueError(
            "Processed data uses the legacy 0-based item-id schema. "
            "Regenerate it explicitly with `python run_research_pipeline.py preprocess` "
            "before running training."
        )
    return data


def _resolve_processed_data_path(config):
    dataset = getattr(config, 'dataset', None)
    candidates = []
    if dataset:
        candidates.append(os.path.join(config.processed_data_dir, dataset, 'processed_data.pkl'))
    candidates.append(os.path.join(config.processed_data_dir, 'processed_data.pkl'))

    for data_path in candidates:
        if os.path.exists(data_path):
            return data_path

    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Processed data file was not found. Training no longer runs preprocessing implicitly.\n"
        f"Checked:\n{attempted}\n"
        "Run `python run_research_pipeline.py preprocess --datasets ml-1m lastfm-1k taobao` first."
    )


def _processed_data_has_zero_item(data):
    for split in ['train_data', 'val_data', 'test_data']:
        for sample in data.get(split, [])[:1000]:
            if sample.get('target') == 0:
                return True
            if any(item == 0 for item in sample.get('input_seq', [])):
                return True
    return False


# ============================================================================
# 各个实验运行函数
# ============================================================================

def run_baseline_experiment(config, model_name, fairness_method='none'):
    """运行基线实验 (Baseline 或 AFRL)"""
    method_label = 'AFRL' if fairness_method == 'afrl' else 'Baseline'
    print(f"\n{'=' * 80}")
    print(f"RUNNING {method_label.upper()} EXPERIMENT: {model_name.upper()}")
    print(f"{'=' * 80}")

    start_time = time.time()

    # 配置调整
    if fairness_method == 'afrl':
        config.model_name = f'afrl_{model_name.lower()}'
    else:
        config.model_name = model_name.lower()

    # 加载数据
    data = load_processed_data_optimized(config)
    config.num_items = data['num_items']
    config.num_users = data['num_users']

    # 确保属性维度
    if not hasattr(config, 'attribute_dims'):
        config.attribute_dims = {'gender': 2, 'age_group': 2}

    # 创建加载器
    if fairness_method == 'afrl':
        train_loader, val_loader, test_loader = create_afrl_compatible_loaders(
            data['train_data'], data['val_data'], data['test_data'], config)
    else:
        train_loader, val_loader, test_loader = create_optimized_data_loaders(
            data['train_data'], data['val_data'], data['test_data'], config)

    # 创建模型
    print(f"Creating model: {config.model_name}...")
    model = create_model(config.model_name, config)

    # 训练
    print("Starting training...")
    if fairness_method == 'afrl':
        train_afrl_model_optimized(model, train_loader, val_loader, config)
        model = load_afrl_checkpoint(model, config)
    else:
        train_model_optimized(model, train_loader, val_loader, config, None)
        model = load_best_model_optimized(model, config)

    # 评估
    print("Starting evaluation...")
    if fairness_method == 'afrl':
        evaluator = AFRLEvaluationAdapter(config)
        test_results = evaluator.evaluate_afrl_model(model, test_loader, eval_mode='average')
    else:
        evaluator = OptimizedFairnessEvaluator(config)
        test_results = evaluator.evaluate_model_optimized(model, test_loader)

    evaluator.print_evaluation_results(test_results)

    # 保存
    csv_name = f'{model_name}_{fairness_method}_metrics.csv'
    save_all_metrics_to_csv(test_results, os.path.join(config.processed_data_dir, csv_name),
                            f'{model_name}_{fairness_method}')

    print(f"Total time: {time.time() - start_time:.1f}s")
    return test_results


def run_sm_experiment(config, model_name):
    """运行 SM (Sensitive Attribute Mitigation) 实验"""
    print(f"\n{'=' * 80}")
    print(f"RUNNING SM EXPERIMENT: {model_name.upper()}")
    print(f"{'=' * 80}")

    start_time = time.time()
    config.model_name = f'sm_{model_name.lower()}'

    data = load_processed_data_optimized(config)
    config.num_items = data['num_items']
    config.num_users = data['num_users']

    train_loader, val_loader, test_loader = create_optimized_data_loaders(
        data['train_data'], data['val_data'], data['test_data'], config)

    model = create_model(config.model_name, config)

    train_sm_model_optimized(model, train_loader, val_loader, config)
    model = load_sm_checkpoint(model, config)

    evaluator = SMEvaluationAdapter(config)
    test_results = evaluator.evaluate_sm_model(model, test_loader, eval_mode='average')
    evaluator.print_sm_evaluation_results(test_results)

    save_all_metrics_to_csv(test_results,
                            os.path.join(config.processed_data_dir, f'{model_name}_sm_metrics.csv'),
                            f'{model_name}_sm')

    return test_results


def run_ffvae_experiment(config, base_model='sasrec'):
    """运行 FFVAE 实验"""
    print(f"\n{'=' * 80}")
    print(f"RUNNING FFVAE EXPERIMENT")
    print(f"{'=' * 80}")

    start_time = time.time()
    config.model_name = 'ffvae_rec'

    # FFVAE Params
    config.ffvae_z_dim = 64
    config.ffvae_alpha = 1.0
    config.ffvae_beta = 1.0

    data = load_processed_data_optimized(config)
    config.num_items = data['num_items']
    config.num_users = data['num_users']

    train_loader, val_loader, test_loader = create_optimized_data_loaders(
        data['train_data'], data['val_data'], data['test_data'], config)

    model = create_model(config.model_name, config)

    train_ffvae_model(model, train_loader, val_loader, config)
    model = load_ffvae_checkpoint(model, config)

    evaluator = FFVAEEvaluationAdapter(config)
    test_results = evaluator.evaluate_ffvae_model(model, test_loader, eval_mode='average')
    evaluator.print_ffvae_evaluation_results(test_results)

    save_all_metrics_to_csv(test_results,
                            os.path.join(config.processed_data_dir, f'{base_model}_ffvae_metrics.csv'),
                            f'{base_model}_ffvae')

    return test_results


def run_fairrec_experiment(config, base_model_name):
    """运行 FairRec 实验"""
    print(f"\n{'=' * 80}")
    print(f"RUNNING FAIRREC EXPERIMENT (Base: {base_model_name.upper()})")
    print(f"{'=' * 80}")

    start_time = time.time()

    # 1. 配置调整
    config.model_name = 'fairrec'
    config.base_model_name = base_model_name  # 传递给 FairRecWrapper

    # FairRec Hyperparameters
    config.fairrec_alpha = 1.0  # Pred Weight
    config.fairrec_beta = 0.5  # Adv Weight
    config.fairrec_gamma = 0.01  # Ortho Weight

    # 2. 数据加载
    data = load_processed_data_optimized(config)
    config.num_items = data['num_items']
    config.num_users = data['num_users']

    train_loader, val_loader, test_loader = create_optimized_data_loaders(
        data['train_data'], data['val_data'], data['test_data'], config)

    # 3. 创建模型
    model = create_model('fairrec', config)

    # 4. 训练
    train_fairrec_model(model, train_loader, val_loader, config)

    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(config.model_save_dir, 'fairrec_best.pth')))

    # 5. 评估 (FairRec 默认使用去偏后的 uf 进行预测，因此可用通用评估器)
    print("Starting FairRec evaluation...")
    evaluator = OptimizedFairnessEvaluator(config)
    test_results = evaluator.evaluate_model_optimized(model, test_loader)

    evaluator.print_evaluation_results(test_results)

    # 保存
    save_all_metrics_to_csv(test_results,
                            os.path.join(config.processed_data_dir, f'{base_model_name}_fairrec_metrics.csv'),
                            f'{base_model_name}_fairrec')

    return test_results


def run_fair_ncl_experiment(config, base_model_name):
    """Run the proposed Fair-NCL experiment."""
    print(f"\n{'=' * 80}")
    print(f"RUNNING FAIR-NCL EXPERIMENT (Base: {base_model_name.upper()})")
    print("WARNING: main_ffvae_comparison.py uses the legacy Fair-NCL trainer, "
          "not the final semantic_hybrid Fair-NCL pipeline.")
    print("For final Fair-NCL experiments, use run_research_pipeline.py with method=fair_ncl.")
    print(f"{'=' * 80}")

    start_time = time.time()
    config.base_model_name = base_model_name
    config.model_name = f'fair_ncl_{base_model_name.lower()}'
    config.use_fairness_augmentation = True

    data = load_processed_data_optimized(config)
    config.num_items = data['num_items']
    config.num_users = data['num_users']

    train_loader, val_loader, test_loader = create_optimized_data_loaders(
        data['train_data'], data['val_data'], data['test_data'], config)

    print("Building Fair-NCL controlled perturbation resources...")
    fair_synonyms = build_genre_similarity_candidates(data['movies'], config)
    movie_bias_scores = compute_movie_bias_scores_optimized(
        data['train_data'], data['users'], data['movies'], config)
    augmenter = OptimizedDifferentialPrivacyAugmenter(config, fair_synonyms, movie_bias_scores)

    print(f"Creating base model for Fair-NCL: {base_model_name}...")
    model = create_model(base_model_name, config)

    train_fair_ncl_model(model, train_loader, val_loader, config, augmenter)
    model = load_best_model_optimized(model, config)

    evaluator = OptimizedFairnessEvaluator(config)
    test_results = evaluator.evaluate_model_optimized(model, test_loader)
    evaluator.print_evaluation_results(test_results)

    save_all_metrics_to_csv(
        test_results,
        os.path.join(config.processed_data_dir, f'{base_model_name}_fair_ncl_metrics.csv'),
        f'{base_model_name}_fair_ncl'
    )

    print(f"Total time: {time.time() - start_time:.1f}s")
    return test_results


def run_comprehensive_comparison(config, model_name):
    """
    运行全面对比实验：Baseline -> AFRL -> SM -> FFVAE -> FairRec
    增加异常捕获，防止单点失败导致实验全崩
    """
    print(f"\n{'=' * 120}")
    print(f"COMPREHENSIVE COMPARISON EXPERIMENT: {model_name.upper()}")
    print(f"Pipeline: Baseline -> AFRL -> SM -> FFVAE -> FairRec")
    print(f"{'=' * 120}")

    all_results = {}
    total_start_time = time.time()

    # 1. Baseline
    try:
        print("🏃 [1/5] Running Baseline Experiment...")
        all_results[f'{model_name}_baseline'] = run_baseline_experiment(config.copy(), model_name, 'none')
    except Exception as e:
        print(f"❌ Baseline Experiment Failed: {e}")
    finally:
        torch.cuda.empty_cache()

    # 2. Fair-NCL
    try:
        print("\n[2/6] Running Fair-NCL Experiment...")
        all_results[f'{model_name}_fair_ncl'] = run_fair_ncl_experiment(config.copy(), model_name)
    except Exception as e:
        print(f"Fair-NCL Experiment Failed: {e}")
    finally:
        torch.cuda.empty_cache()

    # 3. AFRL
    try:
        print("\n🏃 [2/5] Running AFRL Experiment...")
        all_results[f'{model_name}_afrl'] = run_baseline_experiment(config.copy(), model_name, 'afrl')
    except Exception as e:
        print(f"❌ AFRL Experiment Failed: {e}")
        # 如果是因为模型未注册，提示用户
        if "not found" in str(e):
            print("💡 提示: 请检查 models/afrl_model.py 中是否注册了该模型")
    finally:
        torch.cuda.empty_cache()

    # 3. SM
    try:
        print("\n🏃 [3/5] Running SM Experiment...")
        all_results[f'{model_name}_sm'] = run_sm_experiment(config.copy(), model_name)
    except Exception as e:
        print(f"❌ SM Experiment Failed: {e}")
        if "not found" in str(e):
            print("💡 提示: 请检查 models/sm_models.py 中是否注册了该模型")
    finally:
        torch.cuda.empty_cache()

    # 4. FFVAE
    try:
        print("\n🏃 [4/5] Running FFVAE Experiment...")
        all_results[f'{model_name}_ffvae'] = run_ffvae_experiment(config.copy(), model_name)
    except Exception as e:
        print(f"❌ FFVAE Experiment Failed: {e}")
    finally:
        torch.cuda.empty_cache()

    # 5. FairRec
    try:
        print("\n🏃 [5/5] Running FairRec Experiment...")
        all_results[f'{model_name}_fairrec'] = run_fairrec_experiment(config.copy(), model_name)
    except Exception as e:
        print(f"❌ FairRec Experiment Failed: {e}")
    finally:
        torch.cuda.empty_cache()

    # 保存所有结果
    total_time = time.time() - total_start_time
    final_pkl_path = os.path.join(config.processed_data_dir, f'{model_name}_comprehensive_full.pkl')

    # 即使部分失败也保存已有的结果
    with open(final_pkl_path, 'wb') as f:
        pickle.dump(all_results, f)

    # 打印和保存汇总
    if all_results:
        compare_all_methods(all_results)
    else:
        print("⚠️ No results generated to compare.")

    print(f"\n✅ Comprehensive experiment finished in {total_time / 60:.1f} minutes.")

    return all_results


def compare_all_methods(all_results):
    """打印所有方法的对比表"""
    print("\n" + "=" * 120)
    print("ALL METHODS COMPARISON SUMMARY")
    print("=" * 120)

    # 表头
    print(f"{'Method':<25} {'Recall@10':<12} {'NDCG@10':<12} {'G-Gap':<8} {'A-Gap':<8}")
    print("-" * 110)

    for method_key, results in all_results.items():
        perf = results.get('performance', {})
        fair = results.get('fairness', {})

        # 指标提取
        r10 = perf.get('Recall@10', 0)
        n10 = perf.get('NDCG@10', 0)

        g_dp = fair.get('gender', {}).get('group_utility_gap', 0)
        a_dp = fair.get('age_group', {}).get('group_utility_gap', 0)
        # 格式化名称 (e.g., sasrec_fairrec -> Sasrec Fairrec)
        clean_name = method_key.replace('_', ' ').title()

        print(f"{clean_name:<25} {r10:<12.4f} {n10:<12.4f} {g_dp:<8.4f} {a_dp:<8.4f}")

    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="Fairness Comparison Experiments")
    parser.add_argument('--model', type=str, choices=['sasrec', 'bert4rec', 'caser', 'gru4rec'],
                        default='sasrec', help='Base model to use')
    parser.add_argument('--method', type=str,
                        choices=['baseline', 'fair_ncl', 'afrl', 'sm', 'ffvae', 'fairrec', 'comprehensive'],
                        default='comprehensive', help='Experiment method')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--debug', action='store_true', help='Debug mode (fewer epochs)')

    args = parser.parse_args()

    # 全局设置
    print(f"🚀 Starting Fairness Experiments | Model: {args.model.upper()} | Method: {args.method.upper()}")
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print_gpu_utilization()

    # 初始化配置
    config = Config()

    # 注入参数
    config.seed = args.seed
    config.num_epochs = 5 if args.debug else args.epochs
    config.eval_interval = 1 if args.debug else config.eval_interval
    config.method = args.method
    config.base_model_name = args.model
    config.model_name = args.model
    config.dataset = getattr(config, 'dataset', None) or os.path.basename(os.path.normpath(config.data_dir))
    config.apply_memory_profile(
        dataset=config.dataset,
        method=args.method,
        backbone=args.model,
    )

    # 针对特定模型的配置补丁
    if args.model == 'caser':
        if not hasattr(config, 'filter_sizes'): config.filter_sizes = [2, 3, 4]
        if not hasattr(config, 'num_filters'): config.num_filters = 64
        if not hasattr(config, 'num_vertical_filters'): config.num_vertical_filters = 1
        print("🔧 Injected Caser specific config.")

    if args.model == 'gru4rec':
        if not hasattr(config, 'num_layers'): config.num_layers = 2
        print("🔧 Injected GRU4Rec specific config.")

    # 路由逻辑
    try:
        if args.method == 'comprehensive':
            run_comprehensive_comparison(config, args.model)

        elif args.method == 'fairrec':
            run_fairrec_experiment(config, args.model)

        elif args.method == 'fair_ncl':
            run_fair_ncl_experiment(config, args.model)

        elif args.method == 'ffvae':
            run_ffvae_experiment(config, args.model)

        elif args.method == 'sm':
            run_sm_experiment(config, args.model)

        elif args.method in ['baseline', 'afrl']:
            method_arg = 'none' if args.method == 'baseline' else 'afrl'
            run_baseline_experiment(config, args.model, method_arg)

    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
