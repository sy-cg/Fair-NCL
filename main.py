import os
import pickle
import torch
import torch.backends.cudnn as cudnn
from config import Config
from data.preprocessing import MovieLensPreprocessor
from data.dataset import create_optimized_data_loaders
from fairness.similarity_calculator import OptimizedE5SimilarityCalculator
from fairness.dp_augmenter import OptimizedDifferentialPrivacyAugmenter, compute_movie_bias_scores_optimized
from fairness.evaluation import OptimizedFairnessEvaluator
from train import train_model_optimized, load_best_model_optimized
import warnings
import time
import argparse
import pandas as pd


warnings.filterwarnings('ignore')

# 在main.py开头更新导入
from models.sasrec import OptimizedSASRec as SASRec
from models.bert4rec import OptimizedBERT4Rec as BERT4Rec
from utils.utils import set_seed, create_directory_structure, print_gpu_utilization


# 在main函数开始时添加
def run_all_experiments_optimized():
    """运行所有GPU优化的实验"""
    # 设置随机种子
    set_seed(42)

    # 创建目录结构
    create_directory_structure('.')

    # 打印GPU信息
    print_gpu_utilization()

    # GPU优化设置
    setup_gpu_optimization()


# GPU优化设置
def setup_gpu_optimization():
    """设置GPU优化参数"""
    if torch.cuda.is_available():
        # 启用cudnn基准测试
        cudnn.benchmark = True
        cudnn.deterministic = False

        # 设置内存增长策略
        torch.cuda.empty_cache()

        # 打印GPU信息
        print(f"GPU Optimization Setup:")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory // 1024 ** 3} GB)")

        # 设置内存管理
        torch.cuda.set_per_process_memory_fraction(0.95)  # 使用95%的GPU内存

    else:
        print("CUDA not available, using CPU")


def load_processed_data_optimized(config):
    """GPU优化的数据加载"""
    data_path = os.path.join(config.processed_data_dir, 'processed_data.pkl')

    if os.path.exists(data_path):
        print("Loading processed data...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        print("Processed data not found. Running preprocessing...")
        preprocessor = MovieLensPreprocessor(config)
        train_data, val_data, test_data, users, movies = preprocessor.preprocess_data()

        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data


def setup_fairness_components_optimized(config, data):
    """GPU优化的公平性组件设置"""
    similarity_path = os.path.join(config.processed_data_dir, 'movie_similarities.pkl')
    synonyms_path = os.path.join(config.processed_data_dir, 'fair_synonyms.pkl')
    bias_path = os.path.join(config.processed_data_dir, 'movie_bias_scores.pkl')

    # 构建用户历史字典
    user_history_dict = {}
    for sample in data['train_data']:
        uid = sample['user_id']
        if uid not in user_history_dict:
            user_history_dict[uid] = set()
        user_history_dict[uid].update(sample['input_seq'])
        user_history_dict[uid].add(sample['target'])

    # 检查缓存
    if (os.path.exists(similarity_path) and
            os.path.exists(synonyms_path) and
            os.path.exists(bias_path)):

        print("Loading cached fairness components...")
        with open(synonyms_path, 'rb') as f:
            fair_synonyms = pickle.load(f)
        with open(bias_path, 'rb') as f:
            movie_bias_scores = pickle.load(f)
    else:
        print("Computing fairness components with GPU optimization...")

        # 计算电影偏见分数
        movie_bias_scores = compute_movie_bias_scores_optimized(
            data['train_data'], data['users'], data['movies'], config)

        # 计算相似度
        similarity_calculator = OptimizedE5SimilarityCalculator(config)
        similarity_dict = similarity_calculator.compute_movie_similarities_optimized(data['movies'])

        # 构建公平同义词
        fair_synonyms = similarity_calculator.build_fair_synonyms_optimized(
            data['movies'], similarity_dict, movie_bias_scores)

        # 保存结果
        with open(bias_path, 'wb') as f:
            pickle.dump(movie_bias_scores, f)

    # 创建GPU优化的增强器，传入用户历史数据
    augmenter = OptimizedDifferentialPrivacyAugmenter(
        config, fair_synonyms, movie_bias_scores, user_history_dict
    )

    return augmenter


from models.base_model import create_model

def create_optimized_model(config):
    """使用模型工厂创建优化模型"""
    model = create_model(config.model_name, config)
    return model


# 修改 run_experiment_optimized 函数
def run_experiment_optimized(config, model_name, use_fairness,
                             content_weight=None, fairness_weight=None,
                             augment_ratio=None, similarity_threshold=None):
    """GPU优化的实验运行，支持参数调整"""

    # 参数覆盖
    if content_weight is not None:
        config.content_weight = content_weight
    if fairness_weight is not None:
        config.fairness_weight = fairness_weight
    if augment_ratio is not None:
        config.augment_ratio = augment_ratio
    if similarity_threshold is not None:
        config.similarity_threshold = similarity_threshold

    # 确保权重和为1
    if content_weight is not None or fairness_weight is not None:
        config.personalization_weight = 1 - config.content_weight - config.fairness_weight
        _normalize_weights(config)

    # 更新实验名称以包含参数
    experiment_key = (f"{model_name}_"
                      f"{'fair' if use_fairness else 'baseline'}")
    if use_fairness:
        experiment_key += (f"_cw{config.content_weight:.2f}"
                           f"_fw{config.fairness_weight:.2f}"
                           f"_pw{config.personalization_weight:.2f}"
                           f"_ar{config.augment_ratio:.2f}"
                           f"_st{config.similarity_threshold:.2f}")

    print(f"\n{'=' * 100}")
    print(f"GPU OPTIMIZED EXPERIMENT: {experiment_key}")
    print(f"Parameters: content_weight={config.content_weight:.2f}, "
          f"fairness_weight={config.fairness_weight:.2f}, "
          f"personalization_weight={config.personalization_weight:.2f}")
    print(f"augment_ratio={config.augment_ratio:.2f}, "
          f"similarity_threshold={config.similarity_threshold:.2f}")
    print(f"{'=' * 100}")

    start_time = time.time()

    # 更新配置
    config.model_name = model_name.lower()
    config.use_fairness_augmentation = use_fairness

    # 加载数据
    print("Loading data...")
    data = load_processed_data_optimized(config)
    config.num_items = data['num_items']
    config.num_users = data['num_users']

    # 创建GPU优化的数据加载器
    print("Creating optimized data loaders...")
    train_loader, val_loader, test_loader = create_optimized_data_loaders(
        data['train_data'], data['val_data'], data['test_data'], config)

    # 设置公平性组件
    augmenter = None
    if use_fairness:
        print("Setting up fairness components...")
        augmenter = setup_fairness_components_optimized(config, data)

    # 创建模型
    print("Creating optimized model...")
    model = create_optimized_model(config)

    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")

    # 训练模型
    print("Starting optimized training...")
    train_start = time.time()
    history = train_model_optimized(model, train_loader, val_loader, config, augmenter)
    train_time = time.time() - train_start

    print(f"Training completed in {train_time:.1f}s ({train_time / 60:.1f}m)")

    # 加载最佳模型
    model = load_best_model_optimized(model, config)

    # 评估模型
    print("Starting optimized evaluation...")
    eval_start = time.time()
    evaluator = OptimizedFairnessEvaluator(config)
    test_results = evaluator.evaluate_model_optimized(model, test_loader)
    eval_time = time.time() - eval_start

    print(f"Evaluation completed in {eval_time:.1f}s")
    evaluator.print_evaluation_results(test_results)

    total_time = time.time() - start_time

    # 添加实验参数到结果中
    test_results['experiment_params'] = {
        'model_name': model_name,
        'use_fairness': use_fairness,
        'content_weight': config.content_weight,
        'fairness_weight': config.fairness_weight,
        'personalization_weight': config.personalization_weight,
        'augment_ratio': config.augment_ratio,
        'similarity_threshold': config.similarity_threshold,
        'epsilon': config.epsilon,
        'training_time': train_time,
        'evaluation_time': eval_time,
        'total_time': total_time
    }

    # 保存详细的CSV结果
    result_name = experiment_key + '_results.csv'
    csv_path = os.path.join(config.processed_data_dir, 'experiment_results', result_name)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    evaluator.save_evaluation_results(test_results, csv_path)

    print(f"Results saved to {csv_path}")
    print(f"Total experiment time: {total_time:.1f}s ({total_time / 60:.1f}m)")

    return test_results

def _normalize_weights(config):
    """确保权重和为1"""
    total = config.content_weight + config.fairness_weight + config.personalization_weight
    if total > 0:
        config.content_weight /= total
        config.fairness_weight /= total
        config.personalization_weight /= total

def run_all_experiments_optimized():
    """运行所有GPU优化的实验"""
    # GPU优化设置
    setup_gpu_optimization()

    config = Config()

    models = ['sasrec', 'bert4rec']
    fairness_settings = [False, True]

    all_results = {}
    total_start_time = time.time()

    print(f"\n{'=' * 100}")
    print("STARTING ALL GPU OPTIMIZED EXPERIMENTS")
    print(f"Models: {models}")
    print(f"Fairness settings: {['Baseline', 'Fair Augmentation']}")
    print(f"GPU configuration: {config.num_gpus} GPUs, Mixed Precision: {config.use_mixed_precision}")
    print(f"{'=' * 100}")

    for model_name in models:
        for use_fairness in fairness_settings:
            experiment_key = f"{model_name}_{'fair' if use_fairness else 'baseline'}"

            try:
                print(f"\n🚀 Starting experiment: {experiment_key}")
                results = run_experiment_optimized(config, model_name, use_fairness)
                all_results[experiment_key] = results

                # 清理GPU内存
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ Error in experiment {experiment_key}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    total_time = time.time() - total_start_time

    # 保存所有结果
    final_results_path = os.path.join(config.processed_data_dir, 'all_experiments_gpu_optimized.pkl')
    with open(final_results_path, 'wb') as f:
        pickle.dump(all_results, f)

    # 打印总结
    print_experiment_summary_optimized(all_results, total_time)

    return all_results


def print_experiment_summary_optimized(all_results, total_time):
    """打印GPU优化实验总结"""
    print("\n" + "=" * 120)
    print("🎉 GPU OPTIMIZED EXPERIMENT SUMMARY")
    print("=" * 120)

    print(f"Total execution time: {total_time:.1f}s ({total_time / 60:.1f}m {total_time / 3600:.1f}h)")
    print()

    # 创建详细结果表格
    print(
        f"{'Model':<12} {'Fairness':<10} {'R@10':<8} {'N@10':<8} {'R@20':<8} {'N@20':<8} {'G-DP':<8} {'A-DP':<8} {'GA-DP':<8} {'Time(m)':<8}")
    print("-" * 120)

    for exp_name, results in all_results.items():
        if 'test_results' not in results or 'timing' not in results:
            continue

        model_name = results['config']['model_name'].upper()
        use_fair = "✓" if results['config']['use_fairness_augmentation'] else "✗"

        perf = results['test_results']['performance']
        fair = results['test_results']['fairness']
        timing = results['timing']

        # 性能指标
        r10 = perf.get('Recall@10', 0)
        n10 = perf.get('NDCG@10', 0)
        r20 = perf.get('Recall@20', 0)
        n20 = perf.get('NDCG@20', 0)

        # 公平性指标
        g_dp = fair.get('gender', {}).get('demographic_parity', 0)
        a_dp = fair.get('age_group', {}).get('demographic_parity', 0)
        ga_dp = fair.get('gender_age', {}).get('demographic_parity', 0)

        # 时间
        exp_time = timing.get('total_time', 0) / 60

        print(f"{model_name:<12} {use_fair:<10} {r10:<8.4f} {n10:<8.4f} {r20:<8.4f} {n20:<8.4f} "
              f"{g_dp:<8.4f} {a_dp:<8.4f} {ga_dp:<8.4f} {exp_time:<8.1f}")

    print("=" * 120)

    # GPU利用率总结
    if torch.cuda.is_available():
        print(f"\n📊 GPU Memory Summary:")
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
            print(f"  GPU {i}: {mem_allocated:.1f}GB allocated, {mem_reserved:.1f}GB reserved")

    print("\n✅ All GPU optimized experiments completed successfully!")


# 完善参数扫描函数
def run_parameter_sweep():
    """运行完整的参数扫描实验"""
    setup_gpu_optimization()
    config = Config()

    # 创建结果目录
    results_dir = os.path.join(config.processed_data_dir, 'experiment_results',
                               f'sweep_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(results_dir, exist_ok=True)

    # 定义参数网格
    content_weights = [0.2, 0.4, 0.6]  #[0.2, 0.4, 0.6]
    fairness_weights = [0.4, 0.7]  #[0.1, 0.4, 0.7]
    augment_ratios = [0.1, 0.4]  #[0.1, 0.4, 0.7, 1.0]
    similarity_thresholds = [0.25, 0.5, 0.75]


    models = ['sasrec', 'bert4rec']

    all_results = []
    summary_data = []

    total_experiments = len(models) * (1 + len(content_weights) * len(fairness_weights) *
                                       len(augment_ratios) * len(similarity_thresholds))
    exp_count = 0

    print(f"\n{'=' * 100}")
    print(f"PARAMETER SWEEP EXPERIMENT")
    print(f"Total experiments to run: {total_experiments}")
    print(f"Results will be saved to: {results_dir}")
    print(f"{'=' * 100}")

    for model_name in models:
        # Baseline (无公平性)
        exp_count += 1
        print(f"\n[{exp_count}/{total_experiments}] Running baseline for {model_name}")

        try:
            results = run_experiment_optimized(config, model_name, False)

            # 保存单独的CSV
            csv_path = os.path.join(results_dir, f"{model_name}_baseline.csv")
            save_detailed_results(results, csv_path)

            # 添加到汇总数据
            summary_data.append(extract_summary(results, model_name, False))
            all_results.append((f"{model_name}_baseline", results))

        except Exception as e:
            print(f"Error in baseline {model_name}: {e}")
            continue

        # 公平性实验参数扫描
        for cw in content_weights:
            for fw in fairness_weights:
                # 计算个性化权重
                pw = 1 - cw - fw
                if pw < 0 or pw > 1:
                    continue  # 跳过无效组合

                for ar in augment_ratios:
                    for st in similarity_thresholds:
                        exp_count += 1
                        exp_name = f"{model_name}_cw{cw}_fw{fw}_pw{pw:.1f}_ar{ar}_st{st}"
                        print(f"\n[{exp_count}/{total_experiments}] Running {exp_name}")

                        try:
                            results = run_experiment_optimized(
                                config, model_name, True,
                                content_weight=cw,
                                fairness_weight=fw,
                                augment_ratio=ar,
                                similarity_threshold=st
                            )

                            # 保存单独的CSV
                            csv_path = os.path.join(results_dir, f"{exp_name}.csv")
                            save_detailed_results(results, csv_path)

                            # 添加到汇总数据
                            summary_data.append(extract_summary(
                                results, model_name, True, cw, fw, pw, ar, st
                            ))
                            all_results.append((exp_name, results))

                            # 清理GPU内存
                            torch.cuda.empty_cache()

                        except Exception as e:
                            print(f"Error in {exp_name}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue

    # 保存汇总结果
    save_sweep_summary(summary_data, results_dir)

    print(f"\n{'=' * 100}")
    print(f"PARAMETER SWEEP COMPLETED")
    print(f"Total experiments completed: {len(all_results)}/{total_experiments}")
    print(f"Results saved to: {results_dir}")
    print(f"{'=' * 100}")

    return all_results


def save_detailed_results(results, csv_path):
    """保存详细的实验结果到CSV"""
    import pandas as pd

    # 展平嵌套的结果字典
    flat_results = {}

    # 添加实验参数
    if 'experiment_params' in results:
        for key, value in results['experiment_params'].items():
            flat_results[f'param_{key}'] = value

    # 添加性能指标
    if 'performance' in results:
        for key, value in results['performance'].items():
            flat_results[f'perf_{key}'] = value

    # 添加公平性指标
    if 'fairness' in results:
        for attr, metrics in results['fairness'].items():
            for metric, value in metrics.items():
                if not isinstance(value, dict):
                    flat_results[f'fair_{attr}_{metric}'] = value

    # 创建DataFrame并保存
    df = pd.DataFrame([flat_results])
    df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to {csv_path}")


def extract_summary(results, model_name, use_fairness,
                    cw=None, fw=None, pw=None, ar=None, st=None):
    """提取实验结果摘要"""
    summary = {
        'model': model_name,
        'use_fairness': use_fairness,
        'content_weight': cw or 0,
        'fairness_weight': fw or 0,
        'personalization_weight': pw or 0,
        'augment_ratio': ar or 0,
        'similarity_threshold': st or 0,
    }

    # 添加主要性能指标
    if 'performance' in results:
        for k in [5, 10, 20]:
            summary[f'Recall@{k}'] = results['performance'].get(f'Recall@{k}', 0)
            summary[f'NDCG@{k}'] = results['performance'].get(f'NDCG@{k}', 0)
            summary[f'HR@{k}'] = results['performance'].get(f'HR@{k}', 0)

    # 添加公平性指标
    if 'fairness' in results:
        for attr in ['gender', 'age_group', 'gender_age']:
            if attr in results['fairness']:
                summary[f'{attr}_DP'] = results['fairness'][attr].get('demographic_parity', 0)
                summary[f'{attr}_EO'] = results['fairness'][attr].get('equalized_opportunity', 0)

    # 添加时间信息
    if 'experiment_params' in results:
        summary['training_time'] = results['experiment_params'].get('training_time', 0)
        summary['total_time'] = results['experiment_params'].get('total_time', 0)

    return summary


def save_sweep_summary(summary_data, results_dir):
    """保存参数扫描的汇总结果"""
    import pandas as pd

    df = pd.DataFrame(summary_data)

    # 按模型和参数排序
    df = df.sort_values(['model', 'use_fairness', 'content_weight',
                         'fairness_weight', 'augment_ratio', 'similarity_threshold'])

    # 保存完整汇总
    summary_path = os.path.join(results_dir, 'sweep_summary.csv')
    df.to_csv(summary_path, index=False)
    print(f"\nSweep summary saved to {summary_path}")

    # 保存最佳参数配置
    best_configs = find_best_configurations(df)
    best_path = os.path.join(results_dir, 'best_configurations.csv')
    best_configs.to_csv(best_path, index=False)
    print(f"Best configurations saved to {best_path}")


def find_best_configurations(df):
    """找出最佳的参数配置"""
    best_configs = []

    # 对每个模型找出最佳配置
    for model in df['model'].unique():
        model_df = df[df['model'] == model]

        # 按不同指标找最佳配置
        metrics = ['Recall@10', 'NDCG@10', 'gender_DP', 'age_group_DP']

        for metric in metrics:
            if metric in model_df.columns:
                if 'DP' in metric or 'EO' in metric:
                    # 公平性指标，越高越好
                    best_idx = model_df[metric].idxmax()
                else:
                    # 性能指标，越高越好
                    best_idx = model_df[metric].idxmax()

                best_config = model_df.loc[best_idx].to_dict()
                best_config['best_for_metric'] = metric
                best_configs.append(best_config)

    return pd.DataFrame(best_configs)

if __name__ == "__main__":
    print("🚀 Starting GPU Optimized Fair Recommendation Experiments")
    print("Models: SASRec, BERT4Rec")
    print("Method: Differential Privacy based Fair Augmentation")
    print("Dataset: MovieLens-1M")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['sasrec', 'bert4rec'])
    parser.add_argument('--fair', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--sweep', action='store_true', help='Run parameter sweep')
    parser.add_argument('--content-weight', type=float, help='Content similarity weight')
    parser.add_argument('--fairness-weight', type=float, help='Fairness weight')
    parser.add_argument('--augment-ratio', type=float, help='Augmentation ratio')

    args = parser.parse_args()

    if args.sweep:
        results = run_parameter_sweep()
    elif args.all:
        results = run_all_experiments_optimized()
    else:
        config = Config()
        results = run_experiment_optimized(
            config, args.model, args.fair,
            content_weight=args.content_weight,
            fairness_weight=args.fairness_weight,
            augment_ratio=args.augment_ratio
        )
