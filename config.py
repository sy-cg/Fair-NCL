import torch
import os


class Config:
    def __init__(self):
        # GPU优化相关
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count()
        self.use_mixed_precision = True  # 启用混合精度训练
        self.pin_memory = True if torch.cuda.is_available() else False
        self.non_blocking = True

        # 数据相关 - GPU优化
        self.data_dir = './data/ml-1m'
        self.processed_data_dir = './processed_data'
        self.cache_dir = './cache'  # 缓存目录

        # 模型相关
        self.model_name = 'sasrec'  # 'sasrec' or 'bert4rec'
        self.max_seq_len = 100
        self.hidden_units = 256  # 增加隐藏单元以更好利用GPU
        self.num_blocks = 4  # 增加层数
        self.num_heads = 8  # 增加注意力头数
        self.dropout_rate = 0.2

        # 训练相关 - GPU优化
        self.batch_size = 1024 if self.num_gpus > 1 else 512  # 根据GPU数量调整
        self.eval_batch_size = 2048  # 评估时使用更大的batch size
        self.learning_rate = 0.001
        self.num_epochs = 3
        self.patience = 15
        self.l2_emb = 1e-5
        self.gradient_clip_val = 1.0

        # 数据加载优化 - 暂时禁用多进程以避免问题
        self.num_workers = 0  # 暂时禁用多进程
        self.prefetch_factor = 2
        self.persistent_workers = False

        # 公平性增强相关
        self.use_fairness_augmentation = True
        self.epsilon = 0.5
        self.augment_ratio = 0.4    #[0.1 0.4 0.7 1]
        self.k_synonyms = 15  # 增加同义词数量
        # 效用函数权重参数（新增）
        self.content_weight = 0.4  # 内容相似度权重 [0.2, 0.4, 0.6]
        self.fairness_weight = 0.4  # 公平性权重 [0.1, 0.4, 0.7]
        self.personalization_weight = 0.2  # 个性化权重 [0.1, 0.2, 0.3]

        # 确保权重和为1
        total_weight = self.content_weight + self.fairness_weight + self.personalization_weight
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: Weights sum to {total_weight}, normalizing...")
            self.content_weight /= total_weight
            self.fairness_weight /= total_weight
            self.personalization_weight /= total_weight

        # 相似度计算相关 - GPU优化
        self.e5_model_name = 'intfloat/e5-base-v2'
        self.e5_batch_size = 64  # E5编码批次大小
        self.similarity_threshold = 0.25  # 降低阈值以获得更多同义词[0.25 0.5 0.75]
        self.use_cached_embeddings = True  # 启用嵌入缓存
        
        # 离线模式支持
        self.model_cache_dir = './model_cache'  # E5模型本地缓存目录
        if os.path.exists(self.model_cache_dir):
            print(f"Found local model cache at {self.model_cache_dir}")

        # 实验相关
        self.sensitive_attributes = ['gender', 'age_group', 'gender_age']
        self.test_ratio = 0.2
        self.val_ratio = 0.1

        # 评估相关
        self.topk_list = [5, 10, 20]
        self.eval_cache_size = 10000  # 评估缓存大小

        # 并行计算相关
        self.use_data_parallel = self.num_gpus > 1
        self.use_compile = hasattr(torch, 'compile') and torch.cuda.is_available()  # 只在GPU上编译

        # 创建目录
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.model_cache_dir, exist_ok=True)

        print(f"Configuration Summary:")
        print(f"  Device: {self.device}")
        print(f"  GPUs available: {self.num_gpus}")
        print(f"  Mixed precision: {self.use_mixed_precision}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Workers: {self.num_workers} (多进程已禁用)")
        print(f"  Model cache dir: {self.model_cache_dir}")