import torch
import os
import copy


class Config:
    DATASET_MEMORY_PROFILES = {
        'ml-1m': {
            'batch_size': 256,
            'eval_batch_size': 256,
            'num_workers': 2,
            'windows_num_workers': 1,
            'pin_memory': True,
            'prefetch_factor': 2,
        },
        'lastfm-1k': {
            'batch_size': 192,
            'eval_batch_size': 256,
            'num_workers': 2,
            'windows_num_workers': 1,
            'pin_memory': True,
            'prefetch_factor': 2,
        },
        'taobao': {
            'batch_size': 96,
            'eval_batch_size': 128,
            'num_workers': 1,
            'windows_num_workers': 0,
            'pin_memory': False,
            'prefetch_factor': 1,
        },
    }

    METHOD_BATCH_FACTORS = {
        'baseline': 1.0,
        'ncl_only': 0.8,
        'random_aug': 0.8,
        'similarity_aug': 0.8,
        'wo_fairness_sampling': 0.8,
        'wo_semantic_sampling': 0.75,
        'wo_alignment': 0.75,
        'wo_variance': 0.75,
        'wo_covariance': 0.75,
        'wo_augmented_ce': 0.75,
        'random_low_skew': 0.8,
        'high_skew': 0.75,
        'fair_ncl': 0.75,
        'fair_ncl_alpha_tradeoff': 0.75,
        'fair_ncl_semantic_alpha_tradeoff': 0.75,
        'fair_ncl_semantic_hybrid_alpha_tradeoff': 0.75,
        'adv_debias': 0.75,
        'grl': 0.75,
        'sm_pcfr': 0.5,
        'afrl': 0.5,
        'pfrec': 0.7,
        'a_fsr': 0.75,
    }

    BACKBONE_BATCH_FACTORS = {
        'sasrec': 1.0,
        'bert4rec': 0.75,
        'gru4rec': 1.0,
        'caser': 0.85,
    }
    """配置类（修复版，包含copy方法）"""

    def __init__(self):
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.use_data_parallel = self.num_gpus > 1
        self.seed = 42

        # 数据路径
        self.data_dir = './data/ml-1m'
        self.processed_data_dir = './processed_data'
        self.cache_dir = './cache'
        self.model_save_dir = './models'

        # 模型基础配置
        self.model_name = 'sasrec'
        self.max_seq_len = 100
        self.hidden_units = 128
        self.num_blocks = 2
        self.num_heads = 4
        self.dropout_rate = 0.2

        # 训练配置
        self.batch_size = 256
        self.eval_batch_size = 512
        self.learning_rate = 1e-3
        self.num_epochs = 50
        self.eval_interval = 1
        self.patience = 10
        self.progress_postfix_interval = 50
        self.l2_emb = 1e-6
        self.gradient_clip_val = 5.0  # 修复：使用正确的配置名
        self.grad_clip = 1.0  # 默认值

        # GPU优化配置
        self.use_mixed_precision = True
        self.use_compile = False
        self.num_workers = 4
        self.pin_memory = True
        self.persistent_workers = True
        self.prefetch_factor = 2
        self.non_blocking = True

        # 公平性配置
        # 训练和评估只处理性别和年龄。
        self.sensitive_attributes = ['gender', 'age_group']
        # Fairness configuration.
        # Training and evaluation only handle gender and age_group.
        self.sensitive_attributes = ['gender', 'age_group']
        self.report_sensitive_attributes = ['gender', 'age_group']
        self.attribute_dims = {
            'gender': 2,
            'age_group': 2
        }
        self.use_fairness_augmentation = False

        # Fair-NCL / controlled perturbation configuration.
        self.epsilon = 1.0
        self.augment_ratio = 0.2
        self.utility_alpha = 0.7
        self.utility_beta = 0.3
        self.utility_sensitivity = 1.0
        self.similarity_threshold = 0.2
        self.k_synonyms = 20
        self.similarity_window = 20
        self.similarity_candidate_pool_multiplier = 5
        self.low_skew_pool_size = 500
        self.use_similarity_scores = False
        self.similarity_source = "semantic_hybrid"
        self.semantic_embedding_model = "BAAI/bge-m3"
        self.semantic_embedding_batch_size = 128
        self.semantic_retrieval_batch_size = 512
        self.semantic_embedding_device = None
        self.semantic_search_device = None
        self.semantic_text_prefix = None
        self.semantic_hybrid_pool_size = 50
        self.semantic_hybrid_top_k = None
        self.semantic_hashing_dim = 4096
        self.bias_min_count = 5
        self.bias_smoothing = 1.0
        self.fair_ncl_aug_rec_weight = 0.5
        self.fair_ncl_align_weight = 1.0
        self.fair_ncl_var_weight = 1.0
        self.fair_ncl_cov_weight = 0.04
        self.train_num_negative = 1
        self.eval_num_negative = 0

        # 评估配置
        self.topk_list = [5, 10, 20]

        # RQ4 mechanism-analysis configuration. Tuning phases disable this
        # automatically in the experiment runner unless explicitly overridden.
        self.export_mechanism_analysis = True
        self.mechanism_probe_folds = 5
        self.mechanism_probe_max_users = 20000
        self.mechanism_export_max_users = 5000

        # SM框架特定配置
        self.sm_lambda = 1.0
        self.sm_consistency_weight = 0.1
        self.sm_filter_dropout = 0.1
        self.sm_disc_dropout = 0.3
        self.sm_disc_steps = 1
        self.sm_pretrain_steps = 3

        # AFRL配置
        self.afrl_beta = 1.0
        self.afrl_lambda = 1.0
        self.afrl_recon_weight = 0.01
        self.afrl_freeze_backbone = True
        self.afrl_disc_steps = 5
        self.disc_pretrain_steps = 5

        # Adv-Debias / GRL配置
        self.adv_debias_weight = 0.2
        self.adv_debias_disc_lr = self.learning_rate
        self.adv_disc_steps = 1
        self.grl_weight = self.adv_debias_weight

        # PFRec风格prompt baseline配置
        self.pfrec_prompt_weight = 0.05
        self.pfrec_adv_weight = 0.2
        self.pfrec_disc_steps = 1
        self.pfrec_freeze_backbone = True

        # A-FSR风格无人口属性公平序列baseline配置
        self.afsr_uniform_weight = 0.05
        self.afsr_pattern_span = 2
        self.afsr_dns_weight = 0.5
        self.afsr_dro_weight = 0.5
        self.afsr_dro_fraction = 0.3
        self.afsr_dro_temperature = 5.0

        # FFVAE配置
        self.ffvae_z_dim = 64
        self.ffvae_alpha = 1.0
        self.ffvae_beta = 1.0
        self.ffvae_rec_weight = 1.0

        # 在 main_ffvae_comparison.py 的参数或 config 中增加：
        self.fairrec_alpha = 1.0  # 偏差捕获强度
        self.fairrec_beta = 1.0  # 对抗去偏强度
        self.fairrec_gamma = 0.5  # 正交约束强度

        # 创建必要的目录
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)

    def copy(self):
        """创建配置的深拷贝"""
        return copy.deepcopy(self)

    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Config has no attribute '{key}'")

    @staticmethod
    def _scaled_batch_size(base_value, factor):
        scaled = max(16, int(round(float(base_value) * float(factor) / 16.0) * 16))
        return scaled

    def apply_memory_profile(self, dataset=None, method=None, backbone=None, explicit_keys=None):
        """Apply conservative memory defaults for each dataset/method/backbone."""
        explicit_keys = set(explicit_keys or [])
        dataset = (dataset or getattr(self, 'dataset', 'ml-1m') or 'ml-1m').lower()
        method = (method or getattr(self, 'method', 'baseline') or 'baseline').lower()
        backbone = (
            backbone
            or getattr(self, 'base_model_name', None)
            or getattr(self, 'model_name', 'sasrec')
            or 'sasrec'
        ).lower()

        profile = self.DATASET_MEMORY_PROFILES.get(dataset, {})
        method_factor = self.METHOD_BATCH_FACTORS.get(method, 1.0)
        backbone_factor = self.BACKBONE_BATCH_FACTORS.get(backbone, 1.0)
        batch_factor = method_factor * backbone_factor

        if 'batch_size' not in explicit_keys:
            base_batch_size = profile.get('batch_size', self.batch_size)
            self.batch_size = self._scaled_batch_size(base_batch_size, batch_factor)

        if 'eval_batch_size' not in explicit_keys:
            base_eval_batch_size = profile.get('eval_batch_size', self.eval_batch_size)
            eval_factor = min(1.0, max(0.5, batch_factor * 1.2))
            self.eval_batch_size = self._scaled_batch_size(base_eval_batch_size, eval_factor)

        target_device = getattr(self.device, 'type', 'cpu')

        if 'num_workers' not in explicit_keys:
            if target_device != 'cuda':
                self.num_workers = 0
            elif os.name == 'nt':
                self.num_workers = int(profile.get('windows_num_workers', min(profile.get('num_workers', 1), 1)))
            else:
                self.num_workers = int(profile.get('num_workers', self.num_workers))

        if 'pin_memory' not in explicit_keys:
            self.pin_memory = bool(profile.get('pin_memory', self.pin_memory) and target_device == 'cuda')

        if 'persistent_workers' not in explicit_keys:
            self.persistent_workers = bool(self.num_workers > 0 and os.name != 'nt')

        if 'prefetch_factor' not in explicit_keys:
            self.prefetch_factor = int(profile.get('prefetch_factor', self.prefetch_factor)) if self.num_workers > 0 else None

        if 'non_blocking' not in explicit_keys:
            self.non_blocking = bool(self.pin_memory and target_device == 'cuda')

        if dataset == 'taobao':
            if 'k_synonyms' not in explicit_keys:
                self.k_synonyms = min(int(getattr(self, 'k_synonyms', 20)), 10)
            if 'similarity_window' not in explicit_keys:
                self.similarity_window = min(int(getattr(self, 'similarity_window', 20)), 10)

    def __repr__(self):
        """打印配置信息"""
        lines = []
        lines.append("=" * 60)
        lines.append("Configuration")
        lines.append("=" * 60)
        for key, value in sorted(vars(self).items()):
            if not key.startswith('_'):
                lines.append(f"{key:30s}: {value}")
        lines.append("=" * 60)
        return '\n'.join(lines)
