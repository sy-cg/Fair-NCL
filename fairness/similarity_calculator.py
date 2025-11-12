import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from tqdm import tqdm
from torch.cuda.amp import autocast
import gc
from pathlib import Path

class OptimizedE5SimilarityCalculator:
    """GPU优化的E5相似度计算器"""

    def __init__(self, config):
        self.config = config
        self.device = config.device

        model_path = Path(config.model_cache_dir)/config.e5_model_name
        model_path = model_path.as_posix()

        # 初始化模型和分词器
        print("Loading E5 model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True
        )


        # 模型优化
        self.model.eval()
        self.model.to(self.device)

        # 如果支持，使用模型编译
        if config.use_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
            print("Model compiled for optimization")

        # 缓存路径
        self.embeddings_cache_path = os.path.join(config.cache_dir, 'movie_embeddings.pt')

    @torch.no_grad()
    def encode_texts_optimized(self, texts, batch_size=None):
        """GPU优化的文本编码"""
        if batch_size is None:
            batch_size = self.config.e5_batch_size

        # 检查缓存
        if self.config.use_cached_embeddings and os.path.exists(self.embeddings_cache_path):
            print("Loading cached embeddings...")
            return torch.load(self.embeddings_cache_path, map_location=self.device)

        print(f"Encoding {len(texts)} texts with batch size {batch_size}...")
        all_embeddings = []

        # 启用混合精度推理
        with autocast(enabled=self.config.use_mixed_precision):
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
                batch_texts = texts[i:i + batch_size]

                # 添加E5查询前缀
                batch_texts = ["query: " + text for text in batch_texts]

                # 批量分词 - 优化GPU利用率
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device, non_blocking=True)

                # 前向传播
                outputs = self.model(**inputs)

                # 平均池化
                batch_embeddings = self.average_pool_optimized(
                    outputs.last_hidden_state,
                    inputs['attention_mask']
                )

                # L2标准化
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

                all_embeddings.append(batch_embeddings)

                # 清理GPU内存
                del inputs, outputs
                if i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()

        # 合并所有嵌入
        final_embeddings = torch.cat(all_embeddings, dim=0)

        # 缓存结果
        if self.config.use_cached_embeddings:
            torch.save(final_embeddings, self.embeddings_cache_path)
            print(f"Embeddings cached to {self.embeddings_cache_path}")

        return final_embeddings

    def average_pool_optimized(self, last_hidden_states, attention_mask):
        """优化的平均池化"""
        # 使用张量操作避免循环
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def compute_similarity_matrix_gpu(self, embeddings):
        """在GPU上计算相似度矩阵"""
        print("Computing similarity matrix on GPU...")

        # 分块计算以节省GPU内存
        n_items = embeddings.shape[0]
        chunk_size = min(1000, n_items)  # 根据GPU内存调整

        similarity_matrix = torch.zeros((n_items, n_items), device=self.device)

        with autocast(enabled=self.config.use_mixed_precision):
            for i in tqdm(range(0, n_items, chunk_size), desc="Computing similarities"):
                end_i = min(i + chunk_size, n_items)

                for j in range(0, n_items, chunk_size):
                    end_j = min(j + chunk_size, n_items)

                    # 计算块相似度
                    chunk_sim = torch.mm(
                        embeddings[i:end_i],
                        embeddings[j:end_j].t()
                    )

                    similarity_matrix[i:end_i, j:end_j] = chunk_sim

        return similarity_matrix

    def compute_movie_similarities_optimized(self, movies):
        """GPU优化的电影相似度计算"""
        print("Computing movie similarities using optimized E5...")

        # 准备文本数据
        movie_texts = []
        movie_ids = []

        for _, movie in movies.iterrows():
            title = movie['title']
            genres = movie['genres'].replace('|', ', ')
            text = f"{title}. Genres: {genres}"

            movie_texts.append(text)
            movie_ids.append(movie['movie_id'])

        # 编码文本
        embeddings = self.encode_texts_optimized(movie_texts)

        # 计算相似度矩阵
        similarity_matrix = self.compute_similarity_matrix_gpu(embeddings)

        # 转换为CPU字典格式以节省GPU内存
        similarity_matrix_cpu = similarity_matrix.cpu().numpy()

        similarity_dict = {}
        for i, movie_id in enumerate(movie_ids):
            similarity_dict[movie_id] = {}
            for j, other_movie_id in enumerate(movie_ids):
                if i != j:
                    similarity_dict[movie_id][other_movie_id] = float(similarity_matrix_cpu[i][j])

        # 清理GPU内存
        del embeddings, similarity_matrix
        torch.cuda.empty_cache()

        # 保存结果
        save_path = os.path.join(self.config.processed_data_dir, 'movie_similarities.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(similarity_dict, f)

        print(f"Movie similarities saved to {save_path}")
        return similarity_dict

    def build_fair_synonyms_optimized(self, movies, similarity_dict, movie_bias_scores):
        """GPU优化的公平同义词构建"""
        print("Building fair synonyms with GPU optimization...")

        fair_synonyms = {}

        # 并行处理电影
        movie_ids = list(similarity_dict.keys())

        for movie_id in tqdm(movie_ids, desc="Building synonyms"):
            similar_movies = similarity_dict[movie_id]

            # 过滤相似度阈值
            similar_movies = {
                mid: sim for mid, sim in similar_movies.items()
                if sim >= self.config.similarity_threshold
            }

            if not similar_movies:
                fair_synonyms[movie_id] = []
                continue

            # 向量化公平性分数计算
            fair_scores = self._compute_fair_scores_vectorized(
                movie_id, similar_movies, movie_bias_scores
            )

            # 选择top-k
            sorted_synonyms = sorted(fair_scores.items(), key=lambda x: x[1], reverse=True)
            fair_synonyms[movie_id] = [mid for mid, score in sorted_synonyms[:self.config.k_synonyms]]

        # 保存结果
        save_path = os.path.join(self.config.processed_data_dir, 'fair_synonyms.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(fair_synonyms, f)

        print(f"Fair synonyms saved to {save_path}")
        return fair_synonyms

    def _compute_fair_scores_vectorized(self, original_movie, similar_movies, movie_bias_scores):
        """向量化的公平性分数计算"""
        original_bias = movie_bias_scores.get(original_movie, {'total_bias': 0})['total_bias']

        fair_scores = {}

        # 批量计算公平性分数
        for similar_id, content_sim in similar_movies.items():
            similar_bias = movie_bias_scores.get(similar_id, {'total_bias': 0})['total_bias']
            fairness_gain = max(0, original_bias - similar_bias)
            total_score = 0.6 * content_sim + 0.4 * fairness_gain
            fair_scores[similar_id] = total_score

        return fair_scores