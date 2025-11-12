import torch
import numpy as np
import pickle
import os
from collections import defaultdict
from torch.cuda.amp import autocast
import torch.nn.functional as F


# 在 dp_augmenter.py 中修改 OptimizedDifferentialPrivacyAugmenter 类

class OptimizedDifferentialPrivacyAugmenter:
    """GPU优化的差分隐私增强器"""

    def __init__(self, config, fair_synonyms, movie_bias_scores, user_history_dict=None):
        self.config = config
        self.device = config.device
        self.epsilon = config.epsilon
        self.fair_synonyms = fair_synonyms
        self.movie_bias_scores = movie_bias_scores
        self.sensitivity = 2.0

        # 添加用户历史数据
        self.user_history_dict = user_history_dict or {}

        # 添加效用函数权重
        self.content_weight = config.content_weight
        self.fairness_weight = config.fairness_weight
        self.personalization_weight = config.personalization_weight

        self._preprocess_synonyms_for_gpu()
        self._precompute_bias_tensors()
        self._precompute_similarity_tensors()
        self._precompute_user_item_similarities()  # 新增：预计算用户-物品相似度

    def _precompute_user_item_similarities(self):
        """预计算用户历史物品与候选物品的相似度"""
        print("Precomputing user-item similarities for personalization...")

        # 加载物品相似度矩阵
        similarity_path = os.path.join(self.config.processed_data_dir, 'movie_similarities.pkl')
        if os.path.exists(similarity_path):
            with open(similarity_path, 'rb') as f:
                self.item_similarity_dict = pickle.load(f)
        else:
            print("Warning: No similarity file found for personalization")
            self.item_similarity_dict = {}

    def _compute_personalization_score(self, original_movie, candidate_movies, gender, age, user_id=None):
        """基于用户历史行为计算个性化分数"""
        num_candidates = len(candidate_movies)
        personalization_scores = torch.zeros(num_candidates, device=self.device)

        # 如果没有用户ID或历史数据，返回默认分数
        if user_id is None or user_id not in self.user_history_dict:
            return torch.ones(num_candidates, device=self.device) * 0.5

        # 获取用户历史物品
        user_history = self.user_history_dict.get(user_id, set())
        if not user_history:
            return torch.ones(num_candidates, device=self.device) * 0.5

        # 计算每个候选物品与用户历史的相似度
        for i, candidate_idx in enumerate(candidate_movies):
            candidate_id = self.idx_to_movie.get(candidate_idx.item(), None)
            if candidate_id is None:
                continue

            # 计算与历史物品的平均相似度
            similarities = []
            for hist_item in user_history:
                if hist_item in self.item_similarity_dict and candidate_id in self.item_similarity_dict[hist_item]:
                    sim = self.item_similarity_dict[hist_item].get(candidate_id, 0.0)
                    similarities.append(sim)

            if similarities:
                # 使用平均相似度作为个性化分数
                personalization_scores[i] = np.mean(similarities)
            else:
                personalization_scores[i] = 0.3  # 默认较低分数

        # 归一化分数到 [0, 1]
        if personalization_scores.max() > personalization_scores.min():
            personalization_scores = (personalization_scores - personalization_scores.min()) / (
                    personalization_scores.max() - personalization_scores.min()
            )

        return personalization_scores

    def _augment_batch_gpu(self, batch):
        input_seq = batch['input_seq'].clone()
        gender = batch['gender'].squeeze(-1)
        age = batch['age_group'].squeeze(-1)
        user_ids = batch.get('user_id', None)  # 获取用户ID
        mask = input_seq > 0
        lengths = mask.sum(dim=1)

        raw_num_aug = (lengths.float() * self.config.augment_ratio).long()
        num_aug = torch.maximum(torch.ones_like(raw_num_aug), torch.minimum(raw_num_aug, lengths))

        for i in range(input_seq.size(0)):
            pos = torch.nonzero(mask[i]).squeeze(-1)
            if len(pos) == 0:
                continue
            sel = pos[torch.randperm(len(pos), device=self.device)[:num_aug[i]]]
            original_ids = input_seq[i, sel]

            # 传递用户ID
            user_id = user_ids[i].item() if user_ids is not None else None
            replaced_ids = self._dp_replace_movies_batch(
                original_ids, gender[i], age[i], user_id=user_id
            )
            input_seq[i, sel] = replaced_ids

        new_batch = batch.copy()
        new_batch['input_seq'] = input_seq
        return new_batch

    def _dp_replace_movies_batch(self, movie_ids, gender, age, user_id=None):
        """增强的差分隐私电影替换，支持基于用户历史的个性化"""
        replaced = []

        for movie_id in movie_ids:
            if movie_id.item() not in self.movie_to_idx:
                replaced.append(movie_id)
                continue

            midx = self.movie_to_idx[movie_id.item()]
            synonyms = self.synonym_matrix[midx]
            valid_mask = synonyms >= 0
            valid = synonyms[valid_mask]

            if len(valid) == 0:
                replaced.append(movie_id)
                continue

            # 1. 计算内容相似度分数
            content_similarities = self.similarity_tensor[midx][valid_mask]

            # 2. 计算公平性分数
            g_bias = self.gender_bias_tensor[valid, gender]
            a_bias = self.age_bias_tensor[valid, age]
            fairness = torch.clamp(1.0 - 0.5 * g_bias - 0.5 * a_bias, min=0.0)

            # 3. 计算个性化分数（基于用户历史）
            personalization = self._compute_personalization_score(
                movie_id, valid, gender, age, user_id
            )

            # 4. 组合效用函数
            utility = (self.content_weight * content_similarities +
                       self.fairness_weight * fairness +
                       self.personalization_weight * personalization)

            # 5. 应用差分隐私机制
            scaled_utility = self.epsilon * utility / (2 * self.sensitivity)
            probs = torch.exp(scaled_utility - scaled_utility.max())
            probs /= probs.sum()

            # 6. 采样
            sampled_idx = torch.multinomial(probs, 1).item()
            replaced.append(valid[sampled_idx])

        return torch.tensor(replaced, device=self.device)

    def _preprocess_synonyms_for_gpu(self):
        print("Preprocessing synonyms for GPU...")
        all_movies = set(self.fair_synonyms.keys())
        for synonyms_list in self.fair_synonyms.values():
            all_movies.update(synonyms_list)
        self.movie_to_idx = {movie: idx for idx, movie in enumerate(sorted(all_movies))}
        self.idx_to_movie = {idx: movie for movie, idx in self.movie_to_idx.items()}

        max_synonyms = max(len(synonyms) for synonyms in self.fair_synonyms.values()) if self.fair_synonyms else 0
        self.synonym_matrix = torch.full(
            (len(all_movies), max_synonyms), -1, dtype=torch.long, device=self.device
        )
        for movie, synonyms in self.fair_synonyms.items():
            if movie in self.movie_to_idx:
                movie_idx = self.movie_to_idx[movie]
                for i, synonym in enumerate(synonyms):
                    if synonym in self.movie_to_idx:
                        self.synonym_matrix[movie_idx, i] = self.movie_to_idx[synonym]

    def _precompute_bias_tensors(self):
        print("Precomputing bias tensors...")
        num_movies = len(self.movie_to_idx)
        self.gender_bias_tensor = torch.zeros(num_movies, 2, device=self.device)
        self.age_bias_tensor = torch.zeros(num_movies, 2, device=self.device)
        for movie, bias_info in self.movie_bias_scores.items():
            if movie in self.movie_to_idx:
                movie_idx = self.movie_to_idx[movie]
                self.gender_bias_tensor[movie_idx, 0] = bias_info.get('male_bias', 0)
                self.gender_bias_tensor[movie_idx, 1] = bias_info.get('female_bias', 0)
                self.age_bias_tensor[movie_idx, 0] = bias_info.get('age_0_bias', 0)
                self.age_bias_tensor[movie_idx, 1] = bias_info.get('age_1_bias', 0)

    def augment_batch_optimized(self, batch):
        with autocast(enabled=self.config.use_mixed_precision):
            return self._augment_batch_gpu(batch)


    def _precompute_similarity_tensors(self):
        """预计算同义词的内容相似度"""
        print("Precomputing similarity tensors...")

        # 加载相似度矩阵
        similarity_path = os.path.join(self.config.processed_data_dir, 'movie_similarities.pkl')
        if os.path.exists(similarity_path):
            with open(similarity_path, 'rb') as f:
                similarity_dict = pickle.load(f)
        else:
            # 如果没有相似度文件，使用默认值
            print("Warning: No similarity file found, using default similarity values")
            similarity_dict = {}

        # 创建相似度张量
        num_movies = len(self.movie_to_idx)
        max_synonyms = self.synonym_matrix.shape[1]
        self.similarity_tensor = torch.zeros(num_movies, max_synonyms, device=self.device)

        for movie, synonyms in self.fair_synonyms.items():
            if movie in self.movie_to_idx:
                movie_idx = self.movie_to_idx[movie]
                for i, synonym in enumerate(synonyms):
                    if synonym in self.movie_to_idx and movie in similarity_dict:
                        sim_value = similarity_dict[movie].get(synonym, 0.5)
                        self.similarity_tensor[movie_idx, i] = sim_value


def compute_movie_bias_scores_optimized(train_data, users, movies, config):
    print("Computing movie bias scores with GPU optimization...")
    movie_ids, genders, age_groups = [], [], []
    for s in train_data:
        movie_ids.append(s['target'])
        genders.append(s['gender'])
        age_groups.append(s['age_group'])

    movie_tensor = torch.LongTensor(movie_ids).to(config.device)
    gender_tensor = torch.LongTensor(genders).to(config.device)
    age_tensor = torch.LongTensor(age_groups).to(config.device)

    unique_movies = torch.unique(movie_tensor)
    bias_scores = {}

    with autocast(enabled=config.use_mixed_precision):
        for movie_id in unique_movies:
            mask = (movie_tensor == movie_id)
            count = mask.sum().item()
            if count < 10:
                continue
            g = gender_tensor[mask]
            a = age_tensor[mask]
            m, f = (g == 0).sum().item(), (g == 1).sum().item()
            y, o = (a == 0).sum().item(), (a == 1).sum().item()
            mr, fr = m / count, f / count
            yr, or_ = y / count, o / count
            bias_scores[movie_id.item()] = {
                'male_bias': max(0, mr - fr),
                'female_bias': max(0, fr - mr),
                'age_0_bias': max(0, yr - or_),
                'age_1_bias': max(0, or_ - yr),
                'gender_bias': abs(mr - fr),
                'age_bias': abs(yr - or_),
                'total_bias': abs(mr - fr) + abs(yr - or_)
            }

    print(f"Computed bias scores for {len(bias_scores)} movies")
    return bias_scores
