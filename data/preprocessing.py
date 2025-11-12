import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch
from tqdm import tqdm


class MovieLensPreprocessor:
    """GPU优化的MovieLens预处理器"""

    def __init__(self, config):
        self.config = config

    def load_raw_data(self):
        """加载原始数据"""
        print("Loading raw MovieLens-1M data...")

        # 使用更高效的数据加载
        ratings = pd.read_csv(
            os.path.join(self.config.data_dir, 'ratings.dat'),
            sep='::',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python',
            dtype={'user_id': 'int32', 'movie_id': 'int32', 'rating': 'int8', 'timestamp': 'int64'}
        )

        users = pd.read_csv(
            os.path.join(self.config.data_dir, 'users.dat'),
            sep='::',
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
            engine='python',
            dtype={'user_id': 'int32', 'age': 'int8', 'occupation': 'int8'}
        )

        movies = pd.read_csv(
            os.path.join(self.config.data_dir, 'movies.dat'),
            sep='::',
            names=['movie_id', 'title', 'genres'],
            engine='python',
            encoding='latin-1',
            dtype={'movie_id': 'int32'}
        )

        print(f"Loaded {len(ratings)} ratings, {len(users)} users, {len(movies)} movies")
        return ratings, users, movies

    def preprocess_data(self):
        """完整的数据预处理流程"""
        ratings, users, movies = self.load_raw_data()

        # 1. 处理用户数据
        users = self.process_users(users)

        # 2. 处理评分数据
        ratings = self.process_ratings(ratings)

        # 3. 过滤数据
        ratings, users, movies = self.filter_data(ratings, users, movies)

        # 4. 重新编码ID
        ratings, users, movies, user_map, item_map = self.remap_ids(ratings, users, movies)

        # 5. 构建用户序列
        user_sequences = self.build_user_sequences_optimized(ratings)

        # 6. 数据集划分
        train_data, val_data, test_data = self.split_data_optimized(user_sequences, users)

        # 7. 保存处理后的数据
        self.save_processed_data(train_data, val_data, test_data, users, movies, user_map, item_map)

        return train_data, val_data, test_data, users, movies

    def process_users(self, users):
        """处理用户数据"""
        print("Processing user data...")

        # 处理年龄：>35 = 1, <=35 = 0
        users['age_group'] = (users['age'] > 35).astype('int8')

        # 处理性别：M = 0, F = 1
        users['gender_encoded'] = users['gender'].map({'M': 0, 'F': 1}).astype('int8')

        # 创建交叉属性
        users['gender_age'] = users['gender_encoded'].astype(str) + '_' + users['age_group'].astype(str)
        users['gender_age_encoded'] = users['gender_age'].astype('category').cat.codes.astype('int8')

        print(f"Age distribution: {users['age_group'].value_counts().to_dict()}")
        print(f"Gender distribution: {users['gender'].value_counts().to_dict()}")
        print(f"Gender-Age distribution: {users['gender_age'].value_counts().to_dict()}")

        return users

    def process_ratings(self, ratings):
        """处理评分数据"""
        print("Processing rating data...")

        # 评分二值化：>=4 = 1, <4 = 0
        ratings['binary_rating'] = (ratings['rating'] >= 4).astype('int8')

        # 只保留正反馈（评分>=4）
        positive_ratings = ratings[ratings['binary_rating'] == 1].copy()

        print(f"Original ratings: {len(ratings)}")
        print(f"Positive ratings (>=4): {len(positive_ratings)}")

        return positive_ratings

    def filter_data(self, ratings, users, movies, min_user_interactions=10, min_item_interactions=5):
        """过滤数据"""
        print("Filtering data...")

        # 迭代过滤直到稳定
        prev_users, prev_items = 0, 0
        while True:
            # 使用value_counts进行高效统计
            user_counts = ratings['user_id'].value_counts()
            item_counts = ratings['movie_id'].value_counts()

            # 过滤用户和物品
            valid_users = user_counts[user_counts >= min_user_interactions].index
            valid_items = item_counts[item_counts >= min_item_interactions].index

            # 使用isin进行高效过滤
            ratings = ratings[
                ratings['user_id'].isin(valid_users) &
                ratings['movie_id'].isin(valid_items)
                ]

            users = users[users['user_id'].isin(valid_users)]
            movies = movies[movies['movie_id'].isin(valid_items)]

            # 检查收敛
            curr_users = len(users)
            curr_items = len(movies)

            if curr_users == prev_users and curr_items == prev_items:
                break

            prev_users, prev_items = curr_users, curr_items
            print(f"  Users: {curr_users}, Movies: {curr_items}, Ratings: {len(ratings)}")

        print(f"Final data: {len(users)} users, {len(movies)} movies, {len(ratings)} ratings")
        return ratings, users, movies

    def remap_ids(self, ratings, users, movies):
        """重新映射ID到连续的整数"""
        print("Remapping IDs...")

        # 创建映射
        unique_users = sorted(ratings['user_id'].unique())
        unique_items = sorted(ratings['movie_id'].unique())

        user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
        item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}

        # 应用映射
        ratings['user_id'] = ratings['user_id'].map(user_map)
        ratings['movie_id'] = ratings['movie_id'].map(item_map)

        users['user_id'] = users['user_id'].map(user_map)
        users = users.dropna(subset=['user_id'])
        users['user_id'] = users['user_id'].astype('int32')

        movies['movie_id'] = movies['movie_id'].map(item_map)
        movies = movies.dropna(subset=['movie_id'])
        movies['movie_id'] = movies['movie_id'].astype('int32')

        print(f"Remapped to {len(user_map)} users and {len(item_map)} items")
        return ratings, users, movies, user_map, item_map

    def build_user_sequences_optimized(self, ratings):
        """GPU优化的用户序列构建"""
        print("Building user sequences with optimization...")

        # 按时间戳排序
        ratings = ratings.sort_values(['user_id', 'timestamp'])

        # 使用groupby进行高效分组
        user_sequences = {}
        for user_id, group in tqdm(ratings.groupby('user_id'), desc="Building sequences"):
            sequence = group['movie_id'].tolist()
            if len(sequence) >= 3:  # 至少需要3个交互
                user_sequences[user_id] = {
                    'sequence': sequence,
                    'timestamps': group['timestamp'].tolist(),
                    'length': len(sequence)
                }

        print(f"Built sequences for {len(user_sequences)} users")
        return user_sequences

    def split_data_optimized(self, user_sequences, users):
        """最小修改版：不分层，仅使用最后两个item划分val/test，其余为训练"""
        print("Splitting data: last-2 for val, last-1 for test, rest for train")

        train_data, val_data, test_data = [], [], []

        # 创建用户信息字典
        user_info_dict = users.set_index('user_id').to_dict('index')

        for user_id, seq_info in tqdm(user_sequences.items(), desc="Splitting user sequences"):
            sequence = seq_info['sequence']
            user_info = user_info_dict[user_id]

            if len(sequence) < 3:
                continue  # 忽略长度太短的用户

            # 训练样本：滑窗生成 input_seq → target
            for i in range(1, len(sequence) - 2):
                input_seq = sequence[:i]
                target = sequence[i]
                if len(input_seq) > self.config.max_seq_len:
                    input_seq = input_seq[-self.config.max_seq_len:]
                train_data.append({
                    'user_id': user_id,
                    'input_seq': input_seq,
                    'target': target,
                    'gender': user_info['gender_encoded'],
                    'age_group': user_info['age_group'],
                    'gender_age': user_info['gender_age_encoded']
                })

            # 验证样本（倒数第2个为目标）
            val_input = sequence[:-2]
            val_target = sequence[-2]
            if len(val_input) > self.config.max_seq_len:
                val_input = val_input[-self.config.max_seq_len:]
            val_data.append({
                'user_id': user_id,
                'input_seq': val_input,
                'target': val_target,
                'gender': user_info['gender_encoded'],
                'age_group': user_info['age_group'],
                'gender_age': user_info['gender_age_encoded']
            })

            # 测试样本（最后1个为目标）
            test_input = sequence[:-1]
            test_target = sequence[-1]
            if len(test_input) > self.config.max_seq_len:
                test_input = test_input[-self.config.max_seq_len:]
            test_data.append({
                'user_id': user_id,
                'input_seq': test_input,
                'target': test_target,
                'gender': user_info['gender_encoded'],
                'age_group': user_info['age_group'],
                'gender_age': user_info['gender_age_encoded']
            })

        print(f"Data split done: Train {len(train_data)}, Val {len(val_data)}, Test {len(test_data)}")
        return train_data, val_data, test_data


    def save_processed_data(self, train_data, val_data, test_data, users, movies, user_map, item_map):
        """保存处理后的数据"""
        print("Saving processed data...")

        data_to_save = {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'users': users,
            'movies': movies,
            'user_map': user_map,
            'item_map': item_map,
            'num_users': len(user_map),
            'num_items': len(item_map)
        }

        save_path = os.path.join(self.config.processed_data_dir, 'processed_data.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)

        print(f"Data saved to {save_path}")