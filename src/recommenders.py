import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting=True):
        
        # Топ покупок
        self.top_purch = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purch.sort_values('quantity', ascending=False, inplace=True)
        self.top_purch = self.top_purch[self.top_purch['item_id'] != 999999]

        # Топ покупок всех
        self.overall_top_purch = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purch.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purch = self.overall_top_purch[self.overall_top_purch['item_id'] != 999999]
        self.overall_top_purch = self.overall_top_purch.item_id.tolist()
        
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data: pd.DataFrame):
        
        user_item_matrix = pd.pivot_table(data, 
                                  index='user_id', columns='item_id', 
                                  values='quantity', # Можно пробоват ьдругие варианты
                                  aggfunc='count', 
                                  fill_value=0
                                 )
        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
        
        #user_item_matrix = bm25_weight(user_item_matrix.T).T
        
        #user_item_matrix = csr_matrix(user_item_matrix).tocsr()
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model
    
    def get_recommendations(self, user, sparse_user_item, N=5):
        
        """Рекомендуем топ-N товаров"""

        res = [self.id_to_itemid[rec[0]] for rec in 
                        self.model.recommend(userid=self.userid_to_id[user], 
                                        user_items=self.user_item_matrix,   # на вход user-item matrix
                                        N=N, 
                                        filter_already_liked_items=False, 
                                        filter_items=[self.itemid_to_id[999999]],  # !!! 
                                        recalculate_user=True)]
        return res

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        
        top_users_purch = self.top_purch[self.top_purch['user_id'] == user].head(N)

        res = top_users_purch['item_id'].apply(lambda x: self.get_similar_item(x)).tolist()
        res = self.extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        res = []

        similar_users = self.model.similar_users(self.userid_to_id[user], N=N + 1)
        similar_users = [self.id_to_userid[rec[0]] for rec in similar_users]
        similar_users = similar_users[1:]  # сам пользователь нам не нужен

        for _ in similar_users:
            res.extend(self.get_own_recommendations(_, N=1))

        res = self.extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)
        top_rec = recs[1][0]
        return self.id_to_itemid[top_rec]
    
    def extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purch[:N])
            recommendations = recommendations[:N]

        return recommendations
    
    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        if user not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user: max_id})
            self.id_to_userid.update({max_id: user})
            
        return self.get_implicit_recommendations(user, model=self.own_recommender, N=N)
            
    def get_implicit_recommendations(self, user, model, N=5):
        """implicit"""

        if user not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user: max_id})
            self.id_to_userid.update({max_id: user})
            
        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                                                    user_items=csr_matrix(
                                                                        self.user_item_matrix).tocsr(),
                                                                    N=N,
                                                                    filter_already_liked_items=False,
                                                                    filter_items=None,#[self.itemid_to_id[999999]],
                                                                    recalculate_user=False)]

        res = self.extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res