
def prefilter_items(data_train, item_features, take_n_popular):
     
    # Уберем самые популярные товары (их и так купят)
    popularity = data_train.groupby('item_id')['user_id'].nunique().reset_index() / data_train['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data_train = data_train[~data_train['item_id'].isin(top_popular)]
    
    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data_train = data_train[~data_train['item_id'].isin(top_notpopular)]
     
    # Уберем слишком дорогие товарыs
    # data_train = data_train[data_train['price'] > 60]
    
    return data_train 

    
def postfilter_items(user_id, recommednations):
    pass

