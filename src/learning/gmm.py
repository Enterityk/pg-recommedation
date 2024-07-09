# Single-target (ADBIS version)
# GMM, TMM-GS, and TMM-S
import math
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split  

#读取训练数据集
def read_data(metabase):
    data = pd.read_csv(metabase)
    data.drop(['dataset','qps','distcomp'], axis=1, inplace=True)
    return data

#Generic meta-model
def gmm(data, base_target):
    """
        Generic meta-model (GMM).
        ----------------------------------------------
        All meta-instances of our meta-dataset regarding all datasets
        were used for meta-training, except for the meta-instances
        regarding the goal dataset, which was used for meta-testing.
        ----------------------------------------------
    """
    train = data[data.index != base_target]
    test = data[data.index == base_target]
    return train, test #[test.nr_inst == test.nr_inst.max()]

#训练随机森林，在X_test上预测性能
def get_predictions(X_train, X_test, y_train, y_test):
    reg = RandomForestRegressor(n_estimators=100)
    reg.fit(X_train, y_train)
    return reg.predict(X_test)

if __name__ == "__main__":
    metabase = '/home/tyk/annmf/data/mb.csv'
    test = '/home/tyk/annmf/data/mb_test.csv'
    META_TARGETS = ['recall']
    APPROACHES = [gmm]
    
    #获取训练数据和预测目标
    get_xy = lambda x: (x.drop(META_TARGETS, axis=1), x[META_TARGETS])
    
    # reading meta-database
    data = read_data(metabase)
    
    train, val = train_test_split(data, test_size=0.25)
    
    print(train.shape)
    
    # val = read_data(test)
    # train.to_csv('/home/zjlab/tyk/annmf/data/train.csv',index=False)  
    # val.to_csv('/home/zjlab/tyk/annmf/data/val.csv',index=False)

    X_train, y_train = get_xy(train)
    X_val, y_val = get_xy(val)
    
    
    X_train.fillna(0, inplace=True)
    X_val.fillna(0, inplace=True)
    
    predictions = get_predictions(X_train, X_val, y_train, y_val)
    y_val=y_val.values
    y_val=y_val.squeeze(1)
    
    print(predictions)
    print(y_val)
    
    print(predictions.shape)
    print(y_val.shape)
    
    ans=0
    maxn=0
    minn=float('inf')
    for i in range(len(predictions)):
        cha=abs(predictions[i]-y_val[i])
        maxn=max(maxn,cha)
        minn=min(minn,cha)
        ans+=cha
    print(ans/len(predictions))
    print(maxn)
    print(minn)
    # for approach in APPROACHES:
    #         # moduling data according to approach
    #         train, test = approach(data, base_target)

    #         # train, test, and predictions
    #         X_train, y_train = get_xy(train)
    #         X_test, y_test = get_xy(test)

    #         # inducing a meta-model for each meta-target
    #         for metatarget in META_TARGETS:
    #             predictions = get_predictions(X_train, X_test, y_train[metatarget], y_test[metatarget])

    #             # formating results
    #             res = pd.DataFrame({
    #                 "base": X_test.index,
    #                 "true": y_test[metatarget].values,
    #                 "pred": predictions,
    #                 "target": [metatarget] * len(y_test),
    #                 "NN": X_test.IndexParams.values,
    #                 "R": X_test.QueryTimeParams.values,
    #                 "graph_type": X_test.graph_type.values,
    #                 "nr_inst": X_test.nr_inst.values,
    #                 "approach": [approach.__name__] * len(y_test),
    #                 "k_searching": X_test.k_searching.values
    #             })
    #             RESULTS = pd.concat([RESULTS, res])

    # RESULTS.to_csv('results/csv/true_pred_all_methods_v2.csv', index=False)

