import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_iris

#データセットの読み込み
iris = load_iris()
x = iris.data
t = iris.target
feature_names = iris.feature_names
df = pd.DataFrame(x, columns = feature_names)
df['Target']=t

#目標値を数字から花の名前に変更
df.loc[df['Target']==0, 'Target']='setosa'
df.loc[df['Target']==1, 'Target']='versicolor'
df.loc[df['Target']==2, 'Target']='virginica'

#予想モデル構築
x_length = x[:, [0, 2]]
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_length, t)

#VSCode上のターミナルに
#streamlit run iris.py

#サイドバー（入力画面）
st.sidebar.header('Input Features')
sepalValue = st.sidebar.slider('sepal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
petalValue = st.sidebar.slider('petal length (cm)', min_value=0.0, max_value=10.0, step=0.1)

#メインパネル
st.title('Iris Classifier')
st.write('## Input Value')

#インプットデータ（1行目のデータフレーム）
value_df = pd.DataFrame([], columns=['data', 'sepal length(cm)','petal length(cm)'])
record = pd.Series(['data', sepalValue, petalValue], index=value_df.columns)
value_df = value_df.append(record, ignore_index=True)
value_df.set_index('data', inplace=True)

#入力値の値
st.write(value_df)

#予測値のデータフレーム
pred_probs = model.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs, columns=['setosa', 'versicolor', 'virginica'], index=['probability'])
st.write('## Prediction')
st.write(pred_df)

#予測結果の出力
name = pred_df.idxmax(axis=1).tolist()
st.write('## Result')
st.write('このアイリスはきっと', str(name[0]), 'です！')



