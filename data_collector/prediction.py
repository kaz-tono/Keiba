# model : random forest
# used data : jockey_id, 単勝, 人気
# predict: 着順

import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# データの読み込み
# pickleファイルが存在するディレクトリのパス
directory = 'data_collector/'

# 全てのpickleファイルのパスを取得
pickle_files = glob.glob(os.path.join(directory, '*.pickle'))

# DataFrameのリストを作成
dfs = []
for file in pickle_files:
    df = pd.read_pickle(file)
    dfs.append(df)

# 全てのDataFrameを縦方向に連結
df = pd.concat(dfs, axis=0, ignore_index=True)

# タイムの変換
# 1. 文字列に変換
df['タイム'] = df['タイム'].astype(str)

# 2. 分と秒（小数点含む）に分割
df[['分', '秒']] = df['タイム'].str.split(':', expand=True)

# 3. 数値に変換
df['分'] = df['分'].astype(float)
df['秒'] = df['秒'].astype(float)

# 4. 総秒数を計算
df['タイム_秒'] = df['分'] * 60 + df['秒']

# 5. 不要な列を削除
df = df.drop(['タイム', '分', '秒'], axis=1)

# 着順の確認
print("着順の unique な値:", df['着順'].unique())

# 着順の数値変換可能なデータのみを抽出
df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
df = df.dropna(subset=['着順'])  # 着順が数値変換できなかったデータを削除

# 着順を二値に変換（1着=1, その他=0）
df['is_win'] = (df['着順'] == 1).astype(int)

features = ['jockey_id', '単勝', '人気']

# 説明変数と目的変数の分離
X = df[features]
y = df['is_win']
# y = df['着順'].astype(int)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# モデルの学習
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 予測と評価
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# 結果の表示
print(f'全体の正解率: {accuracy:.3f}')
print(f'1着的中率: {precision:.3f}')
print('\n混同行列:')
print(conf_matrix)

# 予測結果をデータフレームに追加
results_df = X_test.copy()
results_df['実際の着順'] = y_test
results_df['予測'] = y_pred

# 日時をファイル名に追加
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'prediction_results_{timestamp}.csv'

# 結果を保存
results_df.to_csv(f'results/{filename}', index=False)

# ログファイルに記録
with open('results/result_log.md', 'a') as f:
    f.write(f'\n## timestamp {timestamp}\n')
    f.write(f'- accuracy: {accuracy:.3f}\n')
    f.write(f'- precision: {precision:.3f}\n')
    f.write(f'- filename: {filename}\n')

        # 混同行列の追加
    f.write('\n### confusion matrix\n')
    f.write('```\n')
    f.write(f'{conf_matrix}\n')
    f.write('```\n')
    
    # 混同行列の説明
    f.write('\n- row : Actual (0: others, 1: 1st position)\n')
    f.write('- column : Prediction (0: others, 1: 1st position)\n')

#結果はresult_log.mdに記録