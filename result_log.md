# 試行結果

| 使用モデル | 使用データ | 予想データ | 正解率 | 備考欄 |
|------------|-------------|-------------|--------|--------|
|  random forest   |   horse_id, jockey_id, 単勝, 人気          |    着順         |    0.11    |        |
|       random forest      |    jockey_id, 単勝, 人気         |        着順      |    0.11    |        |
|    random forest        |     jockey_id, 単勝, 人気        |      一着かどうか       |   0.91     |        |

