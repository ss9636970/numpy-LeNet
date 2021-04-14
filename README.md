# *HW3

# 1.  train model
LetNet_old.ipynb 和 LetNet.ipynb分別為改進前後的LetNet執行程序 train model 指令皆寫在裡面，模型定義寫在 LeNet_module.py

進入 hw3.ipynb 一開會 import 會用到的 pachage 包括 LeNet_module.py 以及 function.py，務必將這四個檔案放在同目錄底下
再來會讀圖片的路徑，務必按照寫在變數 train_df, test_df, val_df 務必把照片的路徑按照後面的方法參數中放。

接著就是提取特徵以及定義模型參數，最後是訓練模型。

# 2. test model
hw1.ipynb 中標題有 test 的部分是對模型用 test 資料己算準確率，底下是畫模型的 accuracy  curve
