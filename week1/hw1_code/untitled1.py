# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:04:47 2018

@author: user
"""

#R 語言的基本變數類型分為以下這幾類：

#數值
#numeric
#integer
#complex
#邏輯值（logical）
#文字（character）

###R 語言基本的資料結構大致有五類：
#vector
#factor
#matrix
#data frame
#list


#Python 的基本變數類型分為以下這幾類：
#數值
#float
#int
#complex
#布林值（bool）
#文字（str）


#Python 基本的資料結構大致有三類：
#list
#tuple
#dictionary

print(type(5.52323))
day = 30
day-=3
print(day)














ironmen = [49, 8, 12, 12, 6, 61]
type(ironmen)


bool= True
print(int(bool))
print(type(str(bool)))


import numpy

apple=[1,2,3,4,5]
apple1=numpy.array([[1,2,3,4,5]])
print(apple1)
print(type(apple1))
apple30=apple1*30
print(apple30)










import numpy as np
import pandas as pd

my_2d_array = np.array([[1, 3],
                        [2, 4],
                        [6, 5]
                       ])

my_df = pd.DataFrame(my_2d_array, columns = ["col1", "col2" ])
print(my_df)












import pandas as pd

groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [46, 8, 12, 12, 6, 58]

ironmen_dict = {"groups": groups,
                "ironmen": ironmen
                }

ironmen_df = pd.DataFrame(ironmen_dict)

print(ironmen_df)

print(ironmen_df.iloc[0:1, 1]) # 第一列第二欄：Modern Web 組的鐵人數
print("---")
print(ironmen_df.iloc[0:1,:]) # 第一列：Modern Web 組的組名與鐵人數
print("---")
print(ironmen_df.iloc[:,1]) # 第二欄：各組的鐵人數
print("---")
print(ironmen_df["ironmen"]) # 各組的鐵人數
print("---")
print(ironmen_df.ironmen) # 各組的鐵人數

print(ironmen_df.shape)
print(ironmen_df.describe())
print(ironmen_df.head(3))
print(ironmen_df.tail(3))
print(ironmen_df.index)
print(ironmen_df.columns)


#for 迴圈

ironmen = [49, 8, 12, 12, 6, 61,6,555,22,922]
for ironman in ironmen:
    print(ironman)
print("---")
print(ironman)


# 帶索引值的寫法
for a in list(range(len(ironmen))): # 產生出一組 0 到 5 的 list
    print(ironmen[a])
print("---")
print(a) # 把迴圈的迭代器（iterator）或稱游標（cursor）最後的值印出來看看






# while迴圈


ironmen = [49, 8, 12, 12, 67, 61]
index = 0
while index < len(ironmen):
    print(ironmen[index])
    index += 1
print("---")
print(index) # 把迴圈的迭代器（iterator）或稱游標（cursor）最後的值印出來看看




##range(起始值 , 終止值 , 遞增(減)值)

for num in range(100):
    for i in range(2,num):
        if num % i == 0:
            break
    else:
     print(num, "是一個質數")



my_seq = list(range(1, 11))
print(my_seq)
for index in my_seq:
    if (index % 2 == 0):
        print(index, "是偶數")
    else:
        print(index, "是奇數")
        
        
        
        
        
        
        
        
my_seq = list(range(1,11))
for index in my_seq:
    if (index % 3 == 0):
        print(index, "可以被 3 整除")
    elif (index % 3 ==1):
        print(index, "除以 3 餘數是 1")
    else:
        print(index, "除以 3 餘數是 2")
        
        
        
        
        
        
        
        
        
       
# break 描述
ironmen = [49, 8, 12, 12, 6, 61]
for ironman in ironmen:
    if (ironman < 10):
        break
    else:
        print(ironman)

print("---")
print(ironman) # 把迴圈的迭代器（iterator）或稱游標（cursor）最後的值印出來看看

print("\n") # 空一行方便閱讀

# continue 描述
for ironman in ironmen:
    if (ironman < 10):
        continue
    else:
        print(ironman)

print("---")
print(ironman) # 把迴圈的迭代器（iterator）或稱游標（cursor）最後的值印出來看看
}
        
        
        
        
        
        
