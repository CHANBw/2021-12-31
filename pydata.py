# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:46:28 2021

@author: genom
"""

'''
3 python环境搭建
Anaconda3下载地址
windows：
https://repo.anaconda.com/archive/Anaconda3-2020.11-Windows-x86_64.exe

'''

'''
6 ipython

Tab ： 自动补齐
! ：调用系统命令，!shell命令
?： 查看对象属性信息
??： 查看源代码
output=!cmd args ：执行 cmd 并赋值
ipython –pylab启动可集成绘图窗口

魔术操作符
%run script.py ：   执行 script.py
%time statement  ：   测试 statement 的执行时间
%hist ：   显示历史命令，很多可选参数，可用于制作命令说明
%hist -f filename.py ：保存ipython历史记录
%quickref  ：  显示 IPython 快速参考
%magic  ：  显示所有魔术命令的详细文档
%debug  ：  从最新的异常跟踪的底部进入交互式调试器
%pdb  ：  在异常发生后自动进入调试器
%reset ：   删除 interactive 命名空间中的全部变量
%prun statement  ：   通过 cProfile 执行对 statement 的逐行性能分析
%timeit statement  ：  多次测试 statement 的执行时间并计算平均值
%dhist ：   显示历史目录，用 cd -n 可以直接跳转；
%who、%who_ls、%whos    显示 interactive 命名空间中定义的变量，信息级别/冗余度可变
%xdel variable   ： 删除 variable，并尝试清除其在 IPython 中的对象上的一切引用
%bookmark ：   使用 IPython 的目录书签系统
%cd direcrory  ：  切换工作目录
%pwd ：  返回当前工作目录（字符串形式）
%env  ： 返回当前系统变量（以字典形式）

'''

'''
7 jupyter
'''
import seaborn as sns
tips = sns.load_dataset("tips")
sns.catplot(x="day", y="total_bill", data=tips);
#取消抖动
sns.catplot(x="day", y="total_bill", jitter=False, data=tips); 


'''
8 基本语法
'''
a=1
b=2
c=a+b
x=[1,2,3,4,5,6,7]

#python循环
for i in [1,2,3,4,5,6]:
    i=i+1
    print (i)
    
'''
9 加载模块
'''
import os
import pandas as pd
from pandas import *


#特殊符号

'''
+ - * \
#
''
""
\t
\n
()
[]
{}

'''
import numpy as np
x=np.random.randint(1,100,size=100000)
len(x)
sum(x)

'''
12 面向对象编程
'''

x=[1,2,3,4,5]
type(x)
dir(x)
'''
13 NumPy介绍
'''
import numpy as np
x=np.random.randint(1,100,size=100)
type(x)
x.dtype
dir(x)
#方法
x.sum()
x.mean()
x.sort()
import pandas as pd
ps=pd.Series([1,3,2,4],index=list('abcd'))
df=pd.DataFrame(np.random.randn(12).reshape(4,3),index=['A','B','C','D'],columns=['one','two','three'])


#比较传统数组与多维数据运行效率
my_arr = np.arange(1000000)
my_list = list(range(1000000))
%time for _ in range(10): my_arr *2
%time for _ in range(10): [x * 2 for x in my_list]

import numpy as np
dir(np)


#如何使用numpy
#直接将python数组或元组转换成ndarray
x=[1,2,3,4,5]
#type函数查看数据类型
type(x)
y=np.array(x)
type(y)
x=(1,2,3,4,5)
type(x)
#将python元组转换为ndarray
y=np.array(x)
type(y)


#比较传统列表数组与ndarray的区别
a=[1,2,3,4,5]
b=np.array([1,2,3,4,5])
a+1
b+1
a*10
b*10

'''
14 NumPy案例
numpy有很多函数，其中使用numpy比较多的功能是利用其生产数字，比如随机数，正太分布，等差数列等。
'''

# 使用array创建数组：
arr = np.array([1,2,3])
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])

# 使用arange创建数组
arr = np.arange(0,10,1)
#创建1-12的3行4列的二维数组

arr = np.arange(12).reshape(3,4)
# random生成随机数
#生成随机数种子
np.random.seed(1234)
#randn 产生正太分布样本
np.random.randn(1000)
#随机生成整数数据集
np.random.randint(size=1000,low=1,high=1000)
 

#数学计算函数
# 使用array创建数组
#x是一个包含1000个随机正整数的集合，取值范围从1~1000。
x=np.random.randint(size=1000,low=1,high=1000)
#输出x
x
#进行集合的求和，平均值，方差，标准差等计算
np.sum(x)
np.mean(x)
np.var(x)
np.std(x)
np.min(x)
np.max(x)
np.argmin(x)
np.argmax(x)
np.cumsum(x)
np.cumprod(x)

#计算机性能测试
#生成100万个数值进行计算
x=np.random.randint(size=1000000,low=1,high=1000)
np.sum(x)
#生成1000万个数值进行计算
x=np.random.randint(size=10000000,low=1,high=1000)
np.sum(x)
#生成1亿个数值进行计算
x=np.random.randint(size=100000000,low=1,high=1000)
np.sum(x)
#如果觉得自己计算机性能不错，可以计算挑战

'''
15 numpy索引
'''
#生成一个3行两列的数组
x=np.arange(6).reshape(3,2)
x
#取第一列
x[:,0]
#取第二行
x[1,:]
#去第一列第二行的值
x[0,1]
#选取多行
x[[0,2],:]
#改变维度
x.reshape(2,3)
#排序
x.sort()

'''
16 数组组合
'''
#创建一个3行2列的数组
a=np.arange(6).reshape(3,2)
#创建一个3行3列的数组
b=np.arange(9).reshape(3,3)
#创建一个2行3列数组
c=np.arange(6).reshape(2,3)
#数组水平组合,水平组合行数需要一致
np.hstack((a,b)) #别忘了中间的括号
np.concatenate((a,b),axis=1)
np.append(a,b,axis=1)
#垂直组合，与水平组合类似，但是要求列数一致
np.vstack((b,c)) 
np.concatenate((b,c),axis=0)
np.append(b,c,axis=0)

 
'''
17 生成扑克牌
'''
#生成A，2-10，J，Q，K
x=np.append("A",np.arange(2,11))
x=np.append(x,["J","Q","K"])

#扩大四倍
x= np.tile(x,4)
#生成4种52张花色
y=np.repeat(("spades","heart","cube","diamond"),13)
#合并两个数组
poker=np.char.add(y,x)
#添加 Joker
poker=np.append(poker,["black Joker","red Joker"])

np.repeat([1,2,3,4,5],repeats=[1,2,3,4,5])
np.tile(A=[1,2,3,4,5],reps=2)

''' 
18 pandas Series

'''
#pandas
import pandas as pd

#python普通列表: 
x=[1,2,3,4,5]
#numpy多维数组：
y=np.array(x)
#pandas Series类：
z=pd.Series(x)
type(z)
dir(z)
z.values
z.index

#生成Series，数据自带数值部分和索引部分
x=pd.Series([1,2,3,4,5,6],
            index=['a','b','c','d','e','f'])

#ndarray转换为Series
x=np.random.randn(5)
x=pd.Series(x)

#将python字典转为换Series
dict={'Beijing':2000,'Guangdong':12000,
      'Jiangsu':9000,'Shandong':8500}
x=pd.Series(dict)

'''
19  DataFrame
'''
s=pd.Series([1,3,2,4],index=list('abcd'))
x=pd.DataFrame(np.random.randn(12).reshape(4,3),index=['A','B','C','D'],columns=['one','two','three'])
a=x['one']
type(a)
A=x.iloc[0,]
A
type(A)

ID=[1,2,3,4]
name=['Tom','Katty','Johon','Brown']
sex=['man','women','men','men']
df=pd.DataFrame({'ID':ID,'Name':name,'Sex':sex})

#随机生成DataFrame
df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])

#python字典可以直接转换为DataFrame
data = {
'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
 'year': [2000, 2001, 2002, 2001, 2002, 2003],
 'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
df = pd.DataFrame(data)
type(df)
dir(df)
df.head()
df.dtypes
df.info()
df.shape
df.index
df.columns
df.describe()
df.values

'''
20 目录结构
windows系统目录结构：“C:/Users/xxx/Desktop”
Linux系统目录结构：“/home/xxx/”
Mac系统目录结构：“/User/xxx/Desktop/”
'''

#设置工作目录
import os
os.getcwd()
os.chdir('Desktop/pydata/datasets') #换成你自己的工作目录         
os.listdir()
'''
21 文件格式
csv
tsv
xlsx
json
'''

'''
22 读写文件
'''
#读取文件
import pandas as pd
mtcars=pd.read_csv('mtcars.csv',sep=",",header=0,index_col=0)
mtcars.head()
mtcars.shape
mtcars.index
mtcars.columns
mtcars.dtypes
mtcars.info()
dir(mtcars)
type(mtcars)
str(mtcars)

'''
23 读写Excel文件
'''

#读写Excel文件
pd.read_excel('vlookup.xlsx',sheet_name=0)
df=pd.read_excel('vlookup.xlsx',sheet_name=0,index_col=0)
df.head()
df.to_csv('test.csv',header=True,index=True)

'''
24 pandas索引
'''
#选取数据
mtcars=pd.read_csv('mtcars.csv',sep=",",header=0,index_col=0)
mtcars.head()
mtcars.shape
mtcars.index
mtcars.columns

mtcars.cyl
mtcars.cyl.size
mtcars['mpg']
mtcars[['mpg']]
mtcars[['mpg','cyl']]
mtcars[0:5]
#标签索引
mtcars.loc[:,['cyl']]
mtcars.loc[['Fiat 128','Valiant']]
#数字索引
mtcars.iloc[ 0:5 ,:  ]
mtcars.iloc[1]
mtcars.loc[:,['disp','hp']]

'''
25 筛选奇偶项
'''
#奇数行
mtcars.iloc[np.arange(0,32,2)]
#偶数行
mtcars.iloc[np.arange(1,32,2)]
#删除行列
mtcars.drop(columns=['cyl','mpg'])
mtcars.drop(index=['Valiant'])

#负数索引
mtcars.iloc[-1,:]
mtcars.iloc[:,-1]

'''
26 逻辑值索引
'''
x=mtcars.iloc[0:3,0:3]
x.loc[:,[True,False,False]]
x.loc[[True,False,True],:]
x.iloc[[True,False,False],:]
#筛选奇数项
logic=np.repeat([True,False],repeats=16)
logic
mtcars[logic]
logic=np.tile([True,False],reps=16)
logic
mtcars.loc[logic,:]

'''
27 筛选数据
'''
mtcars[mtcars.loc[:,'mpg']>=25]
mtcars.loc[mtcars.cyl==4,:]
mtcars[(mtcars.mpg >=25) & (mtcars.cyl==4)]
mtcars[(mtcars.mpg >=25) |(mtcars.cyl==4)]
#根据字符串匹配进行筛选
iris=pd.read_csv('iris.csv')
iris.head()
iris.query('Species=="setosa"')


'''
28 利用python实现vlookup
'''
gene121=pd.read_csv('121genes.csv',squeeze=True)
gene121
#去除重复项
gene121.duplicated()
#gene121[gene121.duplicated()]
gene121.unique()
gene121.nunique()
geneid=gene121.unique()
gene200=pd.read_csv('200genes.csv',index_col=0)
#gene200=pd.read_csv('200genes.csv',index_col=0)
#gene200.reset_index(drop='gene')
gene200.head()
gene200.index
#重新Index，实现vlookup功能
gene93=gene200.reindex(index=geneid)
#去掉缺失值
gene86=gene93.dropna()
#保存最终结果
gene86.to_csv("gene86.csv")


'''
29 修改数据
'''
mt=mtcars
mtcars.iloc[0,0]=12
mtcars.iloc[:,1]=10
mtcars.cyl.replace(4,'four')
mtcars.cyl.replace([4,6,8],['four','six','eight'])

#修改行名
mtcars.rename(index={'Fiat 128':'fiat128'})
mtcars.rename(index={'Fiat 128':'fiat128'},inplace=True)
#修改列名
#mtcars.reset_index()
mtcars.rename(columns={'cyl':'cylinder','mpg':'MPG'})
mtcars.rename({'cyl':'CYL'},axis=1)
mtcars.rename({'Fiat 128':'fiat128'},axis=0)


'''
30 排序
'''
#行名排序
mtcars.sort_index()
#列名排序
mtcars.sort_index(axis=1)
mtcars.sort_index(ascending=False)
#值排序
mtcars.sort_values(by='cyl')  
mtcars.sort_values(by='cyl',ascending=False)  
mtcars.sort_values(by=['cyl','mpg'],ascending=False)
mtcars.sort_values(by=['cyl','mpg'],ascending=[False,True])
mtcars[mtcars.cyl.isin([4,6])]


'''
31 随机抽样
'''
np.random.seed(1234)
mtcars.sample(n=25)
mtcars.sample(n=25,replace=True)
mtcars.sample(frac=0.2)
mtcars.sample(frac=0.2,axis=1)

'''
32 斗地主
'''
#生成A，2-10，J，Q，K
x=np.append("A",np.arange(2,11))
x=np.append(x,["J","Q","K"])

#扩大四倍
x= np.tile(x,4)
#生成4种52张花色
y=np.repeat(("spades","heart","cube","diamond"),13)
#合并两个数组
poker=np.char.add(y,x)
#添加 Joker
poker=np.append(poker,["black Joker","red Joker"])
poker=pd.Series(poker)
shutter=poker.sample(54)
shutter.values
shutter.values[np.arange(0,51,3)]
first=shutter.values[np.arange(0,51,3)]
second=shutter.values[np.arange(1,5,3)]
third=shutter.values[np.arange(2,51,3)]
dipai=shutter.values[[51,52,53]]

'''
33 获取帮助
1、help函数
help()
?
2、官方文档
3、搜索引擎
'''
help(np.arange)
np.arange(3)
np.arange(3.0) #结果为浮点数
np.arange(3,7)
np.arange(3,7,2)

'''
34 计算基因长度分布
'''
gff=pd.read_csv("H37Rv.gff",sep="\t",skiprows=7,header=None)
gff.size
gff.columns
gff.index
#筛选第三列为gene的行
gene=gff[gff[2]=='gene']
#gene=gff[gff.iloc[:,2]=="gene"]
gene.size
#计算基因长度
gene_length=np.abs(gene.iloc[:,4] - gene.iloc[:,3])
gene_length.describe()
gene_length.describe().round(2)
#绘制基因长度分布图
gene_length.hist()
gene_length.hist(bins=80)

'''
35 修改行列
'''
#删除行列
mtcars.drop(index='Volvo 142E')
mtcars.drop(columns=['mpg'])
mtcars.drop(columns=['mpg','cyl'])
mtcars.drop(index="Volvo 142E",columns="cyl")

#增加行列
logic=np.where(mtcars.mpg>20,"T","F")
mtcars['logic']=""
mtcars.logic=logic

[np.log(x) for x in mtcars['cyl']]
log_cyl=[np.log(x) for x in mtcars['cyl']]
np.round(log_cyl,2)

#按行和列求和
gene=pd.read_csv('heatmap.csv',index_col=0)
gene.head()
gene.index
gene.columns
total1=gene.apply(axis=0,func=sum)
gene.loc['Total',:]=tota1
total2=gene.apply(axis=1,func=sum)
gene.loc[:,'Total']=total2
gene


'''
36 处理缺失值
'''
sleep=pd.read_csv("sleep.csv",index_col=0,na_values=['NA'])
sleep.head()
#判断缺失值
sleep.isna()
sleep.isnull()
#删除缺失值
sleep.dropna(axis=0)
sleep.dropna(axis=1)
sleep.dropna(axis=0,how='all')
#填充缺失值
sleep.fillna(axis=0,method='ffill')
sleep.fillna(axis=1,method='bfill')
sleep.fillna(value=sleep.mean())


'''
37 分组计算
'''
x=pd.read_csv('ToothGrowth.csv',index_col=0)
x
x.columns
x.supp
#计算频数
x.supp.value_counts()
x.dose.value_counts()

# 将dose列的类型转换为字符串类型。
x['dose']=x['dose'].astype('str')

#二位列联表
pd.crosstab(x.dose,x.supp)
pd.crosstab(x.supp,x.dose)

#分组统计
x.groupby(by=x.supp).sum()
x.groupby(by=x.supp).mean()
x.groupby(by='supp')['len'].mean()

x.groupby(by=['supp','dose']).mean()
x.groupby(by=['supp','dose']).min()
x.groupby(by=['supp','dose']).max()

'''
38 数据透视表
'''
x=pd.read_excel("2015年度中国城市GDP排名.xlsx",index_col=0)
x.head()
x.columns
#计算频数
x.Province.value_counts()
#添加平均收入项
x['Average']=x.Income / x.People
x.Average=round(x.Average,ndigits=2)

#探索数据
x.sort_index(ascending=False)
x.sort_index(ascending=False,axis=1)
x.sort_values('Income')
x.sort_values('Income',ascending=False)
#分组统计
x.pivot_table(index='Province',aggfunc=sum)
x.pivot_table(values='Income',index='Province',aggfunc=sum)
x.pivot_table(values='Income',index='Province',aggfunc=np.mean)
x.pivot_table(values='Income',index='Province',aggfunc=np.size)
x.pivot_table(values='Income',columns='Province',aggfunc=sum)

x.pivot_table(index='Province',aggfunc=max)
x.pivot_table(index='Province',values='City',aggfunc=np.size)
x.pivot_table(index='Province',values=['City','Income'],aggfunc=max)
x.pivot_table(index=['Province','City'],values=['Income'],aggfunc=np.mean)

'''
39 计算相关性
'''
x=pd.read_csv('state.x77.csv',sep=",",header=0,index_col=0)
x
x.shape
x.index
x.columns
x.corr()   # 计算pearson相关系数
x.corr('kendall')      # Kendall Tau相关系数
x.corr('spearman')     # spearman秩相关
x['Murder'].corr(x['Income'])
x.corrwith(x['Murder'])
x['Murder'].cov(x['Income'])


