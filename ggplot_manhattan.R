#ggplo2绘制曼哈顿图

#读取文件
df <- read.csv("test1.csv")
head(df)
library(ggplot2)

#映射
ggplot(df,aes(x=Chromosome,y=Trait1))+geom_jitter()

#给染色体添加颜色
ggplot(df,aes(x=Chromosome,y=Trait1))+geom_jitter(aes(color=Chromosome))

#去掉右边标题
ggplot(df,aes(x=Chromosome,y=Trait1))+geom_jitter(aes(color=Chromosome))+
  theme(legend.position = "none")

#将染色体转化为因子类型，使染色体按顺序排列
df$Chromosome<-factor(df$Chromosome,levels = c(1:10))  
ggplot(df,aes(x=Chromosome,y=Trait1))+geom_jitter(aes(color=Chromosome))+
  theme(legend.position = "none")

#调整图像纵向长度
ggplot(df,aes(x=Chromosome,y=-log10(Trait1)))+geom_jitter(aes(color=Chromosome))+
  theme(legend.position = "none")+
  scale_y_continuous(expand = c(0,0),limits = c(0,5))

#给染色体命名
ggplot(df,aes(x=Chromosome,y=-log10(Trait1)))+geom_jitter(aes(color=Chromosome))+
  theme(legend.position = "none")+
  scale_y_continuous(expand = c(0,0),limits = c(0,5))+
  scale_x_discrete(labels=paste0("Chr",c(1:10)))

#y轴名
ggplot(df,aes(x=Chromosome,y=-log10(Trait1)))+geom_jitter(aes(color=Chromosome))+
  theme(legend.position = "none")+
  scale_y_continuous(expand = c(0,0),limits = c(0,5))+
  scale_x_discrete(labels=paste0("Chr",c(1:10)))+labs(x=NULL,y="-log10(Pvalue)")

#添加阈值线,x轴美化
ggplot(df,aes(x=Chromosome,y=-log10(Trait1)))+geom_jitter(aes(color=Chromosome))+
  theme(legend.position = "none",axis.text.x = element_text(angle=60,hjust=1))+
  scale_y_continuous(expand = c(0,0),limits = c(0,5))+
  scale_x_discrete(labels=paste0("Chr",c(1:10)))+labs(x=NULL,y="-log10(Pvalue)")+
  geom_hline(yintercept = 4,lty=2,color="red")
