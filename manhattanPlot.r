#clear the workspace for analysis
rm(list = ls())

#install.packages("CMplot")
library(CMplot)

#manhattan plot with CMplot####
manhattan plot with CMplot####
setwd("E:/testTASSEL/")#工作目录
data <- read.table(file = "test.stats.txt", header = T)#read phenotypedata <- merge(data,chr,by = "Chr")#读取文件
#需要删除None的那些行
data <- data[,c(1:4,7)]
colnames(data) <- c("Trait","SNP", "Chromosome", "Position", "p")
data$Trait <-factor(data$Trait)
data$SNP <- factor(data$SNP)
data$Chromosome <- factor(data$Chromosome)
levels(data$Trait)

for (i in 1:length(levels(data$Trait))){
  onetrait <- subset(data, Trait == levels(data$Trait)[i])
  p <- onetrait[,2:5]
  colnames(p) <- c("SNP", "Chromosome", "Position", levels(data$Trait)[i])
  CMplot(p,plot.type = "m",threshold = c(0.01,0.05)/nrow(p),threshold.col=c('grey','black'),
         threshold.lty = c(1,2),threshold.lwd = c(1,1), amplify = T,
         signal.cex = c(1,1), signal.pch = c(20,20),signal.col = c("red","orange"))
  CMplot(p,plot.type="q",conf.int.col=NULL,box=TRUE,file="jpg",memo="",dpi=300,
         ,file.output=TRUE,verbose=TRUE)
}
