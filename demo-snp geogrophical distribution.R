# 加载安装包

library(openxlsx)
library(ggplot2)
library(maps)

#  加载SNP数据

snp <- read.csv('SNP.csv',sep = ',',header = T)
colnames(snp)
table(snp$SUBPOPULATION)  #查看亚型频数

#  只要籼稻和粳稻亚型

attach(snp)
snpXG <- snp[SUBPOPULATION == 'ind1A' | SUBPOPULATION == 'ind1B' 
               |SUBPOPULATION == 'ind2' | SUBPOPULATION == 'ind3'
               |SUBPOPULATION == 'indx' |SUBPOPULATION == 'japx'
               |SUBPOPULATION == 'temp' |SUBPOPULATION == 'subtrop'
               |SUBPOPULATION == 'trop', ]
detach(snp)

head(snpXG)

#  导入品种的产地信息
#保存csv时选用逗号隔开。

country <- read.csv('3kcountry.csv',sep = ',',header = T)  
head(country)

snpc <- merge(snpXG,country,by.x = 'ACCESSION',
              by.y = 'Genetic_ACCESSION',all.x = T,sort = F)  
head(snpc)

write.csv(snpc, file = "d:/bioinformatics/R/SNP_1.csv")

#  修改文件后不能在打开状态下写入
#  导入经纬度信息

lon_lat <- read.csv('geodis.csv',sep = ',',header = T)
head(lon_lat)
snpd <- merge(snpc,lon_lat,by.x = "Country",
              by.y = "Country",all.x = T,sort = F)


#  为了方便后续操作及数据管理建议写入文档

write.csv(snpd, file = "d:/bioinformatics/R/SNP_INFOR.csv")
loc <- read.csv(file = "d:/bioinformatics/R/SNP_INFOR.csv",sep = ","
                ,header = TRUE)
#  去除缺失值

loc <- na.omit(loc)

head(loc)

info <- paste(loc$Latitude,loc$Longitude,loc$SNP, sep = "_",na.rm = TRUE)
head(info)
freq <- as.data.frame(table(info), stringsAsFactors = FALSE)
head(freq)

info_split <- do.call(rbind, strsplit(freq$info, split = "_", fixed = TRUE))

freq$lat <- as.integer(info_split[,1])
freq$lat
freq$lon <- as.integer(info_split[,2])
freq$type <- info_split[,3]




map('world', fill = TRUE, col = 'grey',resolution=1,
    mar = c(0.01, 0.01, par("mar")[2], 0.01),border=NA,
    xlim = c(-180,180),ylim = c(-90,90))


map.axes(cex.axis=0.8) #给毛坯房加上axis,似乎只是变大字体而已


for ( i in 1:nrow(freq)){
  pt_color <- ""
  pt_size <- 1
  if ( freq[i,'type'] == "C"){ #将G指定为蓝色
    pt_color <- "orange"
  } else{
    pt_color <- "blue" #将T指定为橘色
  }
  
  if (freq[i,'Freq'] < 50){ #对频数进行分级
    pt_size <- 1
  } else if (freq[i,'Freq'] < 50){
    pt_size <- 1.5
  } else if(freq[i,'Freq'] < 150){
    pt_size <- 2
  } else{
    pt_size <- 2.5
  }
  points(x=freq[i,'lon'], y =freq[i,'lat'] ,  #使用内置的点图进行画图，位置为经纬度，大小取决于频数，不同的变异类型填充不同的颜色。
         cex = pt_size, col=pt_color, pch = 19)
  
  
}

