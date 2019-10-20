# 实验数据集整理

## 常用数据集
CIFAR
【A. Torralba and R. Fergus and W. T. Freeman, 80 Million Tiny Images: a Large Database for NonParametric Object and Scene Recognition, IEEE PAMI, 2008】
ImageNet
【Deng, Jia, Dong, Wei, Socher, Richard, jia Li, Li, Li, Kai, and Fei-fei, Li. Imagenet: A large-scale hierarchical image database. In In CVPR, 2009.】
MNIST
【LeCun, Y., Jackel, L., Bottou, L., Brunot, A., Cortes, C., Denker, J., Drucker,H., Guyon, I., M¨uller, U., S¨ackinger, E., Simard, P., Vapnik, V.: Comparison of learning algorithms for handwritten digit recognition. In: Int’l Conf. on Art. Neu.Net. pp. 53–60 (1995)】
Traffic sign
【STALLKAMP, J., SCHLIPSING, M., SALMEN, J., AND IGEL, C.Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. Neural Networks, 0 (2012),.】
DREBIN Android恶意软件数据集
（D. Arp, M. Spreitzenbarth, M. Hubner, H. Gascon, and K. Rieck. DREBIN: Effective and Explainable Detection of Android Malware in Your Pocket. In Proceedings of the 2014 Network and Distributed System Security Symposium(NDSS), 2014.）


## 安全领域整理的相关数据集

## 恶意软件/恶意文件/恶意邮件/ Cerber 勒索

 - A-1:恶意软件邮件地址：https://github.com/WSTNPHX/scripts-and-tools/blob/master/malware-email-addresses.txt（下载）
 - A-2: 恶意软件（exe文件）下载： http://vxvault.net/ViriList.php（爬虫）
 - A-3: 恶意软件域名列表：http://dns-bh.sagadc.org/dynamic_dns.txt（下载）
 - A-4: theZoo: theZoo是一个恶意软件分析的开源项目，目前由Shahak Shalev维护。该项目里面包含了几乎所有版本的恶意软件。GitHub - ytisf/theZoo: A repository of LIVE malwares for your own joy and pleasure
 - A-5:OpenMalware: http://www.offensivecomputing.net/ (DannyQuis发起的开源恶意软件搜索平台。)
 - A-6:Contagio: http://contagiodump.blogspot.com/ (Contagio是恶意软件收集平台，主要收集最新的恶意软件样本，威胁以及恶意软件分析)
 - A-7: MalShare: http://malshare.com/ (MalShare 旨在建立一个公共的恶意软件数据库，同时也提供一些工具)
 - A-8:MalwareBlacklist: http://www.malwareblacklist.com/showMDL.php (malwareblacklist收录了恶意软件的URL和样本)
 - A-9：VirusShare： VirusShare.com（这个网站提供恶意样本，恶意软件事件库，分析，和病毒样本的代码）
### Github上面的公开数据集：
 - A-10: https://github.com/ashishb/android-malware：安卓的恶意代码样本数据。
 - A-11 :https://github.com/RamadhanAmizudin/malware：各种恶意软件的源代码
 - A-12: https://github.com/Te-k/malware-classification：用CSV存储的恶意代码
### Kaggle关于Malware detection的挑战赛（都有对应数据集可供下载）：
 - A-13: https://www.kaggle.com/nsaravana/malware-detection 
 - A-14: https://www.kaggle.com/c/malware-classification
 - A-15: https://www.kaggle.com/c/ml-fall2016-android-malware
 - A-16: https://www.kaggle.com/c/adcg-2016-malware-anomaly-detection-w-peinfo

## 恶意域名url与钓鱼网站
 - B-1：Alexa收录知名网站域名：http://alexa.chinaz.com/(爬虫)
 - B-2：恶意网站以及对应的ip：http://cybercrime-tracker.net/all.php （下载）
 - B-3:  ZeuS Tracker提供IP和域名黑名单: https://zeustracker.abuse.ch/blocklist.php（下载）
 - B-4: Malware domain list数据库: http://www.malwaredomainlist.com/ 
 - B-5: 用流行僵尸程序样本生成恶意域名，比如Conficker, Strom, Kraken

## 网络流量信息
 - C-1：CISC 2010   
 - C-2:  Kdd99
 - C-3：CAIDA数据集http://www.caida.org/data（下载）
 - C-4：UNIBS数据集www.ing.unibs.it/ntw/tools/traces/index.php  （下载）
 - C-5：WIDE数据集http://mawi.wide.ad.jp/mawi （下载）
 - C-6：WITS数据集www.wand.net.nz/wits (只能通过IPV6主机访问) 

## 自动驾驶
 - D-1:交通标志牌：http://btsd.ethz.ch/shareddata/（下载）
 - D-2:德国交通标志牌：http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset（下载）
 - D-3: KITTI ：很知名的数据集 数据集链接 http://www.cvlibs.net/datasets/kitti/ 
 - D-4：Oxford RobotCar ：对牛津的一部分连续的道路进行了上百次数据采集，收集到了多种天气、行人和交通情况下的数据，也有建筑和道路施工时的数据。1000小时以上。 数据集链接 http://robotcar-dataset.robots.ox.ac.uk/datasets/ 
 - D-5：Cityscape ：一个面向城市道路街景语义理解的数据集 
数据集链接 https://www.cityscapes-dataset.com/ 
 - D-6：Comma.ai ：geohot创办的comma.ai的数据集，80G左右 
数据集链接 https://github.com/commaai/research 
 - D-7：Udacity 
数据集链接 https://github.com/udacity/self-driving-car/tree/master/datasets  
也有模拟器
 - D-8：BDDV 
Berkeley的大规模自动驾驶视频数据集 
数据集链接 http://data-bdd.berkeley.edu/#video 
 - D-9：GTA 
grand theft auto游戏 
网站链接 http://www.rockstargames.com/grandtheftauto/ 
 - D-10：TORCS 
The Open Racing Car Simulator  
数据集链接 http://torcs.sourceforge.net/ 
 - D-11：CARLA 
Intel和丰田共同推出的一个开源的模拟器 
数据集链接 http://carla.org/  
代码链接 https://github.com/carla-simulator/carla 

## 噪音与隐藏指令（攻击用语音指令数据集训练的模型）
 - F-1: 谷歌语音命令数据集地址：
http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz (下载)
