# MachineLearningAction
Machine Learning Practice Using Python3 and Chinese.  
These are my notes. I think these would help me and the other readers to study or review the knowladge of machine learning quickly.  

## Traditional Machine Learning
传统机器学习方面依托《机器学习实战》这本书来练习，将原书代码用中文变量名在python3中重写，以利于造福中国读者和以后快速复习。  
我认为这本书的风格非常适合我这样有算法竞赛基础想快速入门并在实践中提升的学习者。  
这本书共15章，除去第一章是简介，有14章内容分别从分类、回归、聚类、数据处理等方面对传统机器学习做了主要介绍。  
原书代码由python2编写，没有使用高级库和API，是非常基础的实现，对于平衡的了解机器学习原理和感受其效果非常好。  
第二章介绍的K临近算法因为本人之前已经用C++实现过了就没再实现，考虑将C++模板加入这里。  
第三到第十四章我都尽量用中文变量名本地化的表示算法含义，实现模板和效果速览代码。  
其中第六章支持向量机未实现完全，还有一些练习没做，绘图功能未集成进来；第九章回归树中的可视化建树方案还没有集成进来。  
第十五章是介绍mapreduce，不适合在个人电脑上实现，在运算集群上实现的代码不容易移植，且对日后复习原理意义不大，故未加入此代码库。  

## Deep Learning
深度学习方面依托《Hands-on Machine Learning with Scikit-Learn and TensorFlow》这本书的第二部分来学习。  
此书在第一部分传统机器学习方面使用了sklearn,过于高层次和简单不适合入门和温习。  
第二部分相对来说基础一些，介绍了tensorflow的用法，也采用了一些高级API。本人认为这很好的平衡了对原理和工具的了解和效果预览的速度，故选择了这本书的第二部分来入门深度学习。毕竟深度学习已不该像之前那样从线性代数的级别自己实现而浪费时间。  
由于这一部分计算量普遍比较大,而且原作者没有像《机器学习实战》那样提供合适的小样例,所以只把适合入门的快速样例加入进来。  

# Repository Structure
In order to quick review, all the experiment data had been in the folder named data. What's more, the reference book is in the folder named 'doc'  
代码组织上，每份代码可以一键运行以迅速体会效果。对于普通个人电脑(win7/10,低压CPU、4G内存、集显、机械硬盘)来说运行时间或空间复杂度过大的部分会封装成函数以免一键运行。  