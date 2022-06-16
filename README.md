实验运行环境为python3.9，解释器为aconda
1.二分类器的训练
    原始特征矩阵为matrix.csv，用PCA.py进行降维处理得到降维后矩阵matrix_90.npy，之后分别调用Models中的5种二分类器进行训练，输出评价结果；
2.多分类器的训练
    原始数据集为TCR_all.csv，进行特征提取之后将结果保存在matrix_multi.npy中，用PCA_multi.py进行降维处理，所得结果保存在matrix_multi_90.npy中，之后结合OVO和OVR两种方法进行训练和预测；
3.基于迁移学习的多分类器
    原始数据集为VDJdb+McPAS.xlsx,数据预处理后的结果保存在VDJdb+McPAS_after.csv中，transfer.py是模型生成和训练的过程。
    实验前需要安装transformer模块
    !pip install transformers