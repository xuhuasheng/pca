# =========================================================
# @purpose: principal components analysis
# @date：   2019/12
# @version: v1.0
# @author： Xu Huasheng
# @github： https://github.com/xuhuasheng/pca
# =========================================================

from sklearn.externals import joblib  
import numpy as np  
import glob  
import os  
import time  


def pca(dataMat, r, autoset_r=False, autoset_rate=0.9): 
    """
    purpose: principal components analysis
    """  
    print("Start to do PCA...") 
    t1 = time.time() 
    meanVal = np.mean(dataMat, axis=0) # 竖着求平均值，数据格式是m×n
    meanRemoved = dataMat - meanVal  
    # normData = meanRemoved / np.std(dataMat) #标准差归一化
    covMat = np.cov(meanRemoved, rowvar=0)   # 求协方差矩阵 n×n维
    eigVals, eigVects = np.linalg.eig(np.mat(covMat)) # 求特征值和特征向量  eigVals: 1xn维, eigVects: n×n维    
    eigValIndex = np.argsort(-eigVals) # 特征值由大到小排序，eigValIndex是特征值索引的重排列 1×n维

    # 自动设置r的取值
    if autoset_r:
        r = autoset_eigNum(eigVals, autoset_rate)
        print("autoset: take top {} of {} features".format(r, meanRemoved.shape[1]))

    r_eigValIndex = eigValIndex[:r] # 选取前r个特征值的索引  1×r维   
    r_eigVect = eigVects[:, r_eigValIndex] # 把前r个特征值对应的特征向量组成投影变换矩阵P  n×r维  
    lowDDataMat = meanRemoved * r_eigVect # 矩阵点乘投影变换矩阵, 得降维后的矩阵: m×r维 公式Y=X*P
    reconMat = (lowDDataMat * r_eigVect.T) + meanVal   # 转换新空间的数据  m×n维
    t2 = time.time()   
    print("PCA takes %f seconds" %(t2-t1))
    joblib.dump(r_eigVect, './pca_args_save/r_eigVect.eig')    
    joblib.dump(meanVal, './pca_args_save/meanVal.mean') # save mean value   
    return lowDDataMat, reconMat


def autoset_eigNum(eigValues, rate=0.99):
    """
    purpose: Automatically set the optimal number of eigen vectors
    """
    eigValues_sorted = sorted(eigValues, reverse=True)
    eigVals_total = eigValues.sum()
    for i in range(1, len(eigValues_sorted)+1):
        eigVals_sum = sum(eigValues_sorted[:i])    # 前i个特征向量求和
        if eigVals_sum / eigVals_total >= rate:
            break
    return i