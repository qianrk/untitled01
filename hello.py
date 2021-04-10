import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql import *
from pyspark.sql.types import *

from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler,IndexToString
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

spark = SparkSession.builder \
        .appName("test") \
        .enableHiveSupport() \
        .getOrCreate()
sc = spark.sparkContext

sample_dataset = [
    (0, "unknow", 37, 10, "no", 3, 18, 7, 4),
    (0, "female", 27, 4, "no", 4, 14, 6, 4),
    (0, "female", 32, 15, "yes", 1, 12, 1, 4),
    (0, None, 57, 15, "yes", 5, 18, 6, 5),
    (0, "null", 22, 0.75, "no", 2, 17, 6, 3),
    (0, "female", 32, 1.5, "no", 2, 17, 5, 5),
    (0, "female", 22, 0.75, "no", 2, 12, 1, 3),
    (0, "", 57, 15, "yes", 2, 14, 4, 4),
    (0, "female", 32, 15, "yes", 4, 16, 1, 2),
    (0, "male", 22, 1.5, "no", 4, 14, 4, 5),
    (0, "male", 37, 15, "yes", 2, 20, 7, 2),
    (0, "male", 27, 4, "yes", 4, 18, 6, 4),
    (0, "male", 47, 15, "yes", 5, 17, 6, 4),
    (0, "female", 22, 1.5, "no", 2, 17, 5, 4),
    (0, "female", 27, 4, "no", 4, 14, 5, 4),
    (0, "female", 37, 15, "yes", 1, 17, 5, 5),
    (0, "female", 37, 15, "yes", 2, 18, 4, 3),
    (0, "female", 22, 0.75, "no", 3, 16, 5, 4),
    (0, "female", 22, 1.5, "no", 2, 16, 5, 5),
    (0, "female", 27, 10, "yes", 2, 14, 1, 5),
    (1, "female", 32, 15, "yes", 3, 14, 3, 2),
    (1, "female", 27, 7, "yes", 4, 16, 1, 2),
    (1, "male", 42, 15, "yes", 3, 18, 6, 2),
    (1, "female", 42, 15, "yes", 2, 14, 3, 2),
    (1, "male", 27, 7, "yes", 2, 17, 5, 4),
    (1, "male", 32, 10, "yes1", 4, 14, 4, 3),
    (1, "male", 47, 15, "yes1", 3, 16, 4, 2),
    (0, "male", 37, 4, "yes1", 2, 20, 6, 4)
]
columns = ["labels", "gender", "age", "label", "children", "religiousness", "education", "occupation", "rating"]

pdf = pd.DataFrame(sample_dataset, columns=columns)
data = spark.createDataFrame(pdf)
data.show(5)

#缺失计数
#data.agg(*[F.sum(F.when(df[c].isin(np.nan,"null"), 1).when(F.isnull(df[c]), 1).otherwise(0)).alias(c) for c in df.columns])
cleanStringUDF = F.udf(lambda x : "male" if x in ("null","unknow","","None") or x == None else x)
#splitCalUDF = F.udf(lambda x : float(x.split("*")[0])*float(x.split("*")[1]), returnType=StringType())
#缺失处理
data = data.withColumn("gender",cleanStringUDF("gender"))
#                  .withColumn("religiousness",splitCalUDF("religiousness"))
#类型处理
feature1_list = ['age','label','religiousness','education','occupation','rating']
feature2_list = ['gender','children']
for c in feature1_list:
    data = data.withColumn(c, data[c].cast(DoubleType()))

indexers = [StringIndexer(inputCol=c, outputCol='{0}_indexed'.format(c),handleInvalid='error') for c in feature2_list]
encoders = [OneHotEncoder(dropLast=True,inputCol=indexer.getOutputCol(),
            outputCol="{0}_encoded".format(indexer.getOutputCol())) for indexer in indexers]
assembler = VectorAssembler(inputCols=feature1_list+[encoder.getOutputCol() for encoder in encoders],outputCol="features")
feature_pipeline = Pipeline(stages=indexers + encoders + [assembler])
feature_model = feature_pipeline.fit(data)

#index y
#分训练和测试
#labelIndexer = StringIndexer(inputCol = "affairs", outputCol = "indexedLabel").fit(df)
#data = labelIndexer.transform(df)
Data = feature_model.transform(data)
print("所有的特征名称:{0}".format(Data.columns))
train_data, test_data = Data.randomSplit([0.7, 0.3],seed=1994)
print("训练样本数:%d\n测试样本数:%d"%(train_data.count(),test_data.count()))

#随机森林
rf = RandomForestClassifier(numTrees=100, featuresCol='features', labelCol="labels", seed=7).fit(train_data)
Predictions = rf.transform(test_data)


#f1 = MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='affairs',metricName='f1',metricLabel=1).evaluate(lrPredictions)
#accuracy = MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='affairs',metricName='accuracy',metricLabel=1).evaluate(lrPredictions)
#weightedPrecision = MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='affairs',metricName='weightedPrecision',metricLabel=1).evaluate(lrPredictions)
#weightedRecall = MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='affairs',metricName='weightedRecall',metricLabel=1).evaluate(lrPredictions)

#分类报告
report = Predictions.select("prediction","labels","features","probability").toPandas()
print(classification_report(y_true=report['labels'],y_pred=report['prediction']))
# 使用混淆矩阵评估模型性能[[TP,FN],[TN,FP]]
TP = Predictions.filter(Predictions['prediction'] == 1).filter(Predictions['labels'] == 1).count()
FN = Predictions.filter(Predictions['prediction'] == 0).filter(Predictions['labels'] == 1).count()
TN = Predictions.filter(Predictions['prediction'] == 0).filter(Predictions['labels'] == 0).count()
FP = Predictions.filter(Predictions['prediction'] == 1).filter(Predictions['labels'] == 0).count()
# 计算查准率 TP/（TP+FP）
precision = TP/(TP+FP)
# 计算召回率 TP/（TP+FN）
recall = TP/(TP+FN)
# 计算F1值 （TP+TN)/(TP+TN+FP+FN)
F1 =(2 * precision * recall)/(precision + recall)
#计算accuracy
accuracy = (TP+TN)/(TP+TN+FP+FN)
auc = BinaryClassificationEvaluator(labelCol='labels').evaluate(Predictions)
print(" f1:%1.2f\n accuracy%1.2f\n Precision:%1.2f\n Recall:%1.2f\n auc:%1.2f "%(F1,accuracy,precision,recall,auc))