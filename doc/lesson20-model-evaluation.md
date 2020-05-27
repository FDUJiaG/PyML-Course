# Model Evaluation

## Metrics for Binary Classification

二元分类可以说是机器学习在实践中最常见的、概念上最简单的应用，然而，即使是这样简单的任务，在评估时也有一些注意事项

在我们深入研究其他指标之前，让我们先来了解一下衡量准确度可能会误导人的方法

通常情况下，准确性并不能很好地衡量预测性能，因为我们所犯的错误数量并不包含我们感兴趣的所有信息

### Imbalanced Dataset

为了说明，我们从数字数据集中创建一个 $9:1$ 的不平衡数据集，将数字 $9$ 与其他 $9$ 个类进行区分

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)
```

我们可以用 `DummyClassifier` 来始终预测多数类（这里 "不是 $9$ 的类"），看一看不携带信息的准确度

```python
import numpy as np
from sklearn.dummy import DummyClassifier

dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)

print("Unique predicted labels: {}".format(np.unique(pred_most_frequent)))
print("Test score: {:.2f}".format(dummy_majority.score(X_test, y_test)))
```

**Output**

```console
Unique predicted labels: [False]
Test score: 0.90
```

我们在没有学到任何东西的情况下，就获得了接近 $90\%$ 的准确率，这可能看起来很惊人，想象一下，有人告诉你模型的准确率是 $90\%$，你可能会认为他们做得非常好，但根据问题的不同，这可能只需要预测一个类就能实现！让我们将其与实际的分类器进行比较

```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print("Test score: {:.2f}".format(tree.score(X_test, y_test)))
```

**Output**

```console
Test score: 0.92
```

根据准确度，`DecisionTreeClassifier` 模型仅仅比常数预测器要好一些，这可以揭示要么我们在使用 `DecisionTreeClassifier` 时出了问题，或者说准确度在这里本身就不是一个好的指标

为了比较，我们再多评估两个分类器，即 `LogisticRegression` 和默认的 `DummyClassifier` （进行随机预测，但产生的类的比例与训练集中的比例相同）

```python
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore", category=Warning)

dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print("dummy score: {:.2f}".format(dummy.score(X_test, y_test)))

logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print("logreg score: {:.2f}".format(logreg.score(X_test, y_test)))
```

**Output**

```console
dummy score: 0.81
logreg score: 0.98
```

dummy 分类器产生随机的输出，所以在准确度的指标来说会产生最差的结果，而 `LogisticRegression` 产生的结果就非常好

但是，即使是随机分类器依旧有 $80\%$ 以上的的准确度，所以判断哪个结果更有帮助就显得很困难，毕竟，在这种不平衡的数据集下，精度是一个不足以量化预测性能的衡量标准，我们将探索其他的衡量标准，以提供更好的模型选择指导，特别是，我们希望有一些度量指标来告诉我们一个模型比做 “最频繁” 的预测 `pred_most_frequent` 或随机预测 `pred_dummy` 好多少，如果我们用一个度量标准来评估我们的模型，它肯定应该能够剔除这些无稽之谈的预测

### Confusion Matrix

使用混淆矩阵是表示二元分类评估结果最全面的方法之一，让我们用 `confusion_matrix` 函数来检查上面 `LogisticRegression` 的预测结果，我们已经将测试集上的预测结果存储在 `pred_logreg` 中

```python
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_test, pred_logreg)
print("Confusion matrix:\n{}".format(confusion))
```

**Output**

```console
Confusion matrix:
[[402   1]
 [  6  41]]
```

`confusion_matrix` 的输出是一个二乘二的数组，其中行对应于真实类，列对应于预测类，对于每次预测，计算该行所给的类中有多少个数据点是该列所给的预测类的数据点，这里以是不是数字 $9$ 来区分，我们以图来演示

```python
mglearn.plots.plot_confusion_matrix_illustration()
```

![Confusion Matrix Digits](figures/l20/l20-Confusion-Matrix-Digits.png)

混淆矩阵的主对角线上的元素（二维数组或矩阵 $A$ 的主对角线是 $A[i,i]$）对应于正确的分类，而其他元素则告诉我们有多少个类别的样本被误归为另一个类别

如果我们定义数字 $9$ 为正类，我们可以把混淆矩阵的元素与假阳性和假阴性的术语联系起来，为了完成图示，我们把属于阳性类的正确分类的样本称为真阳性，而属于阴性类的正确分类的样本称为真阴性，正如我们以前介绍的

- **真正例（TP）** 实际上是正例的数据点被标记为正例
- **假正例（FP）** 实际上是反例的数据点被标记为正例
- **真反例（TN）** 实际上是反例的数据点被标记为反例
- **假反例（FN）** 实际上是正例的数据点被标记为反例

```python
 mglearn.plots.plot_binary_confusion_matrix()
```

![Confusion Matrix 2 Class](figures/l20/l20-Confusion-Matrix-2-Class.png)

显然

$$
TP + FP + FN + TN = Total\ Number\ of\ Samples
$$

#### Evaluation Indicators

##### Accuracy

准确率的计算公式为

$$
ACC=\frac{TP+TN}{TP+TN+FP+FN}
$$

准确率是我们最常见的评价指标，而且很容易理解，就是被预测准确的样本数除以所有的样本数，通常来说，正确率越高，分类器越好

准确率确实是一个很好很直观的评价指标，但是有时候准确率高并不能代表一个算法就好，比如某个地区某天地震的预测，假设我们有一堆的特征作为地震分类的属性，类别只有两个，0 表示不发生地震；1 表示发生地震

一个不加思考的分类器，对每一个测试用例都将类别划分为 0，那那么它就可能达到 99% 的准确率，但真的地震来临时，这个分类器毫无察觉，这个分类带来的损失是巨大的，为什么 99% 的准确率的分类器却不是我们想要的，因为这里数据分布不均衡，类别 1 的数据太少，完全错分类别 1 依然可以达到很高的准确率却忽视了我们关注的东西，在正负样本不平衡的情况下，准确率这个评价指标有很大的缺陷，因此，单纯靠准确率来评价一个算法模型是远远不够科学全面的

##### Error Rate

错误率则与准确率相反，描述被分类器错分的比例，计算公式为

$$
r_{error} = \frac{FP+FN}{TP+TN+FP+FN}
$$

对某一个实例来说，分对与分错是互斥事件，所以

$$
ACC = 1 - r_{error}
$$

##### Precision

精确率定义为

$$
Acc_{(Precision)} = \frac{TP}{TP+FP}
$$

精确率是针对我们 **预测结果** 而言的，它表示的是预测为正的样本中有多少是真正的正样本

##### Recall

召回率是覆盖面的度量，公式为

$$
Acc_{(Recall)} = \frac{TP}{TP+FN}
$$

召回率是针对原来的 **样本** 而言的，它表示的是样本中的正例有多少被预测正确了

##### F-Measure

Precision 和 Recall 指标有时候会出现的矛盾的情况，这样就需要综合考虑他们，最常见的方法就是 F-Measure（又称为 F-Score）

F-Measure 是 Precision 和 Recall 加权调和平均

$$
F = \frac{1}{\frac{1}{k_1+k_2}(k_1\cdot\frac1P+k_2\cdot\frac1R)} 
= \frac{(k_1+k_2)P\cdot R}{k_1\cdot R + k_2\cdot P}
$$

其中

$$
k_1 + k_2 = 1, \qquad 0\leq k_1,k_2\leq 1
$$

当 Precision 和 Recall 等权重，即 $k_1=k_2$ 时，就是最常见的 $F1$

$$
F1 = \frac{2\times Acc_{(Precision)} \times Acc_{(Recall)}}{Acc_{(Precision)} + Acc_{(Recall)}}
$$

可知 $F1$ 综合了 $P$ 和 $R$ 的结果，当 $F1$ 较高时则能说明试验方法比较有效