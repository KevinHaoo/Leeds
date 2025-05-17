## **1. 分类任务的评价指标**

### **(1) 常用指标**

| **指标**                            | **定义**                                                     | **特点**                                                     |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **准确率 (Accuracy)**               | $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$        | 简单直观，但在类别不平衡数据集上效果差。                     |
| **精确率 (Precision)**              | $\text{Precision} = \frac{TP}{TP + FP}$                      | 测量模型预测为正类的样本中真正为正类的比例，适合于关注假阳性较少的场景。 |
| **召回率 (Recall)**                 | $\text{Recall} = \frac{TP}{TP + FN}$                         | 测量模型从所有正类样本中找出的比例，适合关注假阴性较少的场景。 |
| **F1-score**                        | $\text{F1-score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | 综合平衡精确率和召回率，适合类别不平衡的数据。               |
| **ROC曲线 (ROC Curve)**             | 显示不同阈值下 $\text{True Positive Rate (TPR)}$ 和 $\text{False Positive Rate (FPR)}$ 的关系。 | 直观展示分类模型性能，常用AUC（曲线下面积）作为单一指标，值越大越好。 |
| **PR曲线 (Precision-Recall Curve)** | 显示不同阈值下精确率和召回率的关系。                         | 适合类别不平衡时，关注正类的表现。                           |

### **(2) 指标特点总结**

- **类别平衡问题**：准确率在类别不平衡时可能误导，应关注F1-score或PR曲线。
- **场景适配**：精确率适合错误代价较高的任务（如垃圾邮件过滤），召回率适合遗漏代价较高的任务（如疾病检测）。

------

## **2. 聚类任务的评价指标**

### **(1) 内部指标（无标签聚类评估）**

| **指标**                              | **定义**                                                     | **特点**                                                     |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **轮廓系数 (Silhouette Coefficient)** | 计算每个点的紧密度（与同簇内其他点的相似性）和分离度（与其他簇的相似性）。公式为：$s = \frac{b - a}{\max(a, b)}$，其中 $a$ 是点与簇内其他点的平均距离，$b$ 是点与最近邻簇的平均距离。 | 取值范围为 [−1,1][-1, 1]，越接近1越好，适合衡量聚类整体的紧密性和分离性。 |
| **Dunn指数 (Dunn Index)**             | $\text{Dunn Index} = \frac{\min(\text{簇间距离})}{\max(\text{簇内距离})}$ | 衡量簇的分离性和紧密性，值越大越好，但对大规模数据不敏感。   |
| **DB指数 (Davies-Bouldin Index)**     | 衡量每个簇与其最近簇的距离和紧密度比值，公式为：$DB =\\ \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} \frac{s_i + s_j}{d_{ij}}$，其中 $s_i$ 是簇 $i$ 的平均紧密度，$d_{ij}$ 是簇 $i$ 和 $j$ 之间的距离。 | 值越小越好，适合对不同聚类结果的相对比较。                   |

### **(2) 外部指标（有标签聚类评估）**

| **指标**                                          | **定义**                                                     | **特点**                                             |
| ------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| **调整互信息 (Adjusted Mutual Information, AMI)** | 衡量真实标签和聚类结果标签之间的相似性，取值范围 [0,1][0, 1]，值越大表示越相似。 | 对标签个数的选择较为敏感。                           |
| **调整兰德指数 (Adjusted Rand Index, ARI)**       | 计算实际标签和预测标签中样本对的一致性，调整了随机划分的影响。 | 对类别数量的变化不敏感，适合对不同方法进行横向比较。 |
| **纯度 (Purity)**                                 | 衡量每个聚类中样本占主导类别的比例，公式为 $\text{Purity} = \frac{1}{N} \sum_{i=1}^k \max_{j}$ | $C_i \cap T_j$                                       |

------

## **3. 回归任务的评价指标**

### **(1) 误差类指标**

| **指标**                      | **定义**                                                     | **特点**                                            |
| ----------------------------- | ------------------------------------------------------------ | --------------------------------------------------- |
| **均方误差 (MSE)**            | $\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$  | 对异常值敏感，适合对较平稳的数据建模。              |
| **均方根误差 (RMSE)**         | $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$ | 是MSE的平方根，更容易与实际值量级对比，具有直观性。 |
| **平均绝对误差 (MAE)**        | $\text{MAE} = \frac{1}{n} \sum_{i=1}^n$                      | $y_i - \hat{y}_i$                                   |
| **平均绝对百分比误差 (MAPE)** | $\text{MAPE} = \frac{1}{n} \sum_{i=1}^n$                     | $\frac{y_i - \hat{y}_i}{y_i}$                       |

### **(2) 相关性类指标**

| **指标**                                             | **定义**                                                     | **特点**                                                     |
| ---------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **决定系数 (R²)**                                    | $R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$ | 衡量模型解释目标变量变化的能力，取值范围 [−∞,1][-\infty, 1]。 |
| **皮尔森相关系数 (Pearson Correlation Coefficient)** | $r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \cdot \sum (y_i - \bar{y})^2}}$ | 衡量预测值与真实值之间的线性相关程度，范围为 [−1,1][-1, 1]。 |



# L6 - Classification

###### KNN:

找到最近的 K 个与预测点最近的 训练集 中的点，K 通常为奇数，投票决定预测点的类别。

###### Decision Tree

==pure node==：只有一个标签的分支结点

==unsplittable node==：由于在某个特征上有重复数据、而无法再分割的结点

==gini impurity==：
 $1 - \sum_{i=0}^k{p_i^2}$ （$p_i$ 表示某个类别在当前划分集合中的比例），表示数据的类别混杂程度。
gini越小，表示该数据集的样本越纯（同一种类别的数量占据了大多数）。
**<u>在决策树构建下一个分裂结点时，作为衡量指标，比较各种划分方法后，选取平均 gini 最低的划分方法。</u>**

==node entropy==：$S = - \sum_{i = 0}^k{p_i*log_{2}{p_i}}$ （$p_i$ 表示某个类别在当前划分集合中的比例）。
衡量结点划分类别的纯度，entropy越小，类别越纯。（只有 1 个类别时 S = 0）
当一个集合中存在相同数量的 C 个类别时，$S = log_2{C}$ 。
<u>**怎么根据 S 构建决策树：$L = \frac {N_1S(N_1) + N_2S(N_2)}{N_1 + N_2} $ ，$N_i$ 是划分子集合的样本总数，S 是该子集的 entropy ，$L$ 越小，就选哪个。**</u>





# L7 - Regression and Model Fitting

###### Regression

==拟合方法==：

**least squares 最小二乘法**：定义 penalty function 损失函数为 $L = \sum_{i}{(y_i - (mx_i + b))^2}$

**L1 regression**：penalty function 为 $L = \sum_{i}{\\|{y_i - (mx_i + b)\\|}}$
可避免最小二乘法中，因为少量离群点导致的损失函数较大偏差

非线性回归

==多项式拟合==：

线性模型：output 向量 t 所表示的函数，是 input 向量 x 的（非）线性组合，但是一定是系数向量 w 的线性组合（**重要性质**）。**我们将这种关于未知参数的函数称为线性模型**。

==过拟合==：

**表现**：oscillate wildly and give poor representation of the function 剧烈振荡并且无法对目标函数概括性表示

**需要修正至的目标**：

achieve good generalization by making accurate predictions for new data 对新数据拥有准确的预测能力

**缓解方法**：

增大数据规模（学到更多通用特征，避免记住噪声）

正则化（损失函数中增加惩罚项、Dropout 一定比例的神经元、BatchNormalization 数据归一化）

交叉验证



###### 限制决策树复杂度（缓解过拟合）

**问题背景**：

在 full-growth 的决策树中由于分裂结点可能过多，可能存在过拟合，在新的测试数据上表现不好。

==解决策略==：

- 设置特殊规则，避免过度生长
- **剪枝**：
  先让树完全生长，再试着以各个结点分裂后的的最高频类别，作为该结点的预测类别。如果去除该结点的分裂后，在 validation set 验证集上没有准确率变化，则采用该剪枝。



# L8 - Dimentionality Reduction

> - 数据集的 dimensionality 等价于表示矩阵的秩 rank
>
> - 矩阵的列 = Dimensionality，即3D、4D

###### ==**Singular Value Decomposition 奇异值分解**==

**目的**：

使用更少的空间储存信息，转换矩阵可以根据算法自动生成，不用人工计算

**特点**：

使用后生成的新特征矩阵仍然是原本的尺寸 （维数 / 特征数量不变）

新矩阵和原来的数据代表的实际含义没有直接关系

$X = UΣ\ \ V^T$

> **==如何得到三个矩阵==**：
>
> $U = XX^T的特征向量线性组合,\ V = X^TX\ 的特征向量线性组合$
>
> $Σ = 奇异值σ_i构成的对角阵, 并且按降序排列,\ σ_i=\sqrt{λ_i}$
>
> 注：$X\ 为\ m\times n, 则\ U\ 为\ m\times m, V\ 为\ n\times n$
>
> **==特性==**：
>
> 1. $Σ$ 是 diagonal（对角矩阵），包含奇异值 $σ_i$ （非负）
> 2. 矩阵 $Σ$ 的秩 $=$ 原矩阵 $X$ 的秩（相当于特征值矩阵）
> 3. $U、V$ 自身的列内部都是特征向量，故两两正交（正交矩阵），且 $U^T=U^{-1}$
>
> ==**如何跟 PCA 对应**==：
>
> 1. PCA 的主成分对应的是 V 矩阵的列向量
>
> 2. 奇异值矩阵 Σ 对应 PCA 中权重概念（即按照了权重大小降序排列，故特征向量也是对应权重降序排列）
>
> 3. 如果保留前 k 个主成分，相当于：
>    截取 U、V 的前 k 列，Σ 的前 k 行前 k 列
>
> 4. $U_{il}$：第 i 个样本在第 l 个主成分上的贡献==每行对应一样本对各主成分的贡献情况==
>
>    $σ_l$：第 l 个主成分的权重（奇异值）
>
>    $V_{jl}$： 第 j 个特征在第 l 个主成分上的贡献==每行对应一特征对各主成分的贡献情况，列向量就是主成分，截断是截断的 $V^T$ 的行，所以就对应截断主成分==
>
> 5. **所以，将 V 的后面几列（ $V^T$ 后面几行）截断，则相当于把贡献小的主成分去除，但是每个特征（行）只是少了几个贡献值，特征数量也没有变，只是 V 的列向量（主成分）减少了**
>
> ==**降维的体现**==：
>
> 1. **【SVD 的基本效果】**如果原矩阵的秩比 $min(m,n)$ 要小，那么特征值、特征向量也一定少，那么得到的三个矩阵在去除 0 的部分后，一定就首先实现了降维的作用（<u>即假如某一个特征是别的特征的线性组合，那么这个一定是冗余特征</u>）
> 2. **【PCA 的思想驱使】**在此基础上，再筛选重要的主成分，截断相应矩阵，进一步实现降维

<img src="https://cdn.jsdelivr.net/gh/KevinHaoo/PicCloud/202501220805092.png" alt="截屏2025-01-22 上午8.05.40" style="zoom:40%;" />

<img src="https://cdn.jsdelivr.net/gh/KevinHaoo/PicCloud/202501220806563.png" alt="截屏2025-01-22 上午8.06.37" style="zoom:40%;" />

**==（$U$ 截列，$V^T$ 截行：$m\times k,\ k\times n,\ k$ 可变）==**



**Rank N Approximation ( 秩 N 近似 )**：

截断主成分后还剩余几个，就叫 Rank N Approximation











# Ethics1

根据直觉判断，根据不变的逻辑判断

基于一定理性逻辑，形成判断的一个总体情况分支框架，这可能和直觉不一定一致

How to cast doubt（针对 行为本身 和 动机）

Doubt method 1: Critique the action. Highlight reasons against（反驳行为，那就要关注行为背后的原因来反驳）

Doubt method 2: Critique the reason. Highlight the problem implications（反驳原因，那就要关注这个原因可能导致的不良影响，原因是否存在本质错误）



# Ethnics2

法律具有道德内涵，要从道德角度进行解读

法律 - 道德是割裂的，不是一回事，可能：都是 / 部分是 / 都不是 两者或两者之一

**UK-GDPR Principles ( General Data Protection Regulation )**:

1. 个人数据公平透明处理
2. 合法目的的收集和处理
3. 数据主体有被遗忘权

> **The function of ethics:**
>
> -  Determine what we should do, what moral (not legal) rights we have
>
> -  Based on how our actions impact others’ moral rights and wellbeing
>
> 
>
> **The function of law (ideally):**
>
> -  Provide guide rails to restrict behavior that could lead to harm
> -  Clarify consequences of that behavior / contain the ‘fallout’ from it (no vengeance spirals)
> -  Protect rights of individuals from powerful others (/the state)
> -  Reassure citizens about participating in a society, accessing goods/services
> -  Implement society’s ethical viewpoint, updated based on results of ethical reasoning



**角色定义**：

Controller：决定数据处理的目的和方式（构建数据库、并将其特定的目标任务分配给一些机构来处理）

Processor：代表 Controller 执行数据处理

Subject：数据信息包含的主体（数据反映了这个人的个人信息）

Information Commissioner：确保 GDPR 实施的机构



**data governance （数据治理）**：

**是什么**：

The control of data collection, processing and use【数据一系列操作的控制】

**怎么做**：

Through sets of processes that ensure certain standards are maintained【设定流程保证标准被维护】

**标准是什么**：

To uphold security and integrity 【增强安全和诚信】

**为什么**：

Protects both the data, and the well-being of the people the data relates to【保障数据和相关群体】



GDPR 把 personal data 分为两类，special 的需要额外的许可才能操作

special 的，包括宗教、政治、个人生物信息、性生活

7 个原则：（好的都选）

subject 权利：告知 / 访问 / 修正 / 被遗忘 / 反对使用 / 限制操作





# Ethics3

###### 数据生命周期：

- 收集：强调同意（Consent）问题
- 处理：涉及隐私（Privacy）问题
- 存储：关注数据的安全性（Security）
- 使用：客观性和责任问题（Objectivity & Responsibility）

###### 同意的三要素：

- **知情**（Informed）：主体了解用途
- **自主**（Autonomous）：确保同意是自由而非被迫的
- **能力**（Competent）：有能力执行同意的行为
- **持续**（Ongoing）：数据主体能随时撤回同意

######  匿名化与数据关联风险：

- 匿名化并不能完全消除风险，因为数据关联（Linkage）可能重新识别个体。
- 直接标识符（如姓名、地址）和间接标识符（如年龄、性别）都可能被用于识别个体。

> - 如果 action impermissible，但是可以通过 consent 来允许
> - 未经允许的使用，道德和法律都是不允许的
> - 隐式同意 implicit consent 有时是有效的
> - 

# Ethics4

###### 维护隐私与使用数据之间取得平衡：

- 数据主体的知情同意
- 数据使用不会对主体造成伤害

###### 偏见可能来源：

- 数据中包含的不公正历史
- 缺乏透明度和专业监督
