class: center, middle, inverse


# Introduction to Machine Learning
</br>
### by [@chenwang](mailto:wangchen53@huawei.com)
---
# Table of Content

</br>
## Part 1. Hello, (ML) World!

## Part 2. Linear Algebra

## Part 3. Python

## Part 4. Roadmap

---
class: center, middle, inverse

# Part 1: Hello, (ML) World
---
# Machine Learning Definition

--

> Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed.
> <footer>Arthur Samuel, 1959</footer>

--

</br>

### New Programming Paradigm

--

<img src="img/ai-paradigm-1.png" alt="drawing" height="100"/>
--
<img src="img/ai-paradigm-2.png" alt="drawing" height="100"/>

???
编程范式
---
# History of Artificial Intelligence

<img src="img/ai-history.png" alt="drawing" width="1000"/>

---
# Artificial Intelligence. v.s. Machine Learning v.s. Deep Learning

--
<img src="img/ai-ml-dl.png" alt="drawing" width="500"/>
--
<img src="img/ai-ml-diff.jpg" alt="drawing" width="500"/>
---
# Branches of Machine Learning

### Supervised Learning

* __Classification__: predict what class an instance of data should fall into;
* __Regression__: prediction of a numeric value;

### Unsupervised Learning

* __Clustering analysis__: group similar items together;
* __Association analysis__: find the relationship between data;
* __Dimension reduction__: reducing the number of random variables by obtaining a set of principal variables

### Self-supervised Learning (supervised learning without human in the loop)
* __Temporally supervised learning__: predict future based on what happened in the past
* __Generative Model__: learn the data distribution from the training set to generate new data with some variations.

### Reinforcement Learning


---
# Classification

### Binary Classification:

- Each instance can be classified into only two categories

### Multi-Class Classification

- Each instance can be classified into more than two categories

### Multi-label Classification

- Each instance can have more than one labels


<img src="img/spam_email.jpg" alt="drawing" height="200"/>
&nbsp;
<img src="img/Mnist-database-hand-written-digits.png" alt="drawing" height="200"/>
&nbsp;
<img src="img/carplaterecognition.jpg" alt="drawing" height="200"/>


---
# Regression

### Scalar Regression

- A task where the target is a continuous scalar value.

### Vector Regression

- A task where the target is a set of continuous values. (Predict trajectory)

<img src="img/age-estimation.png" alt="drawing" height="300"/>
&nbsp;
<img src="img/Weather-forecast.jpg" alt="drawing" height="300"/>

---
# Clustering Analysis

### Partitional

- Decomposes a data set into a set of disjoint clusters.

### Hierarchical

- Produce a nested series of partitions

<img src="img/customer-segmentation.png" alt="drawing" height="250"/>
&nbsp;
<img src="img/fish-taxonomy.jpg" alt="drawing" height="250"/>

[1] Jain, A.K., Murty, M.N. and Flynn, P.J., 1999. Data clustering: a review. ACM computing surveys (CSUR), 31(3), pp.264-323.

---
# Association Analysis (Association Rule Mining)

###  Discovering interesting relationships hidden in large data sets
- Frequent Itemsets
- Association Rules

<img src="img/association-analysis.png" alt="drawing" height="320"/>
<img src="img/amazon-recommendation.png" alt="drawing" height="320"/>

---
# Semi-Supervised Learning

### The Value of Unlabeled Data
- Not enough labeled data
    - Data is cheap, labeled data is expensive
- The distribution of unlabeled data tells something

### Assumptions (Distribution & Label)
- Low-Density Assumption
- Smoothness Assumption

<img src="img/semi-supervised.png" alt="drawing" height="200"/>
<img src="img/iphone-people.png" alt="drawing" height="250"/>

???
Digit number recognition
2,2,2,2,2,2,3

---
# Reinforcement Learning

### Learning What to Do - Map Situation to Actions

- Trial-and-error
- trade-off between **exploration** and **exploitation**

### Markov Decision Model
- Sensation
- Action
- Goal

<img src="img/alpha_go.jpg" alt="drawing" height="200"/>
<img src="img/autonomous-driving.jpg" alt="drawing" height="200"/>
<img src="img/atari.gif" alt="drawing" height="200"/>

???
Supervised learning is not practical for interactive problems
Example: ride a bike

Objective: Maximize the Reward
---
# Generative Models (1/2)

### Image Generation

### Data Manipulation: Meitu

### Privacy Preservation: Sharing Sensitive Data

<img src="img/GAN.png" alt="drawing" height="300"/>
&nbsp;
<img src="img/Gan-data-manipulation.gif" alt="drawing" height="300"/>

---
# Generative Models (2/2)

### Sequence Generation
- Text Generation
- Music Generation: [AIVA](https://www.youtube.com/watch?v=HAfLCTRuh7U)

>微明的灯影里, 我知道她可爱的土壤, 是我的心灵成为俘虏,
>我不在我的世界里, 街上没有一只灯儿舞, 是最可爱的,
>你睁开眼睛做起的梦, 是你的声音啊.
>
><footer>《是你的声音啊》, 微软小冰</footer>

<img src="img/AIVA.png" alt="drawing" height="188"/>

---
class: center, middle

# Homework 1:

# How Machine Learning can help you in your daily work?

---
class: center, middle, inverse

# Part 2: Linear Algebra

???
The Matrix Cookbook (Petersen and Pederson, 2006)

---
# Scalar, Vector, Matrix and Tensor

### Scalar: just a single number

### Vector: an array of numbers - identifying points in space

### Matrix: 2-D array of numbers

- $A_{i,:}$: the ith row
- $A_{:,j}$: the jth column

### Tensor: High-dimensional array

<img src="img/tensor.png" alt="drawing" height="300"/>

???
Draw a table on the board
person : age, height, weight, salary, yrs of edu.,
---

# Matrix Multiplication

### Transpose
Mirror image of the matrix across a main diagonal.

### Element-wise Product (Hadamard Product)
Two matrices must have the same shape.

### Dot Product

- Dot product of two vectors x and y of the same dimensionality is the matrix product $x^Ty$.

<img src="img/matrix-transpose.gif" alt="drawing" height="120"/>
<img src="img/element-wise-product.png" alt="drawing" height="200"/>
<img src="img/Matrix_multiplication_2.png" alt="drawing" height="200"/>
---
# Matrix Multiplication

### Distributive:

$$A(B+C) = AB + AC$$

### Associative:
$$A(BC) = (AB)C$$

### Not commutative: AB = BA does not always hold
- always hold for vectors

$$x^Ty = y^Tx$$

### Transpose:
$$(AB)^T = B^TA^T$$

---
# Why Matrix Multiplication

### Matrix-vector product notation provides a more compact representation for equations of this form (Blackboard).

---
# Identity Matrix and Inverse Matrix

### Identity Matrix:
An identity matrix $I_n$ is a matrix that does not change any n-dimensional vector when we multiply that vector by that matrix.

$$\forall x \in R^n: I^nx = x, I_n \in R^{n*n}$$

All of the entries along the main diagonal are 1, while all of the other entries are zero.

### Inverse Matrix

The inverse matrix of A is denoted as $A^{-1}$, we have:

$$A^{-1}A=I_n$$

### Solve the linear equation

<table style="width:80%">
  <tr>
    <th>$Ax = b$</th>
    <th>$A^{-1}Ax = A^{-1} b$</th>
    <th>$I_nx = A^{-1} b$</th>
    <th>$x = A^{-1} b$</th>
  </tr>
</table>

---
# Norms

### Measuring the size of a vector


###

---
# Linear Regression

Instance | Features | Labels | Output
:-----: | :-----: | :-----: | :-----:
$x^1$ | $[x^1_1, x^1_2, ..., x^1_m]$ | $\hat{y}^1$ | $y^1$
$x^2$ | $[x^2_1, x^2_2, ..., x^2_m]$ | $\hat{y}^2$ | $y^2$
... | ... | ... | ...
$x^n$ | $[x^n_1, x^n_2, ..., x^n_m]$ | $\hat{y}^n$ | $y^n$

- $x^i, i\in[1,n]$: the $i^{th}$ __instance__

- $x^i_j, i\in[1,n], j \in [1, m]$: the $j^{th}$ **feature** of the $i^{th}$ instance

- $y^i, i \in [1, n]$: model **output** for the $i^{th}$ instance (class / prediction/ ...)

- $\hat{y}^i, i \in [1, n]$: the **label** (ground truth) of the $i^{th}$ instance

- **Cost function** (prediction error): $C_i = (y^i - \hat{y}^i)^2, i \in [1,n]$

- **Loss Function** (model accuracy): L = ${1 \over n} \sum^n_{i=1}C_i$

- **Predictive Model**: $f :  X \rightarrow y$

???

Linear Model
---

class: center, middle, inverse

# Part 3: Python
---
# Why Python?

### Point++

* Clear syntax: Executable pseudo-code;
* Python is popular: active community;
* Large number of libraries: sci. computing;

### point--

* Low efficiency;
---
# Anaconda (1/3)

### Download & Install [Anaconda](https://www.anaconda.com)

```bash
echo 'export PATH="~/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```
### Proxy Setting

[URL Encoder](https://www.url-encode-decode.com): Huawei@123 -> Huawei%40123

```bash
set http_proxy=http://username:password@proxycn2.huawei.com:8080
set https_proxy=http://username:password@proxycn2.huawei.com:8080
conda config --set ssl_verify false
```

### Create Env.
```bash
conda create --name tf tensorflow
```
---
# Anaconda (2/3)

### Activate / Deactivate Env.
```bash
activate tf # for Windows
source activate tf # for Linux & Mac

deactivate tf # for Windows
source deactivate tf # for Linux & Mac
```
###  List Env.

```bash
conda env list
```

### Remove Env.
```bash
conda remove --name tf --all
```
---
# Anaconda (3/3)

### List all packages

```bash
conda list
```
### Install a packages

```bash
conda install numpy
```
---
# Jupyter Notebook

### 1. Start up jupyter notebok
```bash
> jupyter notebook &
[I 12:49:03.038 NotebookApp] [nb_conda_kernels] enabled, 1 kernels found
[I 12:49:03.315 NotebookApp] Serving notebooks from local directory: /Users/chenwang
[I 12:49:03.315 NotebookApp] The Jupyter Notebook is running at:
[I 12:49:03.315 NotebookApp] http://localhost:8888/
[I 12:49:03.315 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```
### 2. Open your browser, go to localhost:8888

<img src="img/jupyter-notebook.png" alt="drawing" height="250"/>

---
# Numpy (1/4)

### [NumPy](http://www.numpy.org):  linear algebra (matrix) operations

### Scalar (0-D Tensor)
```python
>>> import numpy as np
>>> x = np.array(12)
>>> x
array(12)
>>> x.ndim
0
```
### Vector (1-D Tensor)

```python
>>> x = np.array([12, 3, 6, 14])
>>> x
array([12, 3, 6, 14])
>>> x.ndim
1
```
---
# Numpy (2/4)

### Matrix (2-D Tensor)
```python
>>> x = np.array([[5, 78, 2, 34, 0],
[6, 79, 3, 35, 1],
[7, 80, 4, 36, 2]])
>>> x.ndim
2
```
### High-Dimension Tensor

```python
>>> x = np.array([[[5, 78, 2, 34, 0],
[6, 79, 3, 35, 1],
[7, 80, 4, 36, 2]],
[[5, 78, 2, 34, 0],
[6, 79, 3, 35, 1],
[7, 80, 4, 36, 2]],
[[5, 78, 2, 34, 0],
[6, 79, 3, 35, 1],
[7, 80, 4, 36, 2]]])
>>> x.ndim
3
```
---

# Numpy (3/4)

### Vectorization: much faster

- The majority of Numpy package is written in C
- Using compiled code
- Parallelization (blackboard)
- 0.3ms v.s. 60ms

```python
# Vectorization version
c = np.dot(a, b)
```

```python
# Loop version
c = 0
tic = time.time()
for i in range (100000):
    c += a[i]*b[i]
```

---
# Numpy (4/4)

### Broadcasting: the smaller array is “broadcast” so that they have compatible shapes.

<img src="img/broadcasting.png" alt="drawing" height="350"/>

```python
x = np.array([1,2,3,4])
print(x+100)

> [101, 102, 103, 104]
```
---
# Pandas (1/3) - DataFrame Data Structure

### Processing Structured Data

### DataFrame: csv

<img src="img/pandas.png" alt="drawing" height="350"/>
---
# Pandas (2/3) - Basic Operations

```python
import pandas as pd
# 读取csv 数据
data_raw = pd.read_csv('file_name.csv')
# 显示前十行数据
first_10_lines = data_raw.head(10)
# 获取所有index
index = data_raw.index
# 获取所有列名
columns = data_raw.columns
# 获取所有数据内容
data = data_raw.values
# 显示每一列的数据类型
data_type = data_raw.dtypes
# 通过列名获取某一列
series_col = data_raw.loc[:,'timestamp']
# 重新设置某一列为index
metric = data_raw.set_index('timestamp')
# 根据index 获取一行或多行
df = metric.loc['2016-10-24']
series_col = metric.loc['2016-10-24 23:55:00']
```

---
# Pandas (3/3) - Data Cleansing: Missing Value

### Fill Missing Value with a Specific Number

```python
dataFrame.fillna(0)
```

### Remove Missing Values

```python
dataFrame.dropna()
```

### Interpolate Values According to Linear Methods.

```python
dataFrame.interpolate()
```

---
# Matplotlib

### Plot interactive figures

<img src="img/matplotlib_example1.png" alt="drawing" height="380"/>
<img src="img/matplotlib_example2.png" alt="drawing" height="380"/>

Examples: https://matplotlib.org/gallery/index.html
---

# Scikit-Learn

### 调包

<img src="img/ml-joke.jpg" alt="drawing" height="500"/>

---
# Homework 2: Auto-Baseline

### Metric Data

<img src="img/hw-metric-10.png" alt="drawing" height="280"/>
<img src="img/hw-metric-plot.png" alt="drawing" height="280"/>

### You have to do:

- Compute daily baseline

- Identify anomalies according to the deviation from the baseline
---
# Homework 2: Auto-Baseline

### Calculate Baseline

<img src="img/hw_1.png" alt="drawing" height=450""/>

---

# Homework 2: Auto-Baseline

### Anomaly Detection

<img src="img/hw_2.png" alt="drawing" height=450""/>

---
# Homework 2: Auto-Baseline

### Take One More Step, If You Have Time...

- Alarm on Positive / Negative Variation
- Use different statistical value for resampling
- Cold start: compute baseline even if there is not enough data
- Real-time alarms
- ...


---
class: center, middle, inverse

# Part 4: Roadmap

---
# What You Can Learn After This Course?

</br>

### - Understand the state of the art of machine learning

### - Understand basic machine learning algorithms

### - Have machine learning **mindset** to solve problems

### - Be able to build a machine learning project

### - Participate AIOps projects
---
# Learning Curve

<img src="img/learning-curve.jpg" alt="drawing" height="500"/>

---
# Roadmap of This Course

### 1. Introduction: Welcome On Board!

### 2. Clustering Analysis

* 2.1 Clustering Basics
* 2.2 Advanced Topics
* 2.3 AIOps Projects: Blackbox

### 3. Association Analysis

* 5.1 A-priori & FP-Growth
* 5.2 AIOps Projects: Insights

### 4. Regression

* 3.1 Linear Regression

### 5. Classification

* 5.1 Classification Basics
* 5.2 Model Evaluation
* 5.3 Advanced Topics


---
# What You Need to Learn This Course (WELL)?

</br></br>

<img src="img/notebook_pen_icon.png" alt="drawing" height="300"/>
&nbsp;
<img src="img/practice.png" alt="drawing" height="300"/>
&nbsp; &nbsp;&nbsp;
<img src="img/team-work.png" alt="drawing" height="300"/>

---
# Next.

### Date: Mar. 13, 2019
### Time: 18:30 ~ 20:30
### Topic: Clustering Analysis I - Basics

<img src="img/clustering_points.png" alt="drawing" width="300"/>
--
<img src="img/clustering_case1.png" alt="drawing" width="300"/>
--
<img src="img/clustering_case2.png" alt="drawing" width="300"/>

</br>

--

### Homework Submission: Mar. 6, 2019

---
class: center, middle, inverse

# Thanks You

### Welcome On Board! ;-)
