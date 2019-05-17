# 数据游戏：预测3天后招商银行close时的股价
<p><span style="font-size: 12px;">&nbsp;</span></p>
<p>　　前阵子报名参加了一个数据比赛，题目是预测5月15号（星期三）招商银行的股价，截止时间是在5月12号（星期天）。在本次预测中，我用到的是岭回归。</p>
<p><span style="font-size: 12px;">&nbsp;</span></p>
<h2>一、岭回归</h2>
<h3>线性回归</h3>
<p>　　先回顾一下普通线性回归。一般来说，线性回归方程：y=w1x1+w2x2...+wnxn。我们把这组变量 xn&nbsp;定成一个矩阵 X，把回归系数存放在向量W中，则 y=X*W。</p>
<p>　　　<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190515142354836-1365548705.png" alt="" /></p>
<h3>存在的问题</h3>
<h4>　&nbsp; &nbsp; (1)、特征数大于样本数</h4>
<p>　　当特征数大于样本数的时候，上面的式子就存在问题了。矩阵要求逆，就必须为满秩矩阵，当特征数大于样本数的时候，就不为满秩了。可以通俗地理解为由于样本数量太少，没有办法提供足够的有效的信息。</p>
<h4>　&nbsp; &nbsp; (2)、多重共线性</h4>
<p>　　多重共线性指线性回归模型中的解释变量之间由于存在精确相关关系或高度相关关系而使模型估计失真或难以估计准确。举个例子，对于一般人来说，体重和身高是有很强的关联的，但如果我们需要预测某样东西，以这两者作为自变量，即使可以很好的拟合，但这个模型的解释性还是不够。</p>
<p>&nbsp;</p>
<p>&nbsp;　　由于上面两个问题的存在，岭回归就出现了。它解决回归中重大疑难问题：排除多重共线性，进行变量的选择，在存在共线性问题和病态数据偏多的研究中有较大的实用价值。按照度娘的解释：岭回归是一种专用于共线性数据分析的有偏估计回归方法，实质上是一种改良的最小二乘估计法，通过放弃最小二乘法的无偏性，以损失部分信息、降低精度为代价获得回归系数更为符合实际、更可靠的回归方法，对病态数据的拟合要强于最小二乘法。</p>
<p>　　岭回归在上面式子的基础上做了点儿改进：<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190515152735289-537364933.png" alt="" />，（其中<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190515155332506-1366814671.png" alt="" />称为岭参数）很好地解决了上面的问题，假如<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190515162330307-1535748698.png" alt="" />是一个奇异矩阵（不满秩），添加<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190515162242843-1886511237.png" alt="" /><span id="MathJax-Element-51-Frame" class="MathJax" data-mathml="&lt;math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;&gt;&lt;msup&gt;&lt;mi&gt;X&lt;/mi&gt;&lt;mi&gt;T&lt;/mi&gt;&lt;/msup&gt;&lt;mi&gt;X&lt;/mi&gt;&lt;/math&gt;"><span class="MJX_Assistive_MathML">后可以保证其可逆</span></span>。</p>
<p>&nbsp;</p>
<h2>二、数据获取</h2>
<p>　　本次数据是通过 Tushare 的 get_hist_data()获取的。Tushare是一个免费、开源的python财经数据接口包。python安装tushare直接通过<br />    pip install tushare 即可安装。</p>
<div class="cnblogs_code">
<pre><span style="color: #0000ff;">import</span><span style="color: #000000;"> tushare as ts
data </span>= ts.get_hist_data(<span style="color: #800000;">'</span><span style="color: #800000;">600848</span><span style="color: #800000;">'</span>)</pre>
</div>
<p> 　　运行之后可以查看它的前后几行数据，按照tushare官方的说明，get_hist_data()只能获取近3年的日线数据，而他的返回值的说明是这样的：<br />〖date：日期；open：开盘价；high：最高价；close：收盘价；low：最低价；volume：成交量；price_change：价格变动；p_change：涨跌幅；ma5：5日均价；ma10：10日均价；ma20:20日均价；v_ma5:5日均量；v_ma10:10日均量；v_ma20:20日均量〗</p>
<p>    　　均价的意思大概就是股票n天的成交价格或指数的平均值。均量则跟成交量有关。至于其他的返回值，应该是一下子就能明白的吧。在获得数据之后，我们查看一下描述性统计，通过&nbsp;<code>data.describe()&nbsp;</code>查看是否存在什么异常值或者缺失值。</p>
<p>　　&nbsp;<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190514091155404-777253643.png" alt="" /></p>
<p>　　这样看来似乎除了由于周末以及节假日不开盘导致的当天的数据缺失以外，并没有其他的缺失和异常。但是这里我们不考虑节假日的缺失值。</p>
<p>&nbsp;</p>
<h2>三、数据预处理</h2>
<p>　　由于获取的数据是按日期降序排序，但本次预测跟时间序列有关，因此我们需要把顺序转一下，让它按照日期升序排序。</p>
<div class="cnblogs_code">
<pre>data1 = data[::-1]</pre>
</div>
<p>　　处理完顺序之后，我们要做一下特征值的选择。由于 volume&nbsp;以及均量的值很大，如果不进行处理的话，很可能对整体的预测造成不良影响。由于时间有限，而且考虑到运算的复杂度，这里我没有对这些特征进行处理，而是直接将它们去掉了。至于均价，我是按照自己的理解，和10日均价、20日均价相比，5日均价的范围没那么大，对近期的预测会比另外两个要好，因此保留5日均价。接着，我用 sklearn.model_selection 的 cross_val_score，分别查看除〖'open', 'close', 'high', 'low', 'ma5'〗以外的其他剩余属性对预测值的影响。发现 &lsquo;p_change&rsquo;、'price_change'&nbsp;这两个属性对预测结果的影响不大，为了节省内存，增加运算速度，提高预测的准确性，也直接把它们去掉了。完了之后，查看前后三行数据。</p>
<div class="cnblogs_code">
<pre>data1 = data1[[<span style="color: #800000;">'</span><span style="color: #800000;">open</span><span style="color: #800000;">'</span>,<span style="color: #800000;">'</span><span style="color: #800000;">high</span><span style="color: #800000;">'</span>,<span style="color: #800000;">'</span><span style="color: #800000;">low</span><span style="color: #800000;">'</span>,<span style="color: #800000;">'</span><span style="color: #800000;">ma5</span><span style="color: #800000;">'</span>,<span style="color: #800000;">'</span><span style="color: #800000;">close</span><span style="color: #800000;">'</span><span style="color: #000000;">]]
data1.head(</span>3), data1.tail(3)</pre>
</div>
<p>　　　　　　　　<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190514094827934-2038051683.png" alt="" /></p>
<h2>四、建模预测</h2>
<p>　　由于提交截止日期是周日，预测的是周三，因此需要先对周一周二的信息进行预测。在这里我突然想到一个问题，是用前一天的所有数据来训练模型以预测当天的 close 比较准确，还是用当天除了&nbsp;close&nbsp;以外的其他数据来训练模型以训练当天的&nbsp;close&nbsp;比较准呢？为了验证这个问题，我分别对这两种方法做了实验。</p>
<p>　　为了减少代码量，定义了一个函数用以评估模型的错误率。</p>
<div class="cnblogs_code">
<pre><span style="color: #0000ff;">def</span><span style="color: #000000;"> get_score(X_train, y_train):
    ridge_score </span>= np.sqrt(-cross_val_score(ridge, X_train, y_train, cv=10, scoring=<span style="color: #800000;">'</span><span style="color: #800000;">neg_mean_squared_error</span><span style="color: #800000;">'</span><span style="color: #000000;">))
    </span><span style="color: #0000ff;">return</span> np.mean(ridge_score)</pre>
</div>
<p>　(1)、用前一天的所有数据来当训练集</p>
<div class="cnblogs_code">
<pre>y_train = data1[<span style="color: #800000;">'</span><span style="color: #800000;">close</span><span style="color: #800000;">'</span>].values[1<span style="color: #000000;">:]
X_train </span>= data1.values[:-1<span style="color: #000000;">]
score </span>= get_score(X_train, y_train)</pre>
</div>
<p>　　输出结果大约为0.469，这个错误率就比较大了，不太合理，更何况还要预测其他特征值作为测试数据。</p>
<p>&nbsp; &nbsp;(2)、用当天除了&nbsp;close&nbsp;以外的其他数据来当训练集</p>
<div class="cnblogs_code">
<pre>data2 =<span style="color: #000000;"> data1[:]
y_train </span>= data2.pop(<span style="color: #800000;">'</span><span style="color: #800000;">close</span><span style="color: #800000;">'</span><span style="color: #000000;">).values
X_train </span>=<span style="color: #000000;"> data2.values
score </span>= get_score(X_train, y_train)</pre>
</div>
<p>　　输出结果大约为0.183，跟第一个相比简直好多了。所以，就决定是你了！</p>
<p>　　接下来建模并把模型保存下来：</p>
<div class="cnblogs_code">
<pre>y_train = data1[<span style="color: #800000;">'</span><span style="color: #800000;">close</span><span style="color: #800000;">'</span><span style="color: #000000;">]
X_train </span>= data1[[<span style="color: #800000;">'</span><span style="color: #800000;">open</span><span style="color: #800000;">'</span>, <span style="color: #800000;">'</span><span style="color: #800000;">high</span><span style="color: #800000;">'</span>, <span style="color: #800000;">'</span><span style="color: #800000;">low</span><span style="color: #800000;">'</span>, <span style="color: #800000;">'</span><span style="color: #800000;">ma5</span><span style="color: #800000;">'</span><span style="color: #000000;">]]
close_model </span>=<span style="color: #000000;"> ridge.fit(X_train, y_train)
joblib.dump(ridge, </span><span style="color: #800000;">'</span><span style="color: #800000;">close_model.m</span><span style="color: #800000;">'</span>)</pre>
</div>
<p>　　在预测之前呢，我们先拿训练集的后8组数据做一下测试，做个图看看：</p>
<div class="cnblogs_code">
<pre>scores =<span style="color: #000000;"> []
</span><span style="color: #0000ff;">for</span> x <span style="color: #0000ff;">in</span> X_train[-8<span style="color: #000000;">:]:
    score </span>= close_model.predict(np.array(x).reshape(1, -1<span style="color: #000000;">))
    scores.append(score)
x </span>= np.arange(8<span style="color: #000000;">)
fig, axes </span>= plt.subplots(1, 1, figsize=(13, 6<span style="color: #000000;">))
axes.plot(scores)
axes.plot(y_train[</span>-8<span style="color: #000000;">:])
plt.xticks(x, data1.index[</span>-8:].values, size=13, rotation=0)</pre>
</div>
<p>　　<img src="https://img2018.cnblogs.com/blog/1458123/201905/1458123-20190514155338491-2123587494.png" alt="" /></p>
<p>&nbsp;　　看到这样子我还是相对比较放心的，不过，这个模型的训练值除了&ldquo;close&rdquo;以外的属性都是已知的，要预测三天后的还得预测前两天的测试值。</p>
<div class="cnblogs_code">
<pre><span style="color: #0000ff;">def</span><span style="color: #000000;"> get_model(s):
    y_train </span>= data1[s].values[1<span style="color: #000000;">:]
    X_train </span>= data1.values[:-1<span style="color: #000000;">]
    model </span>=<span style="color: #000000;"> ridge.fit(X_train, y_train)
    </span><span style="color: #0000ff;">return</span> model</pre>
</div>
<div class="cnblogs_code">
<pre><span style="color: #0000ff;">def</span><span style="color: #000000;"> get_results(X_test):
    attrs </span>= [<span style="color: #800000;">'</span><span style="color: #800000;">open</span><span style="color: #800000;">'</span>, <span style="color: #800000;">'</span><span style="color: #800000;">high</span><span style="color: #800000;">'</span>, <span style="color: #800000;">'</span><span style="color: #800000;">low</span><span style="color: #800000;">'</span>, <span style="color: #800000;">'</span><span style="color: #800000;">ma5</span><span style="color: #800000;">'</span><span style="color: #000000;">]
    results </span>=<span style="color: #000000;"> []
    </span><span style="color: #0000ff;">for</span> attr <span style="color: #0000ff;">in</span><span style="color: #000000;"> attrs:
        result </span>=<span style="color: #000000;"> get_model(attr).predict(X_test)
        results.append(result)
    </span><span style="color: #0000ff;">return</span> results</pre>
</div>
<p>&nbsp;　　接下来预测三天的股价：</p>
<div class="cnblogs_code">
<pre>X_test = data1[-1<span style="color: #000000;">:].values
</span><span style="color: #0000ff;">for</span> i <span style="color: #0000ff;">in</span> range(3<span style="color: #000000;">):
    results </span>=<span style="color: #000000;"> get_results(X_test)
    close </span>= close_model.predict(np.array(results).reshape(1, -1<span style="color: #000000;">))</span><span style="color: #000000;">
    results.append(close)
    X_test </span>= np.array(results).reshape(1, -1<span style="color: #000000;">)
</span><span style="color: #0000ff;">print</span>(<span style="color: #800000;">"</span><span style="color: #800000;">5月15日招商银行关盘时的股价为：</span><span style="color: #800000;">"</span> + str(round(close[0], 2)))</pre>
</div>
<pre>5月15日招商银行关盘时的股价为：33.44</pre>
<h2>五、总结</h2>
<p>　　虽然预测结果是这样子，但感觉这样预测似乎很菜啊。毕竟预测的每个值都会有偏差，多个偏差累加起来就很多了，这让我有点害怕。不知道存不存在不预测其他值直接预测close的方法，或者说直接预测5月15号的而不用先预测13、14号的方法。虽然我知道有种算法是时间序列算法，但不是很懂。希望哪位大神看了能给我一些建议，指点迷津。</p>
<p>　　对于一个自学数据分析的在校学生，苦于没有项目经验，正好赶上这次的【数据游戏】，能利用此次机会操作一波真的很不错。</p>
<p>&nbsp;</p>
