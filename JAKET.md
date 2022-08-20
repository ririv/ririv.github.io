> Yu, D., Zhu, C., Yang, Y., & Zeng, M. (2022). JAKET: Joint Pre-training of Knowledge Graph and Language Understanding. _Proceedings of the AAAI Conference on Artificial Intelligence_, _36_(10), 11630-11638. https://doi.org/10.1609/aaai.v36i10.21417

AAAI 2022
CMU & Microsoft
注意：虽然文章2022.6发表，但这篇文章早在2020.10就挂在了arXiv上
Code: 无

## Intro
所有以前的工作都有一个共同的挑战：当预训练模型在具有以前未见过的知识图的新领域中进行微调时，它很难适应新的实体、关系和结构

因此，我们提出了 JAKET，一个用于知识图和文本的联合预训练框架。我们的框架包含一个知识模块和一个语言模块，它们通过提供所需信息相互帮助，以实现更有效的语义分析。
知识模块基于图形注意网络 (Velickovicet al. 2018)，为语言建模提供结构感知实体嵌入。语言模块在给定描述性文本的情况下生成上下文表示作为 KG 实体和关系的初始嵌入。因此，在这两个模块中，内容理解都是基于相关知识和丰富的上下文。
一方面，联合预训练有效地将实体/关系和文本投影到共享的语义潜在空间中，从而简化了它们之间的语义匹配。另一方面，由于知识模块从描述性文本中产生表示，它解决了过度参数化问题，因为实体嵌入不再是模型参数的一部分。
为了解决两个模块之间的循环依赖，我们提出了一种新颖的两步语言分别是模块LM1和LM2。LM1为LM2和KG提供嵌入。来自 KG 的实体嵌入也被馈送到 LM2，它产生最终表示。LM1 和 LM2 可以很容易地建立为前几个转换器层和预训练语言模型（如 BERT 和 RoBERTa）的其余层。此外，我们设计了一个具有周期性更新的实体上下文嵌入内存，它将预训练速度提高了大约 15 倍。
预训练任务都是自监督的，包括知识模块的实体类别分类和关系预测，以及语言模块的掩码标记预测和掩码实体预测。我们框架的一个很大的好处是它可以很容易地适应看不见的知识图谱。
微调阶段。由于实体和关系的初始嵌入来自它们的描述性文本，因此 JAKET 并不局限于任何固定的 KG。凭借在预训练期间整合结构信息的学习能力，该框架可扩展到**具有以前未见过的实体和关系**的新知识图谱，如图 1 所示。我们对几个知识感知自然语言理解 (NLU) 任务进行了实证研究，包括few-shot关系分类、KG问答和实体分类。结果表明，与强基线方法相比，JAKET 在所有任务上都取得了最好的性能，包括那些具有以前未见过的知识图的任务。

![[Pasted image 20220720030628.png]]

## 方法
![[Pasted image 20220720030715.png]]
图 2：JAKET 结构示例，左侧为语言模块，右侧为知识模块。符号 Circle i  表示计算上下文表示的步骤。 “E:”、“R:”和“C:”分别代表 KG 中的实体、关系和类别。文本中提及的实体被标记为红色和粗体，例如 Sun。

### 定义
知识图由 $\mathcal{K} \mathcal{G}=(\mathcal{E}, \mathcal{R}, \mathcal{T})$ 表示，其中 $\mathcal{E}=\left\{e_{1} \ldots e_{N}\right\}$ 是实体集 and $\mathcal{R}=\left\{r_{1} \ldots r_{P}\right\}$  是关系集. $\mathcal{T}=\left\{\left(e_{t_{i}^{1}}, r_{t_{i}^{2}}, e_{t_{i}^{3}}\right) \mid 1 \leq i \leq\right.$ $\left.T, e_{t_{i}^{1}}, e_{t_{i}^{3}} \in \mathcal{E}, r_{t_{i}^{2}} \in \mathcal{R}\right\}$ 代表头-关系三元组的集合。 $N_{v}=\{(r, u) \mid(v, r, u) \in \mathcal{T}\}$ 表示实体 $v$ 的相邻关系和实体的集合。

我们定义 $\mathcal{V}=\left\{[\mathrm{MASK}],[\mathrm{CLS}],[\mathrm{EOS}], w_{1} \ldots w_{V}\right\}$ 作为标记的词汇表，上下文文本 $\mathbf{x}=$ $\left[x_{1}, x_{2}, \ldots, x_{L}\right]$ 作为标记序列，其中$x_{i} \in \mathcal{V}$。在词汇表中，[MASK] 是掩码语言建模的特殊标记（Devlin et al. 2019），[CLS]、[EOS] 是表示序列开始和结束的特殊标记。我们将 $F$ 定义为token嵌入的维度，它等于来自 KG 的实体/关系嵌入的维度。

文本 $\mathbf{x}$ 有一个实体提及(metion) $\mathbf{m}=$ $\left[m_{1}, \ldots, m_{M}\right]$ 的列表，其中每个提及 $m_{ i}=\left(e_{m_{i}}, s_{m_{i}}, o_{m_{i}}\right)$ : $e_{m_{i}}$ 是对应的实体，$s_ {m_{i}}, o_{m_{i}}$ 是上下文中此提及的开始和结束索引。换句话说，$\left[x_{s_{m_{i}}}, \ldots, x_{o_{m_{i}}}\right]$ 与实体 $e_{m_{i}}$（我们不考虑在这项工作中提到不连续的实体。）。我们假设提及的跨度对于给定的文本序列是不相交的。

由于知识图中的实体由没有上下文的节点表示，因此我们**使用实体描述文本来描述实体的概念和含义**。对于每个实体 $e_{i}$，它的描述文本 $\mathbf{x}^{e_{i}}$ 描述了这个实体。在 $\mathbf{x}^{e_{i}}$ 中提到 $e_{i}$ 表示为 $m^{e_{i}}=\left(e_{i}, s_{i}^ {e}, o_{i}^{e}\right)$，定义同上。例如，实体“sun”的描述文本可以是“[CLS] The Sun is the star at the center of the Solar System [EOS]”。然后metion 是 $m^{\text {Sun }}=(Sun, 3,3)$。如果在其描述文本中多次提及 $e_{i}$，我们选择第一个。如果在其描述文本中没有提及 $e_{i}$，我们设置 $s_{i}^{e}=o_{i}^{e}=1$。同样，我们将关系描述文本定义为可以描述每个关系的文本。

> 3,3 表示 sun在描述文本中，实体Sun始于第三个位置，结束于第三个位置，[CLS]也应计入，它是第一个位置

### 知识模块

知识模块（KM）的目标是对知识图谱进行建模以生成基于知识的实体表示。

为了计算实体节点嵌入，我们使用 Graph Attention Network (**GAT**) (Velickovic et al. 2018) (我们还尝试了图卷积网络 (GCN) (Kipf and Welling 2017)，但它在预训练任务上的表现比 GAT 差)，它使用自注意力机制为不同的相邻节点指定不同的权重。然而，vanilla GAT 是为具有单关系边的同构图设计的。为了利用多关系信息，我们采用**组合算子** (Vashishth et al. 2020) 的思想来组合实体嵌入和关系嵌入。详细地说，在$\mathrm{KM}$的第$l$层，我们更新实体$v$的嵌入$E_{v}^{(l)}$如下：首先对于每个关系实体对$(r, u) \in \mathcal{N}_{v}$，我们将实体 $u$ 的嵌入与关系 $r$ 的嵌入结合起来：
$$
M_{u, r}^{(l-1)}=f\left(E_{u}^{(l-1)}, R_{r}\right)
$$
请注意，关系嵌入 $R_{r}$ 在不同层之间共享。函数 $f(\cdot, \cdot): \mathbb{R}^{F} \times \mathbb{R}^{F} \rightarrow \mathbb{R}^{F}$ 合并一对实体和关系嵌入到一种表示中。在这里，我们设置了受 TransE 启发的 $f(x, y)=x+y$ (Bordes et al. 2013)。也可以应用更复杂的功能，例如 MLP 网络。然后通过图注意力机制聚合组合嵌入：
$$
m_{v}^{k}=\sigma\left(\sum_{(r, u) \in \mathcal{N}_{v}} \alpha_{v, r, u}^{k} W^{k} M_{u, r}^{(l-1)}\right)
$$
其中$k$是注意力头的索引，$W^{k}$是模型参数。注意分数 $\alpha_{v, r, u}^{k}$ 的计算公式为：
$$
\begin{aligned}
S_{u, r} &=\mathbf{a}^{T}\left[W^{k} E_{v}^{(l-1)} \oplus W^{k} M_{u, r}^{(l-1)}\right] \\
\alpha_{v, r, u}^{k} &=\frac{\exp \left(\operatorname{LeakyReLU}\left(S_{u, r}\right)\right)}{\sum_{\left(r^{\prime}, u^{\prime}\right) \in \mathcal{N}_{v}} \exp \left(\operatorname{LeakyReLU}\left(S_{u^{\prime}, r^{\prime}}\right)\right)}
\end{aligned}
$$

最后，实体 $v$ 的嵌入通过组合消息表示 $m_{v}^{k}$ 及其在层 $(l-1)$ 中的嵌入来更新：
$$
E_{v}^{(l)}=\operatorname{LayerNorm}\left(\bigoplus_{k=1}^{K} m_{v}^{k}+E_{v}^{(l-1)}\right)
$$
其中 LayerNorm 代表层标准化（$\mathrm{Ba}$, Kiros, and Hinton 2016）。 $\bigoplus$ 表示连接，$K$ 是注意力头的数量。

初始实体嵌入 $E^{(0)}$ 和关系嵌入 $R$ 是从我们的语言模块生成的，这将在“解决循环依赖”一节中介绍。然后，来自最后一个 GAT 层的输出实体嵌入用作最终实体表示 $E^{\mathrm{KM}}$。请注意，知识图可能非常大，使得对所有实体的嵌入更新变得难以处理。因此，我们遵循小批量设置（Hamilton、Ying 和 Leskovec 2017）：给定一组输入实体，我们执行邻域采样以生成它们的多跳邻居集，并且我们仅计算实体和关系的表示嵌入更新。

### 语言模块
语言模块 (LM) 的目标是对文本数据进行建模并学习上下文感知表示。语言模块可以是任何语言理解模型，例如BERT（Devlin 等人，2019 年）。在这项工作中，我们使用预训练模型 RoBERTa-base (Liu et al. 2019b) 作为语言模块。

#### 解决循环依赖
在我们的框架中，知识和语言模块互惠互利：
- 语言模块 LM 输出上下文感知嵌入，以在给定描述文本的情况下初始化知识图中实体和关系的嵌入；
- 知识模块（$\mathrm{KM}$）为语言模块输出基于知识的实体嵌入。

然而，存在一个循环依赖，它阻止了该设计中的计算和优化。为了解决这个问题，我们提出了一个分解的语言模块，它包括两个语言模型：$\mathrm{LM}_{1}$ 和 $\mathrm{LM}_{2}$。我们使用 RoBERTa 的前 6 层作为 $\mathrm{LM}_{1}$，其余 6 层作为 $\mathrm{LM}_{2}$。计算过程如下：
1. $\mathrm{LM}_{1}$ 对输入文本 $\mathbf{x}$ 进行操作并生成上下文嵌入 $Z$。
2. $\mathrm{LM}_{1}$ 为 $\mathrm{KM}$ 给定描述文本生成初始实体和关系嵌入。
3. KM 生成其输出实体嵌入，与 $Z$ 组合并发送到 $\mathrm{LM}_{2}$。
4. $\mathrm{LM}_{2}$ 产生 $\mathbf{x}$ 的最终嵌入，其中包括上下文信息和知识信息。


详细地说，在步骤 1 中，假设上下文 $\mathrm{x}$ 嵌入为 $X^{\text {embed }}。 \mathrm{LM}_{1}$ 将 $X^{\text {embed }}$ 作为输入并输出隐藏表示：
$$
Z=\mathrm{LM}_{1}\left(X^{e m b e d}\right)
$$

在步骤2中，假设$\mathbf{x}^{e_{j}}$是实体$e_{j}$的实体描述文本，对应的mention是$m^{e_{j}}=\left (e_{j}, s_{j}^{e}, o_{j}^{e}\right)$。 $\mathbf{L M}_{1}$ 采用 $\mathbf{x}^{e_{j}}$ 的嵌入并生成上下文嵌入 $Z^{e_{j}}$。然后，将位置 $s_{j}^{e}$ 和 $o_{j}^{e}$ 的嵌入平均值用作 $e_{j}$ 的初始实体嵌入，即
$$
E_{j}^{(0)}=\left(Z_{s_{j}^{e}}^{e_{j}}+Z_{o_{j}^{e}}^{e_{j}}\right) / 2
$$
知识图关系嵌入 $R$ 是使用其描述文本以类似的方式生成的。

在步骤 3 中，KM 计算最终的实体嵌入 $E^{\mathrm{KM}}$，然后将其与 $\mathrm{LM}_{1}$ 的输出 $Z$ 组合。详细地说，假设 $\mathbf{x}$ 中的提及是 $\mathbf{m}=\left[m_{1}, \ldots, m_{M}\right]$。 $Z$ 和 $E^{\mathrm{KM}}$ 在提及的位置组合：对于每个位置索引 $k$，如果 $\exists i \in\{1,2, \ldots, M\}$英石。 $s_{m_{i}} \leq k \leq o_{m_{i}}$,
$$
Z_{k}^{\text {merge }}=Z_{k}+E_{e_{m_{i}}}^{\mathrm{KM}}
$$
其中 $E_{e_{m_{i}}}^{\mathrm{KM}}$ 是来自 KM 的实体 $e_{m_{i}}$ 的输出嵌入。对于没有相应提及的其他位置，我们保留原始嵌入：$Z_{k}^{\text {merge }}=Z_{k}$。然后我们在 $Z^{\text {merge }}$ 上应用层归一化（Ba、Kiros 和 Hinton 2016）
$$
Z^{\prime}=\operatorname{LayerNorm}\left(Z^{\text {merge }}\right)
$$
最后，$Z^{\prime}$ 被输入 $\mathrm{LM}_{2}$。
在步骤 $4 中，\mathrm{LM}_{2}$ 对输入 $Z^{\prime}$ 进行操作并获得最终嵌入：
$$
Z^{\mathrm{LM}}=\mathrm{LM}_{2}\left(Z^{\prime}\right)
$$
为了更好地说明，这四个步骤在图 2 中用符号 $\otimes$ 标记。

#### 实体上下文嵌入内存
用于加速

许多知识图谱包含大量实体。因此，即使是一个句子，实体的数量加上它们的多跳邻居也可以随着图神经网络中的层数呈指数增长。因此，语言模块根据所有相关实体的描述文本即时计算上下文嵌入非常**耗时**。

为了解决这个问题，我们构建了一个实体上下文嵌入内存 $E^{\text {context }}$，来存储所有 KG 实体的初始嵌入。首先，语言模块预先计算所有实体的上下文嵌入并将它们放入内存中。知识模块只需要从内存中检索所需的嵌入而不是计算它们，即 $E^{(0)} \leftarrow E^{\text {context }}$。

但是，由于内存中的嵌入是从“旧”（初始）语言模块计算的，而训练期间的token嵌入是从更新的语言模块计算的，因此会出现不希望的差异。因此，我们建议每 $T(i)$ 步用当前语言模块更新整个嵌入内存 $E^{\text {context }}$，其中 $i$ 是内存被更新的次数（从 0 开始）。 $T(i)$ 设置如下：
$$
T(i)=\min \left(I_{i n i t} * a^{\lfloor i / r\rfloor}, I_{\max }\right)
$$
其中 $I_{i n i t}$ 是第一次更新之前的初始步数，$a$ 是更新间隔的增加比率。 $r$ 是当前更新间隔的重复次数。 $I_{\max }$ 是更新之间的最大步数。 $\lfloor\cdot\rfloor$ 表示向下取整的操作。在我们的实验中，我们设置$I_{i n i t}=10, a=2, r=$$3, I_{\max }=500$，$T$对应的序列是$[10,10,10,20 ,20,20,40,40,40, \ldots, 500,500]$。请注意，我们选择 $a>1$ 是因为模型参数通常随着训练的进行而变化较小。

此外，我们提出了一个动量更新，以使 $E^{\text {context }}$ 的演化更加平滑。假设$\mathrm{LM}$新计算的embedding memory为$E_{n e w}^{c o n t e x t}$，则更新规则为：
$$
E^{\text {context }} \leftarrow m E^{\text {context }}+(1-m) E_{\text {new }}^{\text {context }}
$$
其中 $m \in[0,1)$ 是动量系数，在实验中设置为 $0.8$。

这种内存设计在预训练期间将我们的模型加速了大约 15x，同时保持了实体上下文嵌入的有效性。出于效率考虑，我们仅在微调期间使用关系嵌入。

### 预训练
在预训练期间，知识模块和语言模块都基于下面列出的几个自监督学习任务进行了优化。所有训练任务的示例如图 2 所示。

在每个预训练步骤中，我们首先对一批根实体进行采样，并对每个根实体执行随机游走采样。采样的实体被送入 KM 用于以下两个任务。

- 实体类别预测。知识模块被训练以根据输出实体嵌入 $E^{\mathrm{KM}}$ 预测实体的类别标签。该任务已被证明在预训练图神经网络中是有效的（Hu et al. 2020）。损失函数是多类分类的交叉熵，记为$\mathcal{L}_{c}$。

- 关系预测。 $\mathrm{KM}$ 也被训练来基于 $E^{\mathrm{KM}}$ 预测给定实体对之间的关系。损失函数是多类分类的交叉熵，记为$\mathcal{L}_{r}$。

然后，我们为以下两个任务统一采样一批文本序列及其实体。

- Masked token 预测。与 BERT 类似，我们随机mask序列中的token，并根据语言模块的输出 $Z^{\mathrm{LM}}$ 预测原始token。我们将损失表示为 $\mathcal{L}_{t}$
> 与Bert MLM任务类似

- Masked entity 预测。语言模块也被训练来预测给定提及的相应实体。对于输入文本，我们随机删除 $15\%$ 的提及 $\mathbf{m}$。然后对于每个删除的提及 $m_{r}=\left(e_{r}, s_{r}, o_{r}\right)$，模型根据提及的嵌入预测掩码实体 $e_{r}$ .详细地，它预测在 $E^{\text {context }}$ 中嵌入最接近 $q=g\left(\left(Z_{s_{r}}^{\mathrm{LM}}+ Z_{o_{r}}^{\mathrm{LM}}\right) / 2\right)$，其中 $g(x)=\operatorname{GELU}\left(x W_{1}\right) W_{ 2}$ 是一个转换函数。 GELU 是 (Hendrycks and Gimpel 2016) 提出的激活函数。由于实体的数量可能非常大，我们使用 $e_{r}$ 的邻居和其他随机采样的实体作为负样本。损失函数 $\mathcal{L}_{e}$ 是基于 $q$ 和每个候选实体嵌入之间的内积的交叉熵。图 2 显示了一个具体示例，其中提及“Earth”没有在输入文本中标记，因为它被屏蔽(masked)了，提及“Earth”链接到实体“Q2: Earth”。

### 微调
在微调期间，我们的模型支持使用预训练期间使用的知识图谱或具有以前未见过的实体（我们假设自定义域带有 NER 和实体链接工具，可以注释文本中的实体提及。这些系统的培训超出了这项工作的范围）的新颖自定义知识图谱。如果使用自定义 $\mathrm{KG}$，实体上下文嵌入内存由预训练的语言模块使用新的实体描述文本重新计算。

我们的模型还支持 only-KG 的任务，例如实体分类或链接预测，其中输入数据是实体描述文本和没有上下文语料库的 KG。在这种情况下，语言模型 1 将实体描述文本作为输入和输出实体嵌入到下游任务的知识模块（即图神经网络）中。不会使用语言模型 2。

在这项工作中，出于效率考虑，我们不会在微调期间更新实体上下文内存。我们还使用预训练的语言模型计算关系上下文嵌入内存。

## Experiment

实现细节与参数：见原文

### 下游任务
Few-shot Relation Classification 小样本关系分类


![[Pasted image 20220720030220.png]]
![[Pasted image 20220720030452.png]]

实体识别

![[Pasted image 20220720030442.png]]

## 其他笔记参考
[2020-12-27_wl_dew的博客-CSDN博客](https://blog.csdn.net/wl_dew/article/details/111798799)

