# 与本赛任务对齐的强方法复现

更新日期：2026-10-26。第二轮真实测试 AUC 为 **0.7678**，对应排名 338/593；离线 valid AUC 为 0.811122。下一阶段目标为 valid AUC ≥0.90，但尚未达成。线上反馈详见 [记录](ONLINE_FEEDBACK.md)。

## 数据是否相同

本赛的十标签、AP/PA/侧位、每次检查一组标签、患者级隔离和元数据字段，与 MIMIC-CXR/CheXpert 体系高度接近。官方包中的患者、检查与图像标识已改为其他格式，未附来源说明；不能仅凭相似结构宣称已确认它是某个公开数据集的同一子集。本赛 18,513 次训练检查、30,344 张训练图像，也不是论文使用的完整公开训练集规模。

因此应同时核对标签子集、检查级/图像级单位、患者划分、训练规模、可用模态、权重来源和评价指标。下面这些是明确的参考对象，不把不同划分的论文分数当作本赛保证。

| 明确复现对象 | 与本赛的对应关系 | 原方法关键组成 | 允许采用的范围 |
|---|---|---|---|
| [CheXFusion，ICCVW 2023](https://arxiv.org/abs/2308.03968)，[作者当前代码](https://github.com/dongkyunk/CheXFusion) | MIMIC-CXR 衍生 CXR-LT；多标签、长尾、检查内多视角。原文为26标签，排行榜主指标mAP。 | ImageNet ConvNeXt + ML-Decoder；第一阶段单图训练，第二阶段冻结骨干并学习 Transformer 融合；加权 ASL。表1单图AUC 0.833、最佳固定多视角和CheXFusion均0.847，融合主要进一步改善mAP。 | 复现两阶段结构、ML-Decoder和加权ASL；仅官方train。原方案外部医学数据、自训练不采用。不能称本项目现有概率集成已经复现了它。 |
| [CaMCheX，ML4H / PMLR 297](https://proceedings.mlr.press/v297/sloan26a.html)，[代码](https://github.com/phillipSloan/CaMCheX)，[全文](https://arxiv.org/html/2511.09581v1) | MIMIC-CXR与CXR-LT，明确以检查为单位处理多个视角。 | 正位/侧位编码器、Transformer融合，并结合临床指征与生命体征；原训练引入CheXpert/NIH/VinDR等。 | 可核对图像多视角结构及消融；不能使用其医学预训练权重、额外医学数据或本赛未提供的临床文本/生命体征。 |
| [Resolution-dependent self-supervised transfer，Communications Medicine 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13558615/)，[作者代码](https://github.com/tayebiarasteh/vit-med) | 包含MIMIC-CXR、CheXpert十标签评测和非医学初始化对照；完整队列大于本赛，标签仍需逐项对齐。 | ImageNet-21K/22K ConvNeXt-B、DINOv2 ViT-B等；224/512对照，Adam 1e-5、加权BCE、轻微旋转与翻转。 | 以通用ImageNet-22K ConvNeXt-B及LVD-142M DINOv2作论文配方对照。DINOv3 gated权重不采用。作者README的配置路径与实际仓库有差异，需按实际代码核对。 |
| [Deep AUC Maximization，ICCV 2021](https://arxiv.org/abs/2012.03173)，[官方实现](https://github.com/Optimization-AI/ICCV2021_DeepAUC)，[当前LibAUC说明](https://docs.libauc.org/api/libauc.losses.html) | 胸片多标签与AUC目标直接对应；原CheXpert比赛标签/划分不同，不能照搬其榜单数值。 | BCE预训练后以AUC margin目标及PESG优化，显式维护辅助变量和正负采样。 | 复现损失、优化器、采样及可恢复辅助状态；不能用普通Adam+BCE加一个任意排序项后声称完整复现PESG。 |

## 已查明的实现差异

此前已读论文和作者实现，但大部分实验仍为单图骨干、普通GAP或自有池化头、既有ASL、概率融合；这没有落实CheXFusion最重要的第二阶段，也不是完整论文复现。

作者加权 ASL 的正类聚焦参数为 **1**、负类为 **4**，类别正/负权重分别为 `exp(1-p)`、`exp(p)`，focusing项参与梯度，最后对样本和类别取均值。本项目此前 `AsymmetricLoss` 的正类聚焦为0、focusing项停止梯度、对类别求和，没有这套正负类别权重。二者不能混为一个配方。

官方 ML-Decoder 使用固定查询、去除冗余 self-attention 的 decoder 与 group-wise readout；本项目已有的可训练标签查询头只是受 Query2Label 启发，不能冒称已经复现 ML-Decoder。

CheXFusion已读代码中还存在需要适配的新旧PyTorch接口、sequence-first/batch-first约定及padding mask问题。复现将保留论文关键结构，并对必要修正独立验证：改变同批其他患者不应改变当前患者预测，填充图不能参与注意力，缺失侧位可正常处理。代码接口错误不作为放弃整个方法的理由。

## 下一阶段按顺序落实

1. 完成与作者实现数值/梯度对应的加权ASL和ML-Decoder，用同一骨干对照GAP与ML-Decoder，记录必要接口修正。先复现可在12GB设备运行的384配置，再根据真实速度/显存决定分辨率。
2. 学习检查级多视角特征融合，同时保留同骨干固定聚合对照；训练单位为检查，掩码和患者独立性有明确检查。
3. 复现上述2026胸片论文中通用ImageNet-22K ConvNeXt-B与DINOv2初始化的对照。两份公开非医学权重已实际下载、计算SHA256、在CPU完成多输入尺寸前向；这些只是初始化与接口证据，不是新的验证成绩。
4. 对明确的BCE优胜模型，再进行独立的AUC优化对照。每个实验写清论文、作者文件/版本、已复现模块、受规则或硬件限制的变化、患者留出和官方valid结果。

当前36轮队列已经完成并归档，最佳组合AUC0.811148，相较0.811122没有实质改进；完整记录见 [长训练结果](LONG_TRAINING_20261026.md)。现有0.811122冻结方案保留作每阶段对照。

## 2026-10-26 已落实的代码与有限实验

- `src/ml_decoder.py`：来自 MIT 许可的 Alibaba-MIIL 原始实现，保留固定query、移除self-attention的decoder和分组输出。只适配现代PyTorch接口及有效padding mask。原始作者文件SHA256写入核对结果。
- `src/paper_losses.py`：按CheXFusion作者ASL公式实现可保存的类别权重缓冲区，训练实际样本统计类别数量，患者留出不参与统计。
- `src/reference_models.py`：单图ConvNeXt+ML-Decoder，以及冻结骨干、stride-2空间投影、两层TransformerEncoder、ML-Decoder的检查级融合。修正作者NCHW/NHWC位置编码轴、batch轴及padding语义；缺失图像不进入CNN。
- `src/paper_augmentation.py`：锁定Albumentations1.3.1，落实作者RandomResizedCrop、翻转、ShiftScaleRotate、亮度对比度、几何形变、噪声/模糊、Lanczos4缩放与标准化的原始顺序。仍接收项目既有448方形填充缓存，valid/deploy使用既有确定性预处理，这是明确适配。
- `src/paper_studies.py` / `tools/train_chexfusion_study.py`：训练按检查读取最多四张图，显示有效掩码；继承第一阶段的患者留出，CNN冻结且保持eval，未读取test。研究检查点的检查级部署入口尚未接入通用提交工具，不能直接称为可交付新方案。

在PyTorch2.8和2.10 CPU上，ML-Decoder与原作者层的输出、输入梯度和参数梯度差为0；加权ASL的损失与梯度差为0并通过有限差分检查。位置编码与上游正确NCHW接口逐值相同。已验证batch中其他患者变化不影响当前患者、任意填充值不影响有效预测、冻结骨干无梯度、融合模块有梯度。两组第一阶段和第二阶段的真实训练入口均完成合成CPU数据检查。最初一次合成检查因Windows空CUDA_VISIBLE_DEVICES仍使用GPU，证据保留；改为`-1`并显式断言CPU后重新通过。这些都是正确性检查，不能当作比赛性能。结果见 [核对汇总](research/reference_module_checks_20261026.json)。

`tools/run_matched_sota.py` 固定以下队列，启动前要求Git工作区干净，保存源代码/CSV摘要并在各阶段核对：

| 阶段 | 固定设置 | 所回答的问题 |
|---|---|---|
| 1A | ImageNet-1K ConvNeXt-T + ML-Decoder + 作者加权ASL，384，batch16，7轮 | 论文单图骨干与分类头能否改善本赛任务 |
| 1B | 相同CNN初始化、增强、损失、训练患者、数据顺序种子，改GAP+线性头，7轮 | 整体分类头（含位置编码）的对照，而非混用不同初始化 |
| 1C | 通用ImageNet-22K ConvNeXt-B + ML-Decoder，同样384与7轮 | 更强非医学初始化的论文相关硬件适配；不能与1A单独归因某一模块 |
| 2 | 仅按训练内患者留出AUC选择1A/1C骨干，冻结后训练多视角融合6轮 | 学习检查级融合相较固定单图聚合是否有效 |

第一阶段固定AdamW 3e-5、weight decay0.01、无学习率预热、余弦、无Mixup、无EMA、无单独分类头预热，全视角按图遍历。留出官方train的10%患者，split seed2030；模型seed2045，数据加载器种子与分类头初始化独立。valid只评估和保存候选，不参与梯度；阶段2的骨干选择按训练内留出完成。与作者7/6轮建立初始配方对应，本赛30344张train少于论文264849张，不能把相同epoch数解释为同等训练量；后续根据真实曲线决定是否需要延长。

公开非医学ConvNeXt-B ImageNet22K权重SHA256为 `b007c30d8173f247df47699bd991f534d0ba5284c49929ac71444f7d82d8cfcf`，实际本地载入及多尺寸有限前向完成。384/batch16完整AMP反向、EMA和AdamW短资源探测峰值allocated7494MiB、reserved7734MiB，约89.9图/秒；这不是AUC或官方T4延迟。DINOv2-register权重也已审计，尚未进入此有限训练队列。

复现命令：先安装`requirements-reference-train.txt`，再运行`python -m tools.run_matched_sota --cache <448缓存目录> --execute`。队列源代码冻结期间不修改训练实现。未执行网站提交；valid≥0.90目标尚未达到。
