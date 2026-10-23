# 面向 72 分反馈的论文与代码研究

研究日期：2026-10-23。目标是改进第八届全国高校计算机能力挑战赛「人工智能挑战赛」胸片十标签分类任务。此前 `final_bundle/submission.csv` 的真实反馈为 **72 分、432/488**；官方分项尚未知，不能将 72 直接写成已确认的测试 AUC。

本次先停止旧训练队列，再查阅论文和作者/维护者代码。已取得 **21 篇论文全文 PDF**，重点阅读其中与任务直接相关的方法、训练设置和消融；另核对 Compositional AUC 的官方实现和文档，其论文全文访问受限。代码覆盖 **21 个独立仓库、44 个文件，其中 27 个 Python 文件**。这不表示逐行审完所有仓库；下表区分核心实现阅读与说明/许可筛查。来源的 commit、文件 blob SHA 见 [source_manifest.json](research/source_manifest.json)，论文文件摘要见 [paper_manifest.json](research/paper_manifest.json)。第三方源码与论文全文仅保留在本地忽略目录，不随本项目重新发布。

## 与本赛直接相关的约束

[官方网页](https://www.ncccu.org.cn/index/Paper/case1.html)明确优先按 Macro-AUC 排名，只有相同才比较 Macro-F1。因此调 Gate、单类温度缩放等单调变换通常不能改善主要指标。CXR-LT 的 mAP 排名与本赛不能直接对照。

[10 月 23 日规则 PDF](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf)允许非医学、可公开下载且许可满足要求的通用预训练权重；禁止医学预训练、医学自监督续训、外部医学数据及测试集拟合。NC 或 gated 权重不接受。部署权重与必要配置合计不超过 500MB，官方 T4 上按 5 张预热、100 张 batch=1 串行、3 轮测量平均耗时，含读图、处理和写出。论文模型能否复现，还必须同时考虑这些条件。

## 论文结论与采用范围

| 论文/报告 | 读到的关键依据 | 对本项目的决定 |
|---|---|---|
| [CXR-LT 2026 总结，2026](https://arxiv.org/abs/2604.15555) | 比较多中心、长尾与零样本任务；强方法使用 DB loss、采样、查询分类头、视角分支和集成。若干方法有医学预训练。 | 借鉴优化与分视角评估；不能照搬冠军权重或把其 mAP 当本赛 AUC。 |
| [Projection-Aware CXR-LT，2026](https://arxiv.org/abs/2604.02185) | AP/PA 与侧位分支，512 输入、PCAM/CaiT/Swin、固定 TTA。 | 本赛已有 ViewPosition，可先用已提供视角信息，无需额外训练 router；其 ConvNeXt V2 上游许可另行限制。 |
| [CXR-LT 2024 总结，2025](https://arxiv.org/abs/2506.07984) | 汇总多视角、多尺度、ASL 和外部医学预训练。不同指标的最优团队并不一致。 | 按本赛 Macro-AUC 与预算筛选，保留原始病人划分。 |
| [CheXFusion，ICCVW 2023](https://arxiv.org/abs/2308.03968) | 单图骨干后冻结，训练多图 Transformer + ML-Decoder。其表 1 中 7:3 概率融合与特征融合的 AUC 均为 0.847，后者主要改善 mAP。 | 候选为轻量检查级特征融合；不能预设一定提高 AUC。外部医学数据、自训练与 1024 大模型配置不照搬。 |
| [CheXtransfer，CHIL 2021](https://arxiv.org/abs/2101.06871) | 比较 16 类 ImageNet 模型；ImageNet 排名不能预测胸片表现，架构家族与初始化需要本地对照。 | 保留 DenseNet121 作为有依据的轻量对照，避免只按模型新旧/大小排序。 |
| [ASL，ICCV 2021](https://arxiv.org/abs/2009.14119) | 正负标签采用不同 focusing，裁剪容易负样本；官方训练配方有 EMA、无 bias/BN decay 等选择。 | ASL 已有，但须单独对照 BCE，并明确 Mixup 的非线性损失语义。不能一次叠加所有配方后宣称 ASL 有效。 |
| [ML-Decoder，CVPR 2022](https://arxiv.org/abs/2111.12933) | 从空间特征通过 cross-attention 查询类别，移除冗余 self-attention，支持 grouped decoding。 | 十类别可用小维度、十查询头，先与相同骨干 GAP 对照；现有 `_cap` 不是 ML-Decoder。 |
| [Query2Label，2021](https://arxiv.org/abs/2107.10834) | 学习 label query 从特征图提取类别相关信息。 | 不把简单 CAM/加权池化冒称 Q2L；比较时需固定骨干与训练设置。 |
| [Distribution-Balanced Loss，ECCV 2020](https://arxiv.org/abs/2007.09654) | 重采样会通过标签共现改变其他类的频率；其重平衡不等于独立逆频率加权。 | 现有 sqrt 正类权重不叫 DB loss。若实现，频率仅从实际训练检查计算，并检查零阳性样本。 |
| [Large-scale Robust Deep AUC Maximization，ICCV 2021](https://arxiv.org/abs/2012.03173) | AUC margin 的 min-max 优化，与 BCE 的分类目标不同。 | 有依据考虑 BCE 后 AUC 微调，但必须连同正负采样、PESG 和辅助状态一起验证。 |
| [Compositional AUC，ICLR 2022](https://openreview.net/forum?id=gPvB4pdu_Z)；[官方实现文档](https://docs.libauc.org/api/libauc.losses.html) | CE/AUC 更新交替，并有对应 PDSCA。全文下载被拒，本次未声称阅读全文。 | 仅作为后续候选，不把它实现成简单的 BCE + AUC 加权和。 |
| [Hill / SPLC，2022](https://arxiv.org/abs/2112.07368) | 面向缺失阳性标签，降低可疑负标签影响；SPLC 会在训练中改标签。 | 本赛未提供缺失/不确定标记，不能据此认定所有负类都缺失。先不自动纠正官方标签。 |
| [Partial Labeling CSL，CVPR 2022](https://arxiv.org/abs/2110.10955) | 需要区分已标注正/负与未标注标签。 | 当前 CSV 是阳性 ID 列表，没有独立 unknown mask，不能直接套其三态标签编码。 |
| [LP-FT，ICLR 2022](https://arxiv.org/abs/2202.10054) | 先线性探测再微调可以在其分布偏移实验中减轻预训练特征破坏。 | 新增分类头预热选项；短预热仅受此启发，并非论文完整复现，不能承诺胸片增益。 |
| [DINOv2，2023 / TMLR 2024](https://arxiv.org/abs/2304.07193) | 通用 LVD-142M 特征支持冻结骨干、线性分类器；官方通用模型卡为 Apache-2.0。 | 优先研究小型通用 DINOv2 的监督探测。必须锁定 LVD 权重；同仓库后增的 XRay-DINO 以及 RAD-DINO 不采用。 |
| [DINOv3，2025](https://arxiv.org/abs/2508.10104) | 强通用特征，但官方获取流程要求接受其许可。 | 官方权重 gated，与本赛规则冲突，排除；镜像不会消除来源限制。 |
| [ConvNeXt，CVPR 2022](https://arxiv.org/abs/2201.03545) | 现代卷积结构可作高效骨干，不能仅凭 Transformer 名称判断优势。 | 通用 V1 权重可作为审计后的候选；本次尚未新增到允许下载清单。 |
| [ConvNeXt V2，CVPR 2023](https://arxiv.org/abs/2301.00808) | FCMAE 与 GRN 联合设计；官方 README 区分代码 MIT 与预训练/微调权重 CC-BY-NC。 | 官方 V2 权重排除，不把代码 MIT 当作权重许可。 |
| [CheXWorld，CVPR 2025](https://arxiv.org/abs/2504.13820) | 医学影像自监督学习局部结构、全局布局和外观变化；下游有 attentive pooler。 | 可研究分类头结构；医学权重及其自监督训练流程不用于本赛。 |
| [RAD-DINO，2024/2025](https://arxiv.org/abs/2401.10815) | 预训练数据是胸片，强调医学图像表征。 | 与通用 DINOv2 区分，医学权重禁止使用。 |
| [CheXagent，2024](https://arxiv.org/abs/2401.12208)；[ChEX，ECCV 2024](https://arxiv.org/abs/2404.15770) | 使用医学指令/多任务或区域标注，任务还包含报告生成与定位。 | 做任务与数据来源筛查；不将医学 VLM 权重或文本/区域标注引入本赛。 |

这里的“相关 SOTA”包括其论文报告的强方法和挑战赛领先方案，不代表截至研究日存在一套能在本赛合法直接复现的统一 SOTA。尤其不能用论文高分替代本赛实测。

## 21 个仓库的实际核对范围

下列路径可在对应仓库中查找；[文件清单](research/source_manifest.json)记录精确版本。

| 仓库 | 已阅读的代码/材料 | 结论 |
|---|---|---|
| [dongkyunk/CheXFusion](https://github.com/dongkyunk/CheXFusion) | `model/layers.py`、`cxr_model.py`、`loss.py`、`dataset/cxr_dataset.py`、配置 | 两阶段融合；发现待修正的张量轴约定，见下文。 |
| [Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL) | 损失、`train.py`、README | 硬标签公式与现实现一致；原始 reduction 与学习率不可分开照搬。 |
| [Alibaba-MIIL/ML_Decoder](https://github.com/Alibaba-MIIL/ML_Decoder) | `src_files/ml_decoder/ml_decoder.py` | 固定 query、cross-attention、group FC；采用思路需记录维度差异。 |
| [SlongLiu/query2labels](https://github.com/SlongLiu/query2labels) | `lib/models/query2label.py` | 可学习 label query、input projection、groupwise classifier。 |
| [wutong16/DistributionBalancedLoss](https://github.com/wutong16/DistributionBalancedLoss) | `mllt/models/losses/resample_loss.py` | 共现重平衡、负 logit 缩放；旧 mmcv/CUDA 假设需改写。未把它当作现有正类加权。 |
| [Optimization-AI/LibAUC](https://github.com/Optimization-AI/LibAUC) | 多标签示例、`auc.py`、`pesg.py`、`sampler.py` | 旧示例 API 与当前版本有差异；需处理辅助状态、每类正负样本与恢复训练。 |
| [xinyu1205/robust-loss-mlml](https://github.com/xinyu1205/robust-loss-mlml) | Hill/SPLC 损失、说明、许可 | SPLC 会改变训练标签；不在缺乏依据时启用。 |
| [Alibaba-MIIL/PartialLabelingCSL](https://github.com/Alibaba-MIIL/PartialLabelingCSL) | `partial_asymmetric_loss.py`、README | 代码三态编码还需与论文约定转换，不能直接复用二值标签。 |
| [Glourier/MICCAI2024-ZeroShotCXR](https://github.com/Glourier/MICCAI2024-ZeroShotCXR) | `arch/vit_mv.py`、`main_task1_mv.py`、README | 特征形状 B×V×D，明确 batch_first 与 mask；入口有调试断点，不能作为可直接运行的完整交付。 |
| [AnanyaKumar/transfer_learning](https://github.com/AnanyaKumar/transfer_learning) | `baseline_train.py`、README | 线性探测时不仅冻结梯度，还保持骨干 eval。 |
| [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | 通用模型卡、`hub/backbones.py`、线性评估入口 | 通用 LVD 与医学变体必须分开；官方单标签分类评估不能直接替代本赛十标签 AUROC。 |
| [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | 许可与官方获取说明 | 本次为资格筛查，未采用权重或训练实现。 |
| [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | `models/convnext.py`、README | V1 架构/来源候选，尚未下载新权重。 |
| [facebookresearch/ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2) | README 与完整 LICENSE | 代码与权重许可不同，官方权重排除。 |
| [LeapLabTHU/CheXWorld](https://github.com/LeapLabTHU/CheXWorld) | attentive pooler、胸片增强、微调说明、README | 医学预训练排除；其强裁剪/擦除并非无条件适合当前标签。 |
| [mlmed/torchxrayvision](https://github.com/mlmed/torchxrayvision) | `scripts/train_utils.py` | 多标签/缺失标签训练处理可参考，医学模型权重不采用。 |
| [stanfordmlgroup/MoCo-CXR](https://github.com/stanfordmlgroup/MoCo-CXR) | README 的流程、预训练来源和评估说明 | 本次为方法/数据资格筛查，医学自监督续训排除。 |
| [Stanford-AIMI/CheXagent](https://github.com/Stanford-AIMI/CheXagent) | README 的模型、数据与任务说明 | 本次为来源/任务筛查，未声称复现。 |
| [philip-mueller/chex](https://github.com/philip-mueller/chex) | README 的训练阶段、数据依赖说明 | 本次为来源/任务筛查，未声称复现。 |
| [pytorch/vision](https://github.com/pytorch/vision) | classification 的 `utils.py`、`transforms.py` | EMA 与 Mixup 参考代码主要是单标签分类，移植时需重新验证目标定义。 |
| [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | `timm/utils/model_ema.py` | EMA 浮点缓冲平滑、整数缓冲复制与现实现一致；没有发现可据此解释 72 分的错误。 |

## 已用代码验证的发现

这些均为合成张量测试，不接触比赛图像，不能当作泛化成绩。[原始数值记录](research/implementation_checks.json)随文公开。

1. **现有 ASL 不是公式抄错。** 与官方 `gamma_neg=4, gamma_pos=0, clip=.05`、相同 reduction 的硬标签实现比较，损失与梯度最大差均为 **0.0**。
2. **ASL 的两种 Mixup 目标不同。** 同一合成 logits 和 λ=0.3：软标签代入 ASL 得 6.162439，两个硬标签损失加权得 7.873755，梯度最大差 0.071222。BCE 两种写法的损失差为 0，梯度差约 1.86e-9。这证明语义差异，不证明哪种写法一定有更高验证 AUC；旧的纯 ASL 训练未启用 Mixup，不能据此归因其 72 分。
3. **上游融合代码需要轴审查。** 所读 CheXFusion `model/layers.py` 使用默认 sequence-first 的 `TransformerEncoderLayer`，却传入形如 B×S×D 的张量。按此约定构造最小例，改变第二个检查使第一个检查的输出最大改变 1.484786；显式 `batch_first=True` 后为 **0.0**。这里只复现张量约定问题，不据此判断作者实际训练版本或论文结果。任何采用的检查级融合都必须通过 batch=1/多样本、改变其他检查、padding、视角排列的独立性测试。
4. **AUC loss 不能接受任意软标签。** 已读 LibAUC 用 `target==1` 与 `target==0` 构造正负掩码；0.3/0.7 都不属于两者。条件均值版本可能得到空掩码/除零；不能直接接现有软标签 Mixup。当前多标签辅助状态还需显式审查 checkpoint 保存和设备迁移，不能只保存模型参数。
5. **缓存限制了分辨率实验。** 当前缓存先把原图变成最长边 448 的 JPEG；把模型输入改成 512/1024 只是放大缓存。真正高分辨率实验需重建原图预处理，并使原图部署与缓存路径一致。

本次直接修改：新增 `mixed_target_loss`，默认使用两份硬标签损失加权；保留 `soft_target` 以复现旧语义，旧 checkpoint 恢复时继承其语义。新增分类头预热，期间骨干参数、BN 统计、dropout 均冻结；进入全量微调时恢复原可训练参数。新增独立的 `split_seed`，避免换初始化种子时也换患者留出，造成对照混杂。

验证：`python -m tools.check_training_semantics` 在训练环境 PyTorch 2.8 和目标环境 PyTorch 2.10 均通过，覆盖 BCE/加权 BCE 等价性、ASL 插值及梯度、预热期间骨干/BN 不变和恢复训练。未因此改写任何已交付权重或 CSV。

## 研究后的实验顺序

旧的“EfficientNet→高分辨率+ASL→大范围集成搜索”队列停止使用。先固定患者留出与有限对照，再扩展架构。

| 顺序 | 对照设计 | 采用条件 |
|---|---|---|
| 1 | 同一 ImageNet DenseNet121、同一 split_seed=2030、同一模型 seed=2036；BCE 基线 vs 仅增加 2 轮分类头预热 | 内部患者留出 Macro-AUC 为主；官方验证先报告。两轮预热是待验证工程选择。 |
| 2 | 在相同骨干/分辨率/视角采样上比较 BCE 配方与 ASL 配方；先不启用 Mixup，防止额外交互 | 同时报告十类 AUC、患者簇配对 bootstrap；不以单个验证峰值下结论。正类加权开关视为配方的一部分。 |
| 3 | 相同骨干 GAP vs 小型 label-query 分类头；再比较固定概率池化与检查级特征融合 | 训练、校验均以检查为单位；具备缺视角与 batch 独立性测试；总大小和原图计时满足预算。 |
| 4 | 通用 LVD DINOv2-S 冻结特征探测，再考虑监督微调 | 完成独立权重链接/哈希/许可审计；只在官方 train 上训练任务头，不做医学自监督续训。 |
| 5 | 固定的少量优胜候选做 BCE→AUC 微调、真实 448 及小型集成 | 正负采样/辅助状态已验证；收益需要患者留出支持；最终在 PyTorch 2.10 batch=1 独立复现并测时。 |

旧方案的 0.788139 与新 ImageNet ResNet50 的 0.793796 都只是本地官方验证成绩，后者增加约 0.00566；尚无证据表明隐藏榜单已改善。不能因为文献研究完成或单项测试通过，就替换可提交文件并宣称涨分。旧包保留，后续新包须附实际训练与独立复现结果；本项目不执行网站提交。
