# CXR-MLC-CAD

**改进状态（2026-10-25）**：上一版提交反馈为 72 分、432/488；下方旧交付的 0.788139 是本地验证 AUC。三组 ImageNet DenseNet121 对照、统一患者留出重算及六轮全 train 微调均已完成。ImageNet EfficientNetV2-S 320 已完成九轮并早停，训练环境单模型验证 AUC **0.801888**，目前在做 448 微调。PyTorch 2.10、batch=1 下，已有固定全局三模型融合验证 AUC **0.802912**；患者分组交叉验证选择收缩强度的逐类融合，完整验证集重拟合 AUC **0.806420**、调阈值 F1 **0.477395**，五次五折平均 OOF AUC **0.801142**。这些是有选择偏差的验证估计。训练结束后会在统一部署导出上比较新旧模型，并对每种 TTA 模式的最佳全局三模型各做一次固定逐类融合验证，再进行低负载测时；尚未生成新 CSV。已完成覆盖 21 个仓库、44 个文件及相关论文的研究，也实现了待实验的疾病查询头。研究结论见 [研究报告](docs/RESEARCH_REVIEW.md)，训练、权重平均、集成和患者对比证据见 [开发记录](docs/DEVELOPMENT_20261025.md)，此前实验见 [第二轮记录](docs/ROUND2.md)。旧交付包保留，尚无新的榜单成绩或第二轮交付包。

2026 年第八届全国高校计算机能力挑战赛 **人工智能挑战赛**：基于胸部 X 光片的多标签疾病分类。使用 PyTorch，在官方数据上训练，对每次检查输出 10 类概率。

比赛规则来源：[官方网站](https://www.ncccu.org.cn/index/Paper/case1.html)（最后核对日期 2026-10-23）。区域赛/省赛截止时间为 2026-11-13 23:59。此项目只生成提交材料，不自动向比赛网站提交。

已完成四组训练及 120 个部署配置的比较。最终采用 **ResNet18 类别空间注意力 + ASL 与 EfficientNetV2-B0 等权集成**，优先正位聚合，不使用翻转增强。

本地交付目录：`submission/final_bundle/`；初赛只上传其中的 **submission.csv**。完整结果见 [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md)。比赛数据、权重与逐例预测保留在本地，不上传公开仓库。

以下均为官方验证集 859 次检查上的 Python 3.12 / PyTorch 2.10 确定性部署结果；每个单模型展示其最佳聚合配置，隐藏测试分数未知：

| 模型 | 已训练轮数 / 选中轮次 | 输入 | 检查聚合 | Macro-AUC | 调阈值 Macro-F1 |
|---|---|---:|---|---:|---:|
| ResNet18 随机初始化基线 | 35 / 26 | 320 | 优先正位 | 0.775391 | 0.447217 |
| ResNet18 类别空间注意力微调 | 12 / 3 | 384 | 优先正位 | 0.779805 | 0.447914 |
| ResNet18 类别空间注意力 + ASL 微调 | 10 / 1 | 384 | 正位加权 | 0.780371 | 0.450580 |
| EfficientNetV2-B0 随机初始化 | 50 / 19 | 384 | 优先正位 | 0.781893 | 0.444824 |
| **最终两模型等权集成** | 上述 ASL / EfficientNet 最佳权重 | 384 | 优先正位 | **0.788139** | **0.455429** |

模型与聚合配置不同，上表不用于单独归因注意力模块或损失的收益。调阈值 F1 在同一验证集评估，可能偏乐观。最终模型共 **68,649,914 字节**；本机 RTX 4070 SUPER 上 60 张 SSD 原图、batch 1 测时平均 **55.82ms**、P95 **67.84ms**、最大 **72.71ms**。完整 2,218 张原图独立推理生成的 14,140 行概率与缓存生成结果完全一致，最大差值 **0.0**。这些是本机结果，**T4 / CUDA 12.8 和全测试集最坏延迟未实测**；未执行比赛网站提交。

## 数据集

用户将官方数据解压到 `raw_data/cxr_first/`。每个划分有一个 CSV，图像路径为 `<split>/<Subject_id>/<Study_id>/<Dicom_id>.jpg`。标签在 `Predict_class` 字段，多个类别用分号分隔。

| 划分 | 检查数 | 图像数 | 患者数 |
|---|---:|---:|---:|
| train | 18,513 | 30,344 | 11,055 |
| valid | 859 | 1,422 | 228 |
| test | 1,414 | 2,218 | 136 |

CSV 审计确认三个划分之间患者、检查、图像 ID 均无重叠；同一次检查的标签一致。测试集没有标签。比赛数据有使用及传播限制，**仓库不包含数据、逐例预测、模型权重或图像**；本地产物均在忽略列表内。只公开代码、方法与汇总指标。

| 类别 ID | 类别 |
|---|---|
| 1 | Enlarged Cardiomediastinum / 纵隔心影增大 |
| 2 | Pneumothorax / 气胸 |
| 3 | Consolidation / 肺部实变 |
| 4 | Pneumonia / 肺炎 |
| 5 | Edema / 肺水肿 |
| 6 | Cardiomegaly / 心影扩大 |
| 7 | Atelectasis / 肺不张 |
| 8 | Lung Opacity / 肺部阴影 |
| 9 | Pleural Effusion / 胸腔积液 |
| 10 | No Finding / 无异常发现 |

## 方案

第一轮未使用任何外部预训练权重；第二轮按官网新增 PDF 第 1 条，采用 torchvision 官方非医学 ImageNet 权重，保留下载链接、SHA-256、许可证与训练来源声明。不使用医学预训练权重或外部医学数据。以轻量 ResNet18 从零训练建立基线，支持 timm 的其他架构及不对称损失进行后续对比。每个训练 epoch 对每次检查随机选择一个视角，避免多图检查被过度加权；验证与测试使用所有图像并在检查层面聚合。使用轻微几何/灰度增强、AdamW、预热和余弦学习率、混合精度及 EMA 权重。模型以官方验证集检查级 Macro-AUC 选择，Macro-F1 单独报告。各类别阈值只影响 F1，不影响 AUC。

验证集只有 859 次检查，部分疾病阳性数较少，阈值优化后的 F1 是同一验证集上的估计，可能乐观。未经主办方评测，不声称测试集得分或排名。未声称实现 SOTA。

## 运行

```powershell
python -m pip install -r requirements.txt
python tools/prepare.py
python tools/pack_cache.py
python train.py --model resnet18 --size 320 --epochs 35 --batch-size 48 --out outputs/resnet18_s42
python predict.py --checkpoint outputs/resnet18_s42/best.pt --out submission
```

恢复训练：在相同配置的训练命令中添加 `--resume outputs/resnet18_s42/last.pt`。缓存为等比例缩放并补黑边的 448×448 灰度 JPG；模型输入复制为 RGB 后归一化。Windows 多进程入口已保护。随机种子默认 42；CUDA 性能优化可能引入非确定性。

机械硬盘建议先用 `python tools/pack_cache.py --out <SSD目录>` 合并缓存，再给 train.py / predict.py 传入 `--cache <SSD目录>`。压缩缓存通过内存映射共享，不要求为多个数据加载进程复制完整图像集。默认使用两个数据加载进程，适应 Windows 内存开销。

可选的 `resnet18_cap` 使用疾病独立的空间注意力池化，支持用 `--init <比赛数据训练的ResNet18权重>` 初始化后微调。它是参考类别注意力思想的简化实现，不宣称复现论文的完整图结构或结果。是否采用需由官方验证集的实际对比决定。

后续候选训练可指定 `--selection-aggregation auto`，在每轮验证中比较全图平均、最大值、优先正位和正位加权四种检查级聚合方式，再保存最佳 AUC 的权重及对应聚合方式。默认仍以全图平均选权重。选择使用验证标签，测试标签始终不可见；多次调参可能使验证分数偏乐观。

候选完成后，`tools/select_ensemble.py --runs <实验目录1> <实验目录2> ...` 比较所有等权子集及四种聚合方式，限制模型总文件大小。按官方顺序先比较 AUC、相同时再比较调阈值 F1，二者相同再优先较少模型。它与正式推理保持相同顺序：先在各成员内聚合检查，再平均各成员概率，尤其避免 max 聚合时交换顺序导致结果不同。选择后需重新推理和端到端测时，不能把单模型耗时套到集成上。

新增 `--weight-grid quarter --max-members 3` 可比较少量固定全局权重：两模型的 1:3/3:1，三模型的 2:1:1 排列，及原等权配置。权重对十类共用，只在验证阶段选择；`predict.py --ensemble-weights ...` 按模型顺序使用并写入冻结报告，`cost.py` 和原图复现沿用同一权重。`tools/finalize.py` 默认使用该有限网格；旧冻结报告继续等权复现。

最终比较优先使用 `predict.py --export-images --validation-only` 导出的确定性验证逐图概率，再以 `tools/select_ensemble.py --prediction-files <对应NPZ文件> ... --runs <对应实验目录> ...` 计算。候选阶段无需 test CSV 或图像，不生成提交 CSV；选定方案后再运行最终测试推理。工具核对模型摘要、验证 CSV 摘要、图像顺序及批量/增强设置，避免混用旧权重或不同推理配置的缓存。普通推理与水平翻转增强分别导出、分别比较；是否采用增强还需满足耗时约束。冻结报告复现会沿用生成时的批量设置。

新提交规则将概率固定到 12 位小数，截断至 [1e-12, 1-1e-12]，阈值取相邻验证概率之间、靠近中点的半步格点；CSV 写出 15 位小数。概率与 Gate 因而不会落在相同格点，避免 `>` / `>=` 或阈值舍入改变分类结果。校准与选择使用相同精度；旧冻结报告沿用其原有写出方式。

`tools/finalize.py` 在 Python 3.12 / PyTorch 2.10 GPU 环境中统一导出两种增强设置、比较所有等权子集及聚合方式，再按验证排名寻找本机原图样本延迟通过的方案；随后重新生成 CSV、绑定模型/环境/测时证据、打包并从 SSD 原图独立复现。`tools/run_finalization.ps1` 可等待训练流水线后调用。本次真实最终流程已完成；本机样本通过不代表官方 T4 或整个测试集最坏延迟通过。已有交付目录会被保护，重跑请另选输出目录。

`outputs/` 保存审计、配置、每轮指标、最佳权重、可恢复检查点；`submission/` 保存 `submission.csv`、验证报告及本地逐例概率。提交 CSV 以官网示例的五列生成：`Subject_id,Study_id,Predict_class,Probability,Gate`。保留官方 CSV 的字符串 ID，每次检查恰好 10 行，概率范围 [0,1]，UTF-8 编码。初赛仅需上传 CSV；复赛还需模型供验证。

`tools/package.py` 将选定权重、源代码、CSV、校准报告和 SHA-256 清单打包。包内复现命令使用 `--calibration report.json`，只读测试集，沿用冻结的验证阈值、聚合和增强设置，并拒绝不匹配的权重。`python tools/check_deployment.py` 用临时合成数据验证打包后 CSV 完全一致、无需验证标签、清单完整及权重不匹配拒绝逻辑；它不会将合成分数混入真实实验结果。

原图直接推理与缓存生成共享灰度缩放和 quality=95 JPEG 量化处理，避免因为是否存在缓存而改变模型输入。部署测试同时覆盖“缓存生成结果、无缓存原图复现”的路径；最终选定模型已完成全量原图整包复验。

部署和测时统一使用确定性推理设置，关闭卷积自动选型与 TF32，并固定 cuBLAS 工作区。`tools/verify_bundle.py` 检查实际文件摘要、权重和报告绑定、类别映射、模型总大小以及完整 CSV；传入 `--reproduced <独立生成的CSV>` 还会核对概率差异。相同环境与批量下的复现检查不能等同于跨 GPU/软件版本数值完全一致。

官方模型文件总大小限制 ≤500MB、单图端到端推理 ≤100ms，参考硬件 T4 16GB。本机测试不能等同于 T4 测试，后续报告应区分实测与未验证约束。官方目标环境 Python 3.12 / PyTorch 2.10 / torchvision 0.25；当前开发环境 Python 3.9 / PyTorch 2.8 cu126 / torchvision 0.23，移植需复验。

requirements.txt 固定开发时的 timm、Pillow、NumPy、pandas 和 scikit-learn 版本，保留 PyTorch/torchvision 的兼容范围；requirements-competition.txt 对应官方 Python 3.12 目标容器的版本约束，尚不代表已在该环境验收。后续训练保存 Git 提交、是否有未提交修改及主要环境版本；提交包附类别映射和逐个权重的训练配置，缺失的旧来源信息保留为 null。

本机已建立 Python 3.12.15 / PyTorch 2.10 / torchvision 0.25 隔离推理环境，GPU 使用 CUDA 12.6 构建。四份实际训练权重均已在该环境完成完整验证/测试导出；最终两模型包还完成独立原图复现。较早版本的 2.8 开发结果保留在实验记录中，顶部表格和最终报告统一使用 2.10 复算结果。**这些不是 T4 / CUDA 12.8 验收**。`tools/check_target_runtime.py --device cpu|cuda ...` 可另外执行接口检查。

[PyTorch 官方版本页](https://pytorch.org/get-started/previous-versions/) 提供 2.10 的 CPU、CUDA 12.6 和 12.8 安装索引。官方容器使用其指定 CUDA 构建；本机隔离验证使用官方 cu126 构建，未更换现有训练环境或显卡驱动。推理和测时报告现在记录实际 Python、框架、图像库、设备、CUDA/cuDNN 及精度信息；集成选择拒绝混合不同环境的逐图概率。

## 研究参考

- [Asymmetric Loss for Multi-Label Classification, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html)：处理多标签正负不平衡；此项目提供独立实现以供对照。
- [ConvNeXt V2, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html)：现代卷积结构候选；不使用外部预训练权重。
- [CheXFusion, ICCVW 2023](https://arxiv.org/abs/2308.03968)：参考多视角融合及类别不均衡处理方向。本项目最初使用概率聚合，未复现其 Transformer 融合模块。
- [Medical X-ray Attention, 2025](https://arxiv.org/abs/2504.02277)：参考局部与全局证据结合方向，未使用其教师或外部数据。
- [Co-occurrence Aware Class Attention, 2026](https://arxiv.org/abs/2608.09774)：参考每类空间池化方向；此项目的可选池化头不含论文的共现图模块。
- [EfficientNetV2, ICML 2021](https://arxiv.org/abs/2104.00298)：独立架构候选，使用 PyTorch/timm 的 EfficientNetV2-B0 从随机初始化训练，不使用论文中的外部数据或预训练权重。架构优点不代表本比赛一定取得更高分。

实验记录见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。
