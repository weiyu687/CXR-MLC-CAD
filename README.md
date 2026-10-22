# CXR-MLC-CAD

2026 年第八届全国高校计算机能力挑战赛 **人工智能挑战赛**：基于胸部 X 光片的多标签疾病分类。使用 PyTorch，在官方数据上训练，对每次检查输出 10 类概率。

比赛规则来源：[官方网站](https://www.ncccu.org.cn/index/Paper/case1.html)（核对日期 2026-10-22）。区域赛/省赛截止时间为 2026-11-13 23:59。此项目只生成提交材料，不自动向比赛网站提交。

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

严格只使用官方训练数据，不下载 ImageNet 或医学数据预训练权重。以轻量 ResNet18 从零训练建立基线，支持 timm 的其他架构及不对称损失进行后续对比。每个训练 epoch 对每次检查随机选择一个视角，避免多图检查被过度加权；验证与测试使用所有图像并在检查层面聚合。使用轻微几何/灰度增强、AdamW、预热和余弦学习率、混合精度及 EMA 权重。模型以官方验证集检查级 Macro-AUC 选择，Macro-F1 单独报告。各类别阈值只影响 F1，不影响 AUC。

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

候选完成后，`tools/select_ensemble.py --runs <实验目录1> <实验目录2> ...` 比较所有等权子集及四种聚合方式，限制模型总文件大小，分数相同时优先使用更少的模型。它与正式推理保持相同顺序：先在各成员内聚合检查，再平均各成员概率，尤其避免 max 聚合时交换顺序导致结果不同。选择后需重新推理和端到端测时，不能把单模型耗时套到集成上。

`outputs/` 保存审计、配置、每轮指标、最佳权重、可恢复检查点；`submission/` 保存 `submission.csv`、验证报告及本地逐例概率。提交 CSV 以官网示例的五列生成：`Subject_id,Study_id,Predict_class,Probability,Gate`。保留官方 CSV 的字符串 ID，每次检查恰好 10 行，概率范围 [0,1]，UTF-8 编码。初赛仅需上传 CSV；复赛还需模型供验证。

`tools/package.py` 将选定权重、源代码、CSV、校准报告和 SHA-256 清单打包。包内复现命令使用 `--calibration report.json`，只读测试集，沿用冻结的验证阈值、聚合和增强设置，并拒绝不匹配的权重。`python tools/check_deployment.py` 用临时合成数据验证打包后 CSV 完全一致、无需验证标签、清单完整及权重不匹配拒绝逻辑；它不会将合成分数混入真实实验结果。

原图直接推理与缓存生成共享灰度缩放和 quality=95 JPEG 量化处理，避免因为是否存在缓存而改变模型输入。部署测试同时覆盖“缓存生成结果、无缓存原图复现”的路径；最终材料还需对实际选定模型执行整包复验。

官方模型文件总大小限制 ≤500MB、单图端到端推理 ≤100ms，参考硬件 T4 16GB。本机测试不能等同于 T4 测试，后续报告应区分实测与未验证约束。官方目标环境 Python 3.12 / PyTorch 2.10 / torchvision 0.25；当前开发环境 Python 3.9 / PyTorch 2.8 cu126 / torchvision 0.23，移植需复验。

## 研究参考

- [Asymmetric Loss for Multi-Label Classification, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html)：处理多标签正负不平衡；此项目提供独立实现以供对照。
- [ConvNeXt V2, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html)：现代卷积结构候选；不使用外部预训练权重。
- [CheXFusion, ICCVW 2023](https://arxiv.org/abs/2308.03968)：参考多视角融合及类别不均衡处理方向。本项目最初使用概率聚合，未复现其 Transformer 融合模块。
- [Medical X-ray Attention, 2025](https://arxiv.org/abs/2504.02277)：参考局部与全局证据结合方向，未使用其教师或外部数据。
- [Co-occurrence Aware Class Attention, 2026](https://arxiv.org/abs/2608.09774)：参考每类空间池化方向；此项目的可选池化头不含论文的共现图模块。

实验记录见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。
