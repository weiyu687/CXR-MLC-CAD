# 研究后开发与受控实验

2026-10-25 继续开发。用户反馈仍为第一轮 `final_bundle/submission.csv` 的 **72 分、432/488**，没有新榜单反馈。论文与仓库阅读范围见 [研究报告](RESEARCH_REVIEW.md)。本轮实际结果完成后追加在此，运行中的任务不记作已完成实验。

## 已启动的有限对照

共用 torchvision ImageNet DenseNet121、320 输入、固定模型 seed=2036、患者 split_seed=2030、12% 训练患者留出、正位优先、同一增强；不使用 Mixup。每轮只在该患者留出集上决定是否保存最佳权重，最多 12 轮、4 轮无提升停止。官方 valid 仅先报告，不决定本组保存轮次。

| 输出目录（本地，不公开数据） | 损失 | 分类头预热 | 状态 |
|---|---|---:|---|
| `outputs/research_ablation_s2036/bce_baseline` | 平方根正类加权 BCE | 0 轮 | 已启动 |
| `outputs/research_ablation_s2036/bce_head_warmup` | 同上 | 2 轮 | 已排队 |
| `outputs/research_ablation_s2036/asl_head_warmup` | ASL，负类 γ=4、正类 γ=0、clip=.05 | 2 轮 | 已排队 |

BCE 的两组只改变冻结预热开关；ASL 与 BCE 是两种损失配方，不能将正类加权与损失的差异拆成单一因素的因果结论。两轮预热是待验证选择。

## 新增实现

- `tools/compare_patient_holdout.py`：重新导出 train 患者留出预测，检查患者列表与 checkpoint 的划分参数相符、不同模型检查/标签/患者顺序一致。拒绝全 train 微调或从比赛 checkpoint 初始化的模型，防止把已见患者当独立留出。不读取 test。
- `tools/patient_statistics.py`：按患者整簇、有放回重采样；所有模型使用相同患者抽样。用患者抽中次数作为其全部检查的权重，计算十类 AUROC、Macro-AUC 与相对参考模型差值的 95% 百分位区间。得分并列按半对计；某轮缺少某类正或负样本时，对全部模型共同剔除此轮，并报告数量。区间仍受 checkpoint/模型选择影响，不宣称是独立确认。
- `tv_densenet121_query`：相同 DenseNet121 骨干接小型疾病查询分类头，256 维、4 头交叉注意力、10 个可学习查询、逐类线性读出；显式 `batch_first=True`。它参考 Query2Label 的一般思想，未宣称完整复现 Query2Label、ML-Decoder 或论文成绩。当前未训练、未加入部署候选，先保留 GAP 分类头完成损失/预热对照。

查询模块不使用独立位置编码：它对 CNN 特征的空间排列不敏感，空间信息来自骨干。分类头内部无跨图像注意力；骨干正常微调时的 BatchNorm 按常规更新，推理时固定为 eval。

## 已通过的检查

在训练环境 PyTorch 2.8 与目标版本 PyTorch 2.10 中均通过：

- 带大量并列分数、患者检查数不等时，加权 AUROC 与 sklearn 的 `sample_weight` 结果误差小于 1e-14。
- 相同模型的患者配对差值区间严格为 [0, 0]；固定种子完全复现；原始划分缺失的类别采用固定排除策略。
- 查询头的 batch/单图结果一致；替换其他图像不改变当前图像结果；空间 token 重排、梯度、严格权重加载、分类头预热期间的骨干参数与 BN 冻结检查通过。这些检查使用合成张量，不是临床数据或泛化成绩。

实际留出比较示例（先等相应任务完成）：

```powershell
python -m tools.compare_patient_holdout --runs outputs/research_ablation_s2036/bce_baseline outputs/research_ablation_s2036/bce_head_warmup outputs/research_ablation_s2036/asl_head_warmup --cache <448缓存路径> --out outputs/research_ablation_s2036/patient_comparison
```

默认 1000 次患者 bootstrap，frontal 聚合，无翻转 TTA。逐例预测与患者列表均只保存于被 Git 忽略的本地 outputs。公开结果仅保留汇总、模型/元数据摘要与实际运行配置。

## 后续选择与交付

先比较本组实际留出结果，再决定分类头和全 train 微调，不预先将未训练的复杂架构放进提交包。选用依据包括留出均值、逐类变化和患者配对区间，官方验证作为另一个划分报告。进入全 train 微调后，原患者留出不再独立，必须明确标注。

新文件夹须完成 PyTorch 2.10、batch=1 的最终推理、原图独立复现、大小核验及包含写出的计时。官网 10 月 23 日 PDF 链接已于 10 月 25 日重新核对。T4 / CUDA 12.8 仍需主办方复测。第一轮交付包保持不变，不执行网站提交。
