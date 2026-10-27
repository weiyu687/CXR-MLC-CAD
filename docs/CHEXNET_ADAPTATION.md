# CheXNet 适配与实验

日期：2026-10-27。用户要求暂停此前实验方向，参考 [arnoweng/CheXNet](https://github.com/arnoweng/CheXNet) 适配本赛数据并实验。[暂停记录](PAUSED_EXPERIMENTS_20261027.md) 保存原方向状态，原第二轮提交文件继续保留，真实测试AUC为0.7678。本页不把新的程序检查当作性能提升。

## 已核对的参考内容

指定仓库是CheXNet的PyTorch再实现，公开了 `model.py`、`read_data.py`、README、NIH胸片训练检查点及划分文件，**没有公开训练循环**。仓库的“improved model”只给出成绩，不能据此声称已复现未公开的改进结构。参考 [原论文第2、3、5节](https://arxiv.org/html/1711.05225v3) 补齐训练。

仓库模型为DenseNet121，分类器线性层后接sigmoid；验证/测试将RGB图像短边缩放到256，取224的十裁剪，每个裁剪先得到概率，再平均。ImageNet均值/标准差用于三个通道。

原论文的单病肺炎任务采用加权BCE；第5节的**多标签版本采用逐类无权重BCE之和**。本赛为十标签，因此采用后者。原论文报告ImageNet初始化、Adam(beta1=0.9、beta2=0.999)、batch16、初始学习率0.001、验证损失平台期学习率乘0.1、按最小验证损失选权重；训练图像224并随机水平翻转。

本项目只使用Torchvision公开ImageNet-1K DenseNet121初始化，SHA256 `a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395`。**未下载或使用仓库的NIH医学训练检查点，未引入NIH图像**。引用该仓库的方法不等于使用其模型权重。比赛2026-10-23规则对医学预训练的限制仍适用。

## 对本赛的具体适配

| 内容 | 实现与变化 |
|---|---|
| 输出 | 14类改为官方CSV的1到10类；含本赛的No Finding，不沿用NIH标签位置 |
| 训练样本 | 官方train中的19,757张AP/PA/PA LLD正位图，按图遍历；不使用valid/test梯度 |
| 标签与评估 | 同检查标签一致；官方valid完整859次检查，优先对正位概率均值，缺正位时使用该检查现有图像 |
| 骨干与损失 | DenseNet121+10维线性层，训练返回logits并用稳定的BCEWithLogitsLoss；逐类求和后对批次取均值，推理仅做一次sigmoid |
| 图像 | RGB、短边256、ImageNet标准化，另建无损PNG打包缓存；不使用原先的448灰度补边/JPEG缓存。训练在共享256输入上再缩放224方形并翻转，这是明确的缓存适配 |
| 裁剪 | 验证同时记录十裁剪均值与其中心裁剪成员，后者无需额外前向；最终部署采用哪个需按验证和耗时另行冻结 |
| 优化 | Adam、无权重衰减、无EMA/Mixup/类别加权/额外旋转；本机采用AMP以适配设备，论文未报告本项目的AMP设置 |
| 调度 | ReduceLROnPlateau，验证检查级BCE触发，factor0.1、patience2、min_lr1e-7；论文未明确patience，本项目明示取值 |
| 轮数 | 每组最多60轮、至少20轮，验证损失连续12轮无改善才停止；这不是已经完成60轮 |
| 检查点 | `loss_best.pt` 按验证BCE选取，`best.pt` 按比赛验证Macro-AUC选取；分别记录，避免混淆论文选型与竞赛选型 |

计划两组仅改变初始学习率：**0.001论文基准** 与 **0.0001小数据适配对照**，均使用seed2050、batch16及相同数据顺序。验证被用于调度和选择，结果存在选型偏差，不是独立隐藏测试成绩。不宣称论文ChestX-ray14的AUC可以直接映射到本赛。

## 已完成的程序与输入核对

`tools/check_chexnet.py` 已在PyTorch2.8和2.10 CPU通过：十裁剪输入张量与指定仓库逐值相同、十标签模型的概率对应、稳定BCE与sigmoid+BCE损失/梯度对应、先sigmoid后均值、中心裁剪索引正确、原图与无损缓存张量逐值相同。没有加载任何医学训练权重。

完整训练入口在合成CPU数据完成两轮，覆盖训练/验证、十裁剪与中心裁剪、学习率调度状态及两种检查点保存。随后在目标PyTorch2.10环境中，以原图和缓存分别推理，概率最大差为0.0。**这些是正确性检查，不是比赛成绩。**

GPU短资源测量包含反向传播、EMA和AdamW状态，224/batch16约113.8图/秒，峰值allocated1175MiB；正式CheXNet不使用EMA、使用Adam，因此此数值仅为短探测参考。DenseNet121/224模型较小，batch16保留论文设置；显存占用不等于优化是否充分。T4与十裁剪完整耗时仍需在候选冻结后复验。

机器可读证据见 [程序核对汇总](research/chexnet_checks_20261027.json)。源码在 `src/chexnet.py`，训练在 `tools/train_chexnet.py`，独立推理在 `tools/predict_chexnet.py`。新检查点会拒绝走旧的448补边推理入口，避免静默使用错误预处理。

## 运行入口

```powershell
python -m tools.prepare_chexnet_cache --root raw_data/cxr_first --cache <CheXNet缓存目录>
python -m tools.run_chexnet --cache <CheXNet缓存目录> --execute
python -m tools.predict_chexnet --checkpoint <实验目录>/best.pt --root raw_data/cxr_first --cache <CheXNet缓存目录> --out <验证导出目录> --validation-only
```

`run_chexnet`不带`--execute`只展示固定命令；执行时要求代码已提交、核对缓存及ImageNet权重摘要、冻结源代码并串行训练两组。结果在 `outputs/chexnet_20261027/`。`status.json`、`plan.json` 和逐组日志保留进度和依据。暂停的AUC/DINOv2队列不会自动恢复。

最终测试推理需要先冻结验证报告：`python -m tools.predict_chexnet --checkpoint <best.pt> --root <官方数据目录> --calibration <冻结report.json> --out <输出目录>`。该模式只读取test元数据和图像，不需要valid标签；可不传缓存以直接复现原图。尚未执行比赛网站提交，验证AUC≥0.90目标仍未达到。

## 启动与后续复核

两组正式队列已于2026-10-27 **10:53:36** 从干净工作区启动，冻结代码提交 `cc2641a88a6dfb443361484e3bcc4f886031788f`。当前先运行0.001组，实际轮数、逐轮汇总及记录时间见[进度快照](research/chexnet_progress_20261027.json)与[阶段曲线](figures/chexnet_progress_20261027.png)。这是仍在训练的记录，不是完成两组实验后的结论。原AUC/DINOv2方向未恢复。

`tools/review_chexnet.py` 在两组训练全部完成后，以Python3.12/PyTorch2.10、batch1重新评估两种选型检查点及两种裁剪，复核冻结源文件和权重摘要，只读取官方valid。**中心裁剪单独以batch1前向**；训练日志从十裁剪批次抽取的中心成员仅作训练期观察，不能当作不同CUDA批次形状下的部署复核。两种模式均采用实际部署的概率舍入策略。汇总写入 `docs/research/chexnet_verified_20261027.json`，本地回执在 `outputs/chexnet_20261027/review_complete.json`。新成绩仍属于反复选型后的验证结果，不能称为测试成绩。该复核不会执行网站提交或自动恢复暂停队列。

独立测时入口为 `python -m tools.cost_chexnet --checkpoint <best.pt> --calibration <冻结报告> --submission <CSV> --root <原图目录> --pictures <官方100张目录> --warmup-pictures <官方5张目录> --out <测时结果目录>`。默认连续3轮，计入原图和同检查配套图读取、十裁剪、聚合及真实CSV写入/flush/fsync，模型初始化不计。GPU测时前检查空闲负载。已用合成CPU数据验证其与冻结CSV一致；样本仅2/预热1，不计作官方协议或T4通过。正式GPU测时应在训练结束、候选冻结后运行。

## 完成后的比较与交付

`tools/finish_chexnet.py` 接在目标环境复核之后，重新从按模型/CSV/图像顺序绑定的验证预测构建原交付AUC，逐例与新候选对齐，再按228名验证患者配对重采样1000次。结果报告每个独立候选的逐类AUC、与原交付差值及区间；这仍是选型后的描述性估计。

完成流程先冻结随机种子2052选取的100张验证原图和5张预热图，不参考标签；以这组输入连续三轮测量完整检查推理。至少测量两种裁剪各自验证AUC最好的候选，也测量其他AUC超过原交付的候选。验证测时明确记录 `split=valid`，不称为官方测试样本测时。

只有**验证AUC超过0.8111219097，且验证原图平均耗时≤100ms**的CheXNet单模型才进入新候选交付。按验证AUC选择并冻结后，才读取测试集，执行两次独立原图推理及测试5/100/3测时；若此时失败则保存证据并停止，不根据测试耗时换模型。没有合格替换者时，不生成较弱候选的新测试CSV，原76.78分文件继续保留。

```powershell
python -m tools.finish_chexnet --timing-root <验证原图SSD目录> --test-root <测试原图SSD目录>
```

汇总、训练曲线和结果说明将写入 `docs/research/chexnet_completion_20261027.json`、`docs/figures/chexnet_20261027.png`、`docs/CHEXNET_RESULTS_20261027.md`。满足替换条件时，`tools/package_chexnet.py` 生成 `submission/chexnet_bundle_20261027/` 及zip，包含独立推理入口、原图测时入口、权重、类别映射、固定阈值、来源和文件摘要；不包含比赛原图。加载权重与必要配置合计限制为500MB。冻结完成流程的代码摘要另记于本地 `completion_plan.json`，没有更改正在训练的20个冻结源文件。

完成流程的程序检查使用合成CPU数据：独立材料在仅有test元数据/图像的目录复现概率最大差0.0，文件被修改时拒绝验证，把中心裁剪耗时报告套给十裁剪时也拒绝；zip的CRC与每个文件摘要通过。它们只说明交付流程可运行，不代表正式GPU耗时、比赛得分或T4合格。
