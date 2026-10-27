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
