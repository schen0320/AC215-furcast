# Pawpularity：真实预处理结果与训练建议

更新：2026-10-16。预处理与校验已实际执行；尚未训练视觉模型。以下模型优先级是针对本项目规模和实现成本的建议，不是本项目上的实测排名。

## 已完成的数据版本

版本：`pawpularity-v1-2026-10-16`。

| 项目 | 实测结果 |
|---|---:|
| 原始带标签记录 / 图片 | 9,912 |
| 原始无标签示例图片 | 8，仅供接口测试 |
| 参与跨源核对的旧图片指纹 | 100,705 |
| 隔离的跨源重复候选记录 | 2,101 |
| 可建模记录 / 300×300 图片 | 7,811 |
| 训练 / 验证 / 留出测试 | 4,689 / 1,562 / 1,560 |
| 保留的图片连通组 | 7,795 |
| 检测到的重复组跨拆分 | 0 |
| 输出图片损坏、哈希不匹配 | 0 |

源 ZIP SHA-256：`6b57735ba5e6041c79b73833fff267db54e425134efbab42728e3025c712153c`。

图片 ZIP SHA-256：`5fe966d538969f897c31a6098f552810db6f58434be5fb5869fc1682afdfe923`。

重复规则沿用仓库实现：EXIF 校正后 RGB 像素完全一致，或 64-bit DCT pHash 汉明距离 ≤6；先构建传递连通组，再排除触及旧数据的 Pawpularity 组，以保护旧模块的固定测试集。共发现 2,218 个跨源候选对（97 个像素完全相同对、2,121 个感知候选对），涉及的记录数与对数不同。Pawpularity 内还有 27 个近似候选对，保留在同一拆分。

**候选不是人工确认的重复宠物。** 本版选择保守隔离，原始 ZIP 不变；后续人工复核可以产生新版本，但必须同步修改全部受影响模块的分组/拆分。缺少 animal/rescuer/timestamp 标识，不能声称完整动物身份独立或时间外验证。也不能保证发现所有裁剪、翻转或同一宠物的不同照片。

## 文件怎么用

- `processed/image_inputs.csv`：7,811 个可用图片路径、拆分、重复组；不含评分或手工属性。
- `processed/labels_train.csv`、`labels_validation.csv`、`labels_test.csv`：按 Id 对应的回归标签。
- `processed/labels_and_splits.csv`：全部 9,912 条原始训练记录及隔离原因；不能忽略 `split` 直接训练全部行。
- `processed/auxiliary_photo_labels.csv`：12 个手工视觉属性，留作辅助监督。使用时仍须按固定 split 过滤。
- `processed/duplicate_candidates.csv`：跨源和内部候选关系，供人工复核。
- `processed/prepared_image_manifest.csv`：每张输出图片的 SHA-256。
- `processed/audit.json`、`output_validation.json`、`provenance.json`：真实执行结果、独立校验及版本信息。
- `Pawpularity_Photos_300.zip`：内部是 `train/<Id>.jpg`、`validation/<Id>.jpg`、`test/<Id>.jpg`。
- `reference/`：此次使用的旧图片指纹，复现跨源检查时使用。

拆分固定为种子 215 的 GroupShuffleSplit，按组约 60/20/20；未根据目标值选择种子。留出测试来自原始有标签 train.csv，不是原始的 8 条无标签示例。所有模型共用这些拆分，验证集用于选模型，最终选定后才使用留出测试。

原图转 RGB、校正 EXIF、保持比例、不放大小图、中心灰色补边至 300×300，JPEG quality 90。未做增强、曝光修复、磨皮、锐化或模型归一化。照片构图、背景和模糊本身可能与评分相关，因此本版避免裁切主体或改善照片外观。缩小和 JPEG 再编码仍会改变细节；高分辨率/模糊敏感实验需要回到原图另建数据版本。

## 推荐的训练顺序

| 优先级 | 模型 | 建议用途与成本 |
|---|---|---|
| 第一版 | **ImageNet 预训练 EfficientNet-B0 + 单输出回归头** | 首选图像 baseline，参数约 5.3M，输入 224×224；实现和迭代成本较低 |
| 第二版 | **ImageNet 预训练 EfficientNet-B3 + 单输出回归头** | 与 MS1 的 EfficientNet 回归方向一致，输入 300×300，参数约 12.2M；比较更高分辨率是否改善验证误差 |
| 可选对照 | **ImageNet 预训练 Swin-T + 单输出回归头** | 不同骨干的对照实验，输入 224×224，约 28.3M 参数；在 baseline 跑通且算力允许后尝试 |

不要从零训练大网络。没有 GPU 时，可先冻结 B0 提取图像特征，再仅训练小型回归头或 Ridge 回归；这仍需计算图像特征，但避免完整反向传播。

CatBoost/XGBoost + 12 个手工属性可作为额外的结构化诊断基线，不能直接充当“用户只上传照片”的最终模型。新照片没有人工属性时，要另做属性预测或由工作人员确认。当前交付不把这些属性当成已知推理输入。

## B0 起步配置（建议，尚未执行）

- 输入：把已补边的 300×300 图整体缩放为 224×224，不再套用默认中心裁切；B3 直接使用 300×300。
- 归一化：除以 255 后，使用 ImageNet mean `[0.485, 0.456, 0.406]`、std `[0.229, 0.224, 0.225]`。这是保留构图的项目变体，和官方权重默认的 resize/center-crop 有意不同；训练与推理必须一致。
- 目标：`Pawpularity / 100`；单输出线性回归头，MSE loss；评估时乘回 100，报告原始分制 RMSE、MAE。
- 第一阶段：冻结骨干并保持其 BatchNorm 为 eval，只训练回归头约 2 个 epoch，AdamW，学习率 `1e-3`。
- 第二阶段：小学习率微调骨干，初始 `1e-4`，weight decay `1e-4`，batch size 16（显存充足再加），最多 20 个 epoch，以验证集 RMSE 早停，patience 5。
- 第一轮不做强随机裁切、颜色增强、模糊增强或 MixUp。后续改变增强策略，应单独记录实验，避免随意改变与评分相关的图像属性。
- 数据版本、Git commit、预训练权重枚举（如 `EfficientNet_B0_Weights.IMAGENET1K_V1`）、依赖版本、随机种子和参数写入实验记录。上述超参数是起点，不是最优配置。
- 主比较用未裁剪预测分数；若产品需要把输出限制在 1–100，预先固定规则并另报裁剪后的误差，不在测试集调规则。

训练数据加载时按 `image_inputs.csv` 的 Id 与对应 `labels_<split>.csv` 一对一关联。输入只取照片像素；Id、路径、split、duplicate_component 和目标列都不输入网络。

## 已计算的比较基准

固定预测训练集均值 **37.8004**，在验证集上得到：

- RMSE：**21.3150**
- MAE：**15.9283**

这是不看照片的常数参考，不是训练后的视觉模型成绩。后续视觉模型应与它在同一验证集比较；多个模型共用同一拆分，报告重复种子或组级重采样的不确定性。没有测试集模型分数，也没有承诺的最终精度。

可把 Spearman 相关性作为排序辅助指标；它不等于“同一只宠物换照片”的实际收益，因为本数据没有可靠的同宠物多照片对照标签。

## Photo Attractiveness Assessment 的边界

产品可以称为照片吸引力评估，但训练目标应明确写成 **Pawpularity / photo-profile engagement score prediction**。官方分数来自页面浏览统计的归一化，不是客观美学真值、领养速度或干预效果。

若要输出“主体不清楚、被遮挡”等具体建议，可后续使用 `Subject Focus`、`Blur`、`Occlusion` 等属性作为独立辅助分类任务，分别报告指标并让工作人员确认。单个回归分数或注意力图不足以证明具体原因，不能承诺改图会缩短领养时间。

## 复现与校验

代码使用本仓库 `data/pawpularity-preprocessing` 分支（合并后可用主分支）。将原始 ZIP 解压为 `RAW`，处理包解压为 `BUNDLE`，在仓库目录运行：

```bash
python -m pip install -r requirements-pawpularity.lock.txt
python prepare_pawpularity.py --raw RAW \
  --external-manifest BUNDLE/reference/breed_fingerprints.csv \
  --external-manifest BUNDLE/reference/petfinder_fingerprints.csv \
  --out results_pawpularity_new \
  --export-images outputs/Pawpularity_Photos_300.zip --size 300 --workers 4
python validate_pawpularity_outputs.py --out results_pawpularity_new \
  --photos outputs/Pawpularity_Photos_300.zip
```

目录必须是新的，防止覆盖固定版本。原始大文件、逐条记录及图片不提交 GitHub。此次尚未配置 DVC remote、GCP、Docker，也未训练上述视觉模型。

## 官方资料

- 数据与评分定义：https://www.kaggle.com/competitions/petfinder-pawpularity-score/data
- 比赛评估：https://www.kaggle.com/competitions/petfinder-pawpularity-score/overview/evaluation
- EfficientNet-B0：https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html
- EfficientNet-B3：https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b3.html
- Swin-T：https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.swin_t.html
- CatBoost 回归：https://catboost.ai/docs/en/concepts/loss-functions-regression

模型结构/权重信息来自官方资料；推荐顺序、超参数和保留构图策略是本项目的工程建议，需要用上述固定验证集验证。
