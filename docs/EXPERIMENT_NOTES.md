# 实验口径与文件用途

## 两个第三阶段实现

原始GHOST对比入口恢复“达到阈值则生成一张，若不满足成功条件则继续同一个AdamW优化器”的流程。默认最多100次更新、最多4次生成，原脚本未强制固定随机种子。比较任务为双物体Multi-O，是本项目对原始GHOST任务的扩展。

ST-GHOST从同批2000张中每个组合随机选200张，共1000个独立图片任务。21个实际执行的去噪步骤分别有可优化条件，干净CLIP嵌入初始化，45次Adam更新、学习率0.03。最后固定21组条件，用4个生成种子获得候选；种子0对应优化过程中使用的噪声状态，其余3个不再优化。

### 必须区分的指标

- 优化时连续Yes概率、负对数损失，与最终解码回答Yes的成功率不同。
- 原始GHOST记录受害模型Yes，也记录目标检测验证后的有效成功。
- 当前ST-GHOST入口只用受害模型Yes判断成功，没有自动执行目标检测。
- 跨方法直接比较时应使用相同1000张图片、相同问句与相同成功定义；不要将原始方法的检测后成功率和ST-GHOST的原始Yes成功率混用。
- 4个生成种子的候选是同一图片任务的4次尝试，不能把4000张候选当作4000个独立任务。
- SSIM≥0.8是软约束目标，不是每张生成图像必然满足的门槛。

### VAE行为差异

当前原始GHOST对比刻意保留原始utils的历史图像归一化/后处理；ST-GHOST使用修正后的`[-1,1]` VAE编码输入和`decoded/2+0.5`输出。源脚本已记录这一差异，本迁移包未改变它。论文比较时应说明，或另做控制变量实验。

## 为什么需要看起来属于第一、二阶段的脚本

| 文件 | 第三阶段的用途 |
|---|---|
| `run_stage1.py` | 公共CLIP、受害模型、投影器、检测器及扩散管线加载 |
| `run_stage1_exp1.py` | 原始GHOST对比的问句、目标组合、排序、对比损失和结果工具 |
| `run_stage1_exp4.py`、`run_stage1_exp4_level_oo.py` | 公共Runtime、双物体规格和确定性生成种子工具 |
| `run_stage2_level_oo_generation_feedback.py` | 可微分扩散状态、逐步条件注入、VAE解码及CLIP再编码 |
| `run_stage2_level_oo_generation_feedback_step_compare.py` | 共享选择/加载/摘要工具及协议常量 |
| `run_stage2_level_oo_generation_feedback_victim_evaluation.py` | 21条件优化、三个损失、图像/嵌入快照、断点与受害模型评估 |
| `run_stage2_level_oo_generation_feedback_victim_evaluation_500_four_seeds.py` | 固定21组条件后生成4个候选并汇总任务成功率 |
| `GHOST-master/main.py` | 原始参考实现，且被基线实验的实现哈希记录引用 |
| `GHOST-master/train_projector.py`、`evals/validate_projector.py` | 原始投影器训练及物体问答验证参考；方法与运行说明见`PROJECTOR_TRAINING.md` |
| `GHOST-master/utils.py`、`data.py`、`enhanced_text_representations.py`、`projector/` | 运行时实际导入的辅助实现 |

包中的这些源文件与整理时原项目一致。目录名里的500或旧阶段编号是历史命名，不代表第三阶段仍使用500张。

`run_stage2_level_oo_atomic_guided_embeddings.py`、`run_stage2_level_oo_atomic_guided.py`及`run_stage2_level_oo_loss_diagnostic.py`仅为现有反馈优化测试提供归档格式工具及其导入依赖；第三阶段启动不需要执行这些历史实验入口。

## 小型输入文件

`migration_inputs/ranking_cache/`只含CLIP排序和当时配置，不含旧受害模型攻击成功筛选。新配置工具重新生成与新路径匹配的缓存配置，但要求CLIP权重身份保持一致。

`migration_inputs/reference/stghost_reference.json`满足当前ST-GHOST入口的参数一致性检查。它只是参数模板，不会加载旧750份实验的embedding或结果。

`migration_inputs/historical/`保存旧图池、已选1000张、历史协议及摘要，供论文和样本匹配核对。没有完整优化轨迹或攻击图片，不能据此恢复已完成任务。JSON中的旧项目根路径已改为相对路径，因此这些导出文件不是原始签名校验所用的字节副本。

`SOURCE_SNAPSHOT.json`与`BUNDLE_MANIFEST.json`帮助检查源文件复制和Git下载是否完整；它们不能取代原实验输出中的实验身份签名。

## 大文件的迁移

模型、COCO、投影器和历史输出应使用独立的文件传输或下载流程。若之后需要在同一模型上恢复已有第三阶段断点，需要另行迁移整个相关输出目录，并处理绝对路径与实现签名的绑定；本包没有执行这种断点转换。

## 上游许可

引用的GHOST源代码及MIT许可保存在`GHOST-master/`。上游论文引用信息见该目录的README。本迁移包没有为用户新增的研究代码另行指定开源许可。
