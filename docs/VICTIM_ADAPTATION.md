# 更换受害模型时需要修改的接口

当前运行代码明确使用`Qwen2_5_VLForConditionalGeneration`，并非通过配置即可切换到所有视觉语言模型。以下区分“同架构换权重/规模”和“换架构”。迁移包没有代替你选定或下载新受害模型。

## 同属Qwen2.5-VL

在`server_paths.local.json`修改模型路径、`target_dim`、`num_tokens`和投影器路径，使用新`run_name`。还要检查处理器产生的视觉token数量、模型隐藏维度以及原始GHOST投影注入接口。

原始GHOST的投影器必须与新模型匹配；不能仅因为隐藏维度相同就默认旧权重合适。训练脚本`st-ghost/train_projector_qwen.py`已带入包中。

如何使用同图像的CLIP向量与原生视觉token进行MSE训练、准备COCO train/val图片、选择权重并接入实验，见[PROJECTOR_TRAINING.md](PROJECTOR_TRAINING.md)。该说明同时记录原始`GHOST-master/train_projector.py`的训练参数与限制。

## 更换为其他架构

| 文件/位置 | 需要适配的内容 |
|---|---|
| `st-ghost/run_stage1.py: VictimRuntime` | 模型类、处理器、加载精度、模型前向、正常图像问答、Yes token处理 |
| `GHOST-master/utils.py: vllm_standard_preprocessing` | 对话模板、图像预处理、输入字典及设备；保持生成图像梯度 |
| `GHOST-master/utils.py: get_model_inputs`及对应模型分支 | 原始GHOST将投影特征注入语言模型输入的方式；不能固定使用qwen分支 |
| `st-ghost/run_stage3_original_ghost_comparison.py: checked_probability` | 投影输出形状、模型需要的token结构和精度；本函数调用公共受害模型接口 |
| `st-ghost/run_stage2_level_oo_generation_feedback_victim_evaluation.py: victim_terms` | 当前直接使用`model.visual.dtype`、`pixel_values`、`logits_to_keep=1`，新架构未必支持 |
| `st-ghost/train_projector_qwen.py` | 新模型视觉特征提取、token数量和维度；其他架构需要相应训练器 |
| `tools/configure_server.py` | 完成上述适配后，更新`victim_family`检查和资产检查规则 |

原始`GHOST-master/utils.py`保留了LLaVA、Llama、GLM等部分辅助函数，可作参考；第三阶段入口尚未自动接入这些分支。

## 在正式跑之前验证

用一张图检查正常问答与token注入问答。随后使用一轮完整生成优化，确认：

1. 输入生成图像经过处理器后仍连接计算图，没有通过PIL、NumPy或`detach()`切断梯度。
2. `L_victim`能向生成图片、再向21组条件产生有限、非零的梯度。
3. Yes概率与模型解码回答的统计定义一致；不同分词器可能需要不同Yes token集合，不能盲目复用旧ID。
4. 新投影器输出和模型视觉token接口匹配；模型权重保持冻结。
5. 一次完整更新和最终生成/问答评估都能在目标设备上完成。

这些验证应在新的实验输出中记录。迁移包的CPU测试不能代替它们。

## 对已有清单和检查点的处理

`migration_inputs/historical/selected_images_2000.json`与`selected_images_1000.json`用于重建旧实验样本ID；里面的No回答来自旧Qwen，不是新模型的测试结果。

推荐使用同一CLIP候选排序、在新受害模型上重新执行`--prepare-only`，然后让两种方法共享该模型对应的新图池。ST-GHOST再均衡抽1000张，比较时从原始GHOST结果中取这同1000张。

不要修改历史检查点中的模型路径、预筛选答案或SHA-256，使其看似来自新模型。跨模型继续的是实验方案，而不是旧模型优化器的运行状态。
