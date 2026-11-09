# 受害模型专用投影器训练说明

本说明依据本项目中的`GHOST-master/train_projector.py`、`projector/projector.py`和`utils.py`整理。原始README没有给出完整训练步骤，因此以下“原始默认值”来自代码，并不代表论文中每个模型的最终训练配置。

迁移包同时保留两种训练入口：

| 入口 | 用途 |
|---|---|
| [`GHOST-master/train_projector.py`](../GHOST-master/train_projector.py) | 原始训练实现，按`model_name`选择固定模型，供核对和适配其他架构 |
| [`st-ghost/train_projector_qwen.py`](../st-ghost/train_projector_qwen.py) | 本项目的本地Qwen2.5-VL训练器，支持本地模型路径、配置文件和断点；当前迁移启动工具调用它 |
| [`GHOST-master/evals/validate_projector.py`](../GHOST-master/evals/validate_projector.py) | 比较原生图像输入与投影token输入的物体问答表现；属于功能验证，不是训练入口 |

## 1. 投影器学什么

GHOST的外部投影器把一个CLIP全局图像向量转换成受害模型可接收的视觉token序列。它与受害模型自带的视觉编码器、视觉合并器或多模态投影层是不同的组件。

对每张干净图像$x$，构造一对训练样本：

$$
c(x)=E_{\mathrm{CLIP}}(x)\in\mathbb R^{d_c},
\qquad
T_M(x)=V_M(x)\in\mathbb R^{N\times d_M}.
$$

其中，$E_{\mathrm{CLIP}}$是冻结的OpenCLIP图像编码器；$V_M$表示从受害模型提取、与后续注入位置对应的原生视觉特征。对于当前Qwen分支，它是处理器预处理后调用`model.visual(...)`的输出，已经包含该视觉模块内部的合并与对齐，不能随意替换成更早的ViT patch特征。

外部投影器$\Pi_\theta$的训练目标是逐token、逐通道的均方误差：

$$
\mathcal L_{\mathrm{proj}}(\theta)
=\frac{1}{B N d_M}
\sum_{b=1}^{B}
\left\|\Pi_\theta(c(x_b))-T_M(x_b)\right\|_F^2.
$$

两条特征提取路径都在`torch.no_grad()`中执行，优化器只接收投影器参数。训练不需要目标幻觉问句、Yes/No标签或对抗图像；其目的是近似受害模型正常的视觉表示。之后攻击实验冻结投影器，沿其梯度更新输入CLIP嵌入。

原始代码直接使用`clip_model.encode_image(...)`的输出，没有在MSE训练前额外做单位范数归一化。复现时应保持训练与攻击输入的特征尺度一致。

## 2. 网络结构及输出尺寸

实际使用的是`TokenMLP`，而不是同文件中另外定义的Transformer解码器或`TokenMLPRes`。它为每个输出位置维护一个可学习向量$p_i\in\mathbb R^{d_p}$，把同一个CLIP向量与各位置向量拼接，再通过共享MLP：

$$
\hat t_i=\operatorname{MLP}_\theta([c;p_i]),
\qquad
\Pi_\theta(c)=[\hat t_1,\ldots,\hat t_N].
$$

MLP为4个线性层，中间3次GELU激活；各位置向量也参与训练。批量输入形状为`[B, clip_dim]`，输出形状为`[B, num_tokens, target_dim]`。

原始`utils.get_num_tokens`硬编码了以下尺寸：

| `model_name` | 代码加载的受害模型 | token数$N$ | 通道维度$d_M$ |
|---|---|---:|---:|
| `qwen` | Qwen2.5-VL-7B-Instruct | 144 | 3584 |
| `llava` | llava-v1.6-mistral-7b-hf | 1176 | 4096 |
| `glm4.1v-thinking` | GLM-4.1V-9B-Thinking | 144 | 4096 |
| `llama` | Llama-3.2-11B-Vision-Instruct | 6404 | 4096 |

这些是特定模型及预处理下的代码设置，不能套用到整个模型家族。换受害模型时，应先按攻击阶段所用的图像处理规则，实际测量`T_M(x).shape`，再确定`num_tokens`和`target_dim`。动态分辨率或裁剪策略改变时，token数也可能改变。

当前Qwen本地配置为：`clip_dim=1024`、`context_dim=1024`、`hidden_dim=2048`、`expected_num_tokens=144`、`target_dim=3584`。训练器会检查预测与目标形状是否完全一致，不应依靠MSE广播来处理尺寸差异。

## 3. 数据与训练流程

原始实现使用COCO图片，训练集为`train2017`的随机子集，验证集为`val2017`全部图片。图片先转换为RGB并缩放到336×336，再分别通过CLIP和受害模型自己的处理器。两种训练器都逐批读取图片，不预先缓存全部视觉特征。

```text
COCO/
  train2017/*.jpg
  val2017/*.jpg
  annotations/                 # 投影器训练本身不读标注；第三阶段筛选仍需标注
```

**投影器训练比第三阶段攻击额外需要`val2017`图片。** 当前两种训练器都直接读取`coco_root/train2017`和`coco_root/val2017`；若图片放在`COCO/images/`下，训练配置的`coco_root`应指向`COCO/images`，或事先建立相应目录链接。第三阶段的`data_path`仍须指向含`annotations`的COCO根目录。

每次训练按如下顺序执行：

1. 固定随机种子并选取训练子集，加载CLIP和受害模型。
2. 对同一批图片提取CLIP向量及受害模型视觉token，两条路径不建立训练计算图。
3. 仅对`TokenMLP`前向结果计算MSE，以AdamW更新投影器。
4. 每轮在验证集上计算相同MSE，记录曲线并保存权重。
5. 选择验证损失较好的权重，再检查投影token路径的正常图像问答表现。

默认子集比例为10%；COCO完整train2017约11.8万张，对应约1.18万张训练图，验证集通常5000张。具体数量由实际文件数决定。训练器没有自动剔除第三阶段测试图片；若论文要求数据划分互斥，需要另行明确训练划分。

## 4. 原始默认值与当前配置的差异

| 项目 | 原始`train_projector.py`默认值 | 当前`train_projector_qwen.json` |
|---|---|---|
| 优化器 | AdamW | AdamW |
| 学习率 / weight decay | 0.001 / 0 | 0.001 / 0 |
| 训练子集比例 | 0.1 | 0.1 |
| epoch数 | 10 | 8 |
| 训练 / 验证batch size | 1 / 16 | 1 / 1 |
| 梯度裁剪范数 | 0，关闭 | 1.0 |
| 学习率调度 | 默认关闭；可选step或cosine | cosine，`T_max=epochs` |
| 随机种子 | 42 | 42 |
| context / hidden维度 | 1024 / 2048 | 1024 / 2048 |
| 权重保存 | 每轮纯`state_dict` | 最优纯权重、最优完整断点、末轮及逐轮断点 |

因此，当前本地训练器遵循原始特征回归目标，但不能把它的8轮、cosine等配置表述为原始脚本的默认值。

原始参数`--no_selected_tokens=32`对应的token抽样代码已注释，实际训练与验证比较全部输出token。当前本地JSON中的`warmup_steps=0`保持为0即可，该本地训练器没有实现非零warmup分支；原始训练器有warmup实现。

## 5. 新服务器推荐运行方式：Qwen2.5-VL

下面命令均在迁移包根目录执行，示例`run_name`为`qwen25vl7b_stage3`，实际使用时替换成自己的新名称。先按README安装环境与准备本地模型。

在`server_paths.local.json`中设置新模型、同一CLIP权重和新投影器保存路径。例如：

```json
{
  "projector_path": "/srv/checkpoints/projector_new_victim/best_state_dict.pt",
  "target_dim": 3584,
  "num_tokens": 144
}
```

以上是完整路径配置中的三个字段，其他必需字段见`configs/server_paths.example.json`。3584和144仅为Qwen2.5-VL-7B示例。

先生成配置，此时投影器文件尚未生成，不要加`--check-assets`：

```bash
python tools/configure_server.py --paths server_paths.local.json
```

检查生成的`runtime/qwen25vl7b_stage3/configs/projector.json`，确认`qwen_model_path`、`clip_checkpoint_path`、`coco_root`及尺寸。若COCO采用`images/train2017`布局，单独把这份训练配置的`coco_root`改成`.../COCO/images`；重新运行配置工具会覆盖生成配置。若需长期使用自定义训练配置，可将其另存后用`--config`直接启动训练器。

开始训练：

```bash
bash runtime/qwen25vl7b_stage3/train_projector.sh
```

输出目录是`projector_path`所在目录，主要文件为：

| 文件 | 内容 / 用途 |
|---|---|
| `best_state_dict.pt` | 验证MSE最低轮次的纯投影器权重，当前路径示例供攻击使用 |
| `best.pt` | 最优投影器、优化器、调度器状态及网络尺寸元数据 |
| `last.pt` | 最近一轮完整断点，用于继续训练 |
| `epoch_XX.pt` | 各轮完整断点 |
| `metrics.csv`、`loss_curve.png`、`train.log` | 训练/验证MSE、学习率、时间等 |
| `train_indices.json`、`resolved_config.json` | 抽样索引和训练配置；复现时还应固定图片文件清单 |

继续同一模型的训练时，增加配置中的`epochs`并显式指定断点，例如：

```bash
bash runtime/qwen25vl7b_stage3/train_projector.sh \
  --resume_checkpoint /srv/checkpoints/projector_new_victim/last.pt \
  --epochs 12
```

该训练器只在轮次边界恢复；恢复时加载原优化器/调度器状态，增加`epochs`不会自动重建原cosine计划。当前恢复运行的`metrics.csv`和曲线仅记录本次轮次，续跑前应保留原日志和指标副本。

评估已训练的最优投影器：

```bash
bash runtime/qwen25vl7b_stage3/train_projector.sh \
  --resume_checkpoint /srv/checkpoints/projector_new_victim/best.pt \
  --validate-only
```

这里`--validate-only`会加载受害模型、CLIP及投影器，在GPU上遍历验证集，并写入配置、抽样索引和日志。**它不是CPU配置检查；不带`--resume_checkpoint`则验证随机初始化的投影器。** 纯权重`best_state_dict.pt`不能作为该训练器的恢复断点，应使用`best.pt`或`last.pt`。

训练结束后再执行资产检查，并用一张图片试跑原始GHOST：

```bash
python tools/configure_server.py --paths server_paths.local.json --check-assets
bash runtime/qwen25vl7b_stage3/run_original_ghost.sh --validate-only
```

后一个命令只检查第三阶段配置和样本输入；还需按README执行预筛选与`--max-tasks 1`，才能验证真实投影token注入与梯度路径。

## 6. 直接使用原始训练器

下列命令展示原始代码的10轮、无调度配置，需先准备COCO train/val图片及模型访问权限：

```bash
python GHOST-master/train_projector.py \
  --model_name qwen \
  --coco_dir /srv/datasets/COCO \
  --cache_path /srv/model_cache \
  --save_projector_dir runtime/original_qwen_projector \
  --coco_subset 0.1 \
  --epochs 10 \
  --batch_size 1 \
  --val_batch_size 16 \
  --lr 0.001 \
  --weight_decay 0 \
  --context_dim 1024 \
  --hidden_dim 2048 \
  --scheduler none \
  --seed 42
```

显存紧张时可降低`--val_batch_size`。原始入口没有指定新本地受害模型路径或CLIP权重路径的命令行参数，`--cache_path`只是缓存目录；它会调用`utils.get_model`中的固定模型和`get_clip_model`中的默认OpenCLIP权重。更换为其他检查点必须适配加载代码，不能只改`--model_name`。

每轮输出的权重文件名包含`model_name`、学习率、epoch预算、context/hidden维度、子集比例和当前轮次。代码虽记录`best_val_loss`，却保存每一轮权重，并未单独导出`best.pt`；须根据验证损失自行选择轮次。

原始`--run_test_each_epoch`默认关闭，所指默认`GHOST-master/test_projector.py`在当前源目录中未找到。不要直接启用该选项；功能验证可参考已带入迁移包的`evals/validate_projector.py`。从仓库根目录运行它时需让`GHOST-master`可导入，例如：

```bash
PYTHONPATH="$PWD/GHOST-master${PYTHONPATH:+:$PYTHONPATH}" \
  python GHOST-master/evals/validate_projector.py --help
```

该验证器同样使用固定模型加载分支，且从纯权重文件名推断context/hidden维度；用于当前本地训练器的权重或新模型时，需先适配模型路径与权重配置读取。它比较物体存在/缺失问句下的原生图像路径与投影路径，不应把这个准确率当成投影器的攻击成功率。

## 7. 更换模型后如何接入第三阶段

更换受害模型后，以该模型视觉token为监督重新训练投影器；仅输出尺寸相同不能说明旧权重仍匹配。CLIP、受害模型权重版本、处理器或特征提取位置改变时，也要重新核对映射关系。

对于新架构，需要同时适配`utils.get_model_image_features`的教师特征提取和`get_model_inputs`的注入位置，以及第三阶段实际调用的受害模型接口。两者必须对应同一层特征语义；尺寸正确但层位置错误仍会破坏对齐。迁移包当前的自动配置与本地训练器仍只接入Qwen2.5-VL，原始训练器存在其他模型分支不代表第三阶段已经支持这些模型。

攻击使用`best_state_dict.pt`时，第三阶段配置中的`clip_dim`、`context_dim`、`hidden_dim`、`num_tokens`、`target_dim`必须与训练一致。当前`ProjectorRuntime`也支持读取含`projector_state_dict`及`projector_config`的完整`best.pt`断点，可据元数据重建网络；无论哪种格式，都仍需核对新受害模型的token接口。

当前ST-GHOST的干净嵌入初始化和原生图像反馈路径不经过此外部投影器；需要它的是原始GHOST对比中的嵌入攻击路径。训练新投影器不会自动完成其他架构的ST-GHOST适配，具体接口见[`VICTIM_ADAPTATION.md`](VICTIM_ADAPTATION.md)。

本次整理只添加说明及原始参考源码，没有启动训练，也没有改动现有实验算法或权重。
