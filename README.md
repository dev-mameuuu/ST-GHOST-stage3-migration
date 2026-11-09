# ST-GHOST 第三阶段迁移包

这是独立的第三阶段代码快照，可把**本目录作为GitHub仓库根目录**。原项目文件没有被移动或修改。`SOURCE_SNAPSHOT.json`记录原样复制的文件及其SHA-256；`BUNDLE_MANIFEST.json`记录整个迁移包的文件校验信息。

包含两个比较方法：

| 方法 | 入口 | 当前协议 |
|---|---|---|
| 原始GHOST对比 | `st-ghost/run_stage3_original_ghost_comparison.py` | 5个组合，每组400张；Multi-O；阈值触发生成，失败继续优化，最多4次 |
| ST-GHOST | `st-ghost/run_stage3_st_ghost.py` | 从同批2000张中均衡抽1000张；干净嵌入初始化；21个去噪条件独立优化；45步，学习率0.03；最终4个种子 |

ST-GHOST损失为 `2 L_victim + L_clean + L_ssim`，SSIM在256×256副本上计算、软阈值0.8，最终图片分辨率保持原设置。

**这里没有加入上一轮讨论的16GB显存改造。** 打包的是当前实验实现；迁移本身不会降低运行显存。

## 目录内容

```text
st-ghost/                 两个第三阶段入口及实际需要的第一、二阶段公共实现
  configs/                原始配置、投影器训练配置
  tests/                  已有协议、数值与断点恢复测试
GHOST-master/             实际依赖的原始实现、main.py、MIT LICENSE和README
configs/                  新服务器路径配置示例
migration_inputs/
  ranking_cache/          五个类别各Top-1000的CLIP排序缓存
  reference/              ST-GHOST所需的小型参数模板
  historical/             历史2000/1000张清单、协议与结果摘要（供对照）
  external_assets.json    另行准备的资产说明、CLIP/旧投影器校验值
tools/                    路径配置与迁移包完整性校验
docs/                     模型更换、实验口径及文件用途说明
tests/                    迁移配置工具的CPU测试
```

没有包含模型权重、COCO图片和标注、投影器权重、已生成攻击图、优化嵌入、完整实验输出、日志、虚拟环境或账号凭据。`.gitignore`已排除这些内容。历史清单保存图片ID、SHA-256和原Qwen预筛选记录，其中原服务器项目根路径已改为相对路径。

## 1. 上传GitHub

在**本迁移包目录内**执行以下命令。先在GitHub新建一个空仓库，再把最后一条命令中的地址替换为你的仓库地址。

```bash
git init
git add .
git status --short
git commit -m "Add Stage 3 GHOST and ST-GHOST comparison experiment"
git branch -M main
git remote add origin git@github.com:YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

本次整理没有初始化Git仓库，也没有执行上传。

## 2. 新服务器安装环境

```bash
git clone git@github.com:YOUR_ACCOUNT/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
python3.11 -m venv .venv-stage1
source .venv-stage1/bin/activate
python -m pip install --upgrade pip
```

先根据新服务器驱动安装相匹配的PyTorch与torchvision。源服务器组合为 `torch 2.8.0+cu128 / torchvision 0.23.0+cu128`；若目标环境支持此组合，可使用：

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-stage3.txt
python tools/verify_bundle.py
```

`requirements-stage3.txt`记录源服务器主要依赖版本；`requirements-stage1.txt`保留较宽泛的原依赖说明。不要同时安装两份，以免改变版本。新硬件、驱动和依赖组合仍需进行单样本试跑。

## 3. 另行准备模型和数据

参见 `migration_inputs/external_assets.json`。至少需要：

- 受害模型完整目录，包含处理器、分词器、配置及权重。
- OpenCLIP ViT-H/14 `laion2b_s32b_b79k`权重。复用排序缓存时必须使用同一份权重。
- stable-diffusion-2-1-unclip完整目录，包括UNet、VAE及其他管线组件。
- OWLv2检测模型，用于原始GHOST的目标缺失验证。
- COCO train2017图片和两个标注：`instances_train2017.json`、`captions_train2017.json`。
- 与新受害模型匹配的投影器权重；ST-GHOST干净初始化本身不经过投影器，但原始GHOST比较需要。

COCO目录可以是：

```text
COCO/
  train2017/000000000009.jpg   # 也支持 images/train2017/
  annotations/instances_train2017.json
  annotations/captions_train2017.json
```

若要重新预筛选，不能只准备旧的1000张测试图片；还需要排序缓存涉及的候选图片。最方便的方式是另行迁移完整COCO数据集。

旧Qwen投影器位于原项目的 `outputs/projector_qwen/best_state_dict.pt`，约76.6MiB，本包没有复制它。新受害模型通常需要重新训练投影器；参数匹配也不意味着旧投影器能直接复用。

投影器训练方法、原始GHOST与本地训练器的参数差异、COCO验证集要求及新模型接入步骤见 [docs/PROJECTOR_TRAINING.md](docs/PROJECTOR_TRAINING.md)。原始`GHOST-master/train_projector.py`及物体问答验证器也已带入包中供核对。

## 4. 配置新服务器路径

```bash
cp configs/server_paths.example.json server_paths.local.json
```

编辑`server_paths.local.json`，填写新服务器真实路径和新的`run_name`。`target_dim`、`num_tokens`必须匹配受害模型和投影器。当前接口支持Qwen2.5-VL；其他架构请先按 `docs/VICTIM_ADAPTATION.md` 适配。

```bash
python tools/configure_server.py --paths server_paths.local.json
```

该命令只生成配置和启动脚本，不加载模型、不启动实验。输出位于`runtime/<run_name>/`，由Git忽略。下面以示例名称`qwen25vl7b_stage3`为例；实际运行时请替换为你的名称。

若需要训练新投影器，可先运行：

```bash
bash runtime/qwen25vl7b_stage3/train_projector.sh
```

训练器输出`best_state_dict.pt`，因此`projector_path`应指向该文件。训练需要模型以及COCO的`train2017`和`val2017`图片。训练器的`--validate-only`是GPU验证，不是静态配置检查；验证已训练权重须同时传入`--resume_checkpoint .../best.pt`，详见训练说明。

资产准备完成后再检查：

```bash
python tools/configure_server.py --paths server_paths.local.json --check-assets
```

检查内容包括本地路径、模型类型/隐藏维度、CLIP权重SHA-256和排序候选图片是否存在；不会执行推理，也不会验证投影器张量形状。配置之后不要移动仓库目录；移动后请使用新`run_name`重新生成配置。

## 5. 开始第三阶段比较

先准备原始GHOST的干净图池：

```bash
bash runtime/qwen25vl7b_stage3/run_original_ghost.sh --validate-only
bash runtime/qwen25vl7b_stage3/run_original_ghost.sh --prepare-only
```

`--prepare-only`会加载受害模型，在相同CLIP排序候选中重新预筛选，并生成该受害模型的2000张清单。**旧Qwen的“No”记录不能直接作为新受害模型的预筛选结果。** 新模型筛选出的图片可能不同；若要严格固定旧2000张，应额外设计逐图重测与分母统计，此包不会伪造这一步。

运行原始GHOST：

```bash
bash runtime/qwen25vl7b_stage3/run_original_ghost.sh --max-tasks 1
bash runtime/qwen25vl7b_stage3/run_original_ghost.sh
```

清单准备完成后，ST-GHOST即可使用它；不需要等待原始GHOST的2000份攻击全部完成。但同一GPU上请顺序运行，避免两个方法竞争显存。

```bash
bash runtime/qwen25vl7b_stage3/run_st_ghost.sh --validate-only
bash runtime/qwen25vl7b_stage3/run_st_ghost.sh --prepare-only
bash runtime/qwen25vl7b_stage3/run_st_ghost.sh
```

ST-GHOST当前入口没有`--max-tasks`参数。它先完成1000份45步优化，再评估g0，最后生成并评估另外三个种子。四张中至少一张使受害模型回答Yes即计为任务成功。

结果分别保存在：

```text
runtime/<run_name>/original_ghost/
runtime/<run_name>/stghost/optimization/
runtime/<run_name>/stghost/four_seeds/
```

可在`screen`或`tmux`中运行生成的启动脚本，或用`nohup bash ... > ... 2>&1 &`。包内原有`wait_stage3_*.py`和`*_service.sh`作为源实现保留，但它们仍启动原默认配置、检查约46GiB空闲显存；**不要直接用它们启动新路径配置的实验**。

## 6. 测试与迁移边界

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s st-ghost/tests -p 'test_stage3*.py'
python -m unittest discover -s st-ghost/tests -p 'test_stage2_level_oo_generation_feedback_victim_evaluation.py'
```

这些是CPU或模拟测试，不证明新受害模型的GPU执行、梯度或显存一定正确。

本包用于迁移代码和开启新受害模型实验，**不包含恢复旧Qwen实验所需的完整断点**。现有断点绑定绝对路径、配置与代码哈希；更换模型后应新建输出，不应覆盖或修改旧签名来强行续跑。

更换模型的具体改动见 `docs/VICTIM_ADAPTATION.md`；比较口径和代码依赖见 `docs/EXPERIMENT_NOTES.md`。
