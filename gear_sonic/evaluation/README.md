# SONIC A3 可扩展 Isaac 评估

本模块用于在 IsaacLab/PhysX 中批量评估 SONIC A3 checkpoint，输出全量
`metrics_eval.json`。它支持按动作长度排序、单 GPU 分桶、batch 原子落盘、中断恢复、
离线聚合和结果校验。这里不负责视频渲染。

## 1. 环境与输入

```bash
# Activate a Python environment that already contains the compatible Isaac Lab
# installation and `pip install -e "gear_sonic/[training]"`.
cd /path/to/repository
python -c "import gear_sonic; print(gear_sonic.__file__)"
```

`gear_sonic` 必须解析到当前仓库。首次使用 Isaac Sim 前，使用者需要自行
阅读并接受 NVIDIA EULA 和隐私条款，然后在当前 shell 设置相应环境变量；本模块不会
替使用者接受条款。

输入要求：

- `--checkpoint`：`.pt` checkpoint；同目录应保留训练生成的 `config.yaml`。
- `--dataset`：直接包含 `.pkl` 文件的 motion-lib 目录。
- motion-lib PKL 应包含 `pose_aa`、`root_trans_offset`、`dof`、`root_rot` 和正确的
  `fps`。CSV 和原始 retarget PKL 不能直接输入本模块。
- 60 Hz、120 Hz 等数据均可使用；加载器根据 PKL 中的 `fps` 处理时间轴。错误或缺失
  的 `fps` 会导致动作速度错误，缺失时旧数据路径可能按 30 Hz 解释。

CSV 需要先转为 motion-lib PKL。例如 120 Hz CSV 转为 50 Hz motion-lib：

```bash
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
  --input <csv_dir> \
  --output <motionlib_output> \
  --individual \
  --robot a3_29 \
  --fps 50 \
  --fps_source 120 \
  --num_workers 8
```

`--individual` 会增加一层目录；评估时应指向实际含 `.pkl` 文件的那一层。

## 2. 直接运行

RTX 4090 单卡使用：

```bash
python -m gear_sonic.evaluation run \
  --config gear_sonic/evaluation/config/single_gpu_4090.yaml \
  --checkpoint <checkpoint.pt> \
  --dataset <motionlib_pkl_dir> \
  --output <output_dir>
```

首次跑新数据集，建议先加 `--dry-run`：

```bash
python -m gear_sonic.evaluation run \
  --config gear_sonic/evaluation/config/single_gpu_4090.yaml \
  --checkpoint <checkpoint.pt> \
  --dataset <motionlib_pkl_dir> \
  --output <output_dir> \
  --dry-run
```

`--dry-run` 会完成数据扫描和分桶，并输出每桶的 motion 数、真实帧数、padding、
batch 数、预计串行 simulation steps 和底层命令，但不会启动 Isaac。

## 3. YAML 配置

配置文件：

- `config/default.yaml`：通用默认值，关闭自动分桶。
- `config/single_gpu_4090.yaml`：24 GB RTX 4090 单卡分桶 preset，可直接使用。

关键结构如下：

```yaml
runtime:
  num_envs: 512
  encoder: a3_fast
  multi_thread_motion_loading: true
  empty_cache_freq: 20
  world_size: 1
  headless: true

persistence:
  enabled: true
  resume: true
  snapshot_every_n_batches: 5

dataset:
  sorting:
    mode: auto                 # off | on | auto
    order: ascending
    materialization: symlink   # symlink | copy
    auto_min_motions: 2048
    auto_padding_ratio: 1.5
    workers: 8
  long_motion:
    enabled: false
    max_frames: 4000
  bucketing:
    enabled: true
    buckets:
      - name: short
        max_frames: 3000
        num_envs: 2048
      - name: medium
        max_frames: 6000
        num_envs: 1024
      - name: bulk_long
        max_frames: 9000
        num_envs: 800
      - name: tail_long
        max_frames: 12000
        num_envs: 256
      - name: ultra_long
        max_frames: null
        num_envs: 40

evaluation:
  allow_partial: false
  termination_config: tracking/eval
  smpl_motion_file: dummy
```

实际 4090 `num_envs` 以 `single_gpu_4090.yaml` 为准。边界是左开右闭，例如
`medium` 表示 `3000 < frame_count <= 6000`。bucket 名必须唯一，有限
`max_frames` 必须严格递增，最后一个 bucket 必须使用 `null`。分桶模式目前只支持
`world_size: 1`，不能同时启用 `dataset.long_motion`。

修改阈值或 `num_envs` 后必须使用新的 `--output`，因为配置会进入恢复指纹。

## 4. 准备、恢复、聚合与验证

只准备和查看分桶数据：

```bash
python -m gear_sonic.evaluation prepare \
  --config gear_sonic/evaluation/config/single_gpu_4090.yaml \
  --dataset <motionlib_pkl_dir> \
  --output <output_dir>
```

中断后，重新执行完全相同的 `run` 命令即可恢复。工具会校验 checkpoint、数据集、
配置和 motion keys，跳过已经完整完成的桶，从首个未完成桶继续。

从已有 batch 文件重新聚合：

```bash
python -m gear_sonic.evaluation aggregate --output <output_dir>
```

仅检查中途已有结果时可以加 `--allow-partial`；partial 结果不能作为完整评估结果。

评估结束后验证：

```bash
python -m gear_sonic.evaluation validate --output <output_dir>
```

验证会检查 motion 覆盖率、重复 key、指标字段长度、有限数值，以及
`success_rate == mean(not terminated)`。失败时返回非零状态。

## 5. 输出

```text
<output>/
├── run_manifest.json
├── bucket_plan.json
├── eval.log
├── prepared_dataset/
│   ├── dataset_manifest.json
│   └── buckets/<bucket>/motions/*.pkl
├── buckets/<bucket>/
│   ├── run_manifest.json
│   ├── eval.log
│   ├── batch_results/rank_000/*.pt
│   ├── snapshots/rank_000/*.json
│   └── metrics/metrics_eval.json
└── metrics/metrics_eval.json
```

顶层 `metrics/metrics_eval.json` 是恢复为全局 canonical motion 顺序的完整结果。
主要指标：

- `success_rate`：未触发 eval termination 的 motion 比例。
- `progress_rate`：动作平均完成比例。
- `mpjpe_g`：世界坐标关节位置误差，单位 mm。
- `mpjpe_l`：pelvis 对齐关节位置误差，单位 mm；legs、VR 三点、上身和
  foot 子集也统一使用完整身体的 pelvis 作为 root。
- `joint_mse_g/joint_mse_l`：三维关节位置平方误差，单位 m²。

轨迹类指标只统计首次 eval termination 前的有效帧，不包含 termination 当帧或
自动 reset 后的新 episode。聚合按各指标自己的时间样本数计算：position 使用 `T`，
velocity 使用 `T-1`，acceleration 使用 `T-2`；旧 batch 缺少逐指标计数时会按该规则推导。

三维关节位置 RMSE 为 `1000 * sqrt(eval/all/joint_mse_g)`，不是电机角度 RMSE。

## 6. 常见问题

- 卡在 `Loading training config`：先确认已接受并设置 Isaac Sim EULA/隐私变量，再检查
  CUDA 初始化和 `gear_sonic` import 路径。
- CUDA 初始化卡住：先用
  `timeout 30 python -c "import torch; torch.cuda.init(); print('OK')"` 收集证据；不要直接
  终止其他用户的 MPS 进程。
- OOM：只降低发生 OOM 的桶的 `num_envs`，并换新的 output 目录。
- manifest 或 fingerprint 不匹配：不要混用两个实验的 batch 文件，改用新的 output。
- output 已是 `complete`：工具不会覆盖；重新评估必须选择新目录。
