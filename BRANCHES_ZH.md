# humanoid-lab 分支说明

本文档记录仓库各分支的用途、继承关系、主要修改和当前验证状态，方便后续训练、排错与合并代码。

更新时间：2026-09-02

## 一、分支继承关系

```text
fix-local-command-import
└── add-linglong-robot
    └── add-linglong-locomotion-task
```

后面的分支包含前面分支的全部修改。例如，`add-linglong-locomotion-task` 同时包含本地导入修复、灵龙 URDF/网格资源和灵龙强化学习任务。

## 二、各分支用途

### 1. `fix-local-command-import`

这是当前仓库的默认分支，也是后续两个灵龙分支的基础。

主要修改：

- 修复 locomotion 奖励和终止条件中的命令类导入路径。
- 将外部项目路径 `whole_body_tracking...CustomUniformVelocityCommand` 改为本项目相对导入 `.commands`。
- 避免在本地安装和运行时因为缺少 `whole_body_tracking` 包而导入失败。

代表提交：

- `1854a1d`：`fix: use local locomotion command imports`

适用场景：

- 运行仓库原有的 L7 任务。
- 作为后续新增机器人和任务的稳定基础。

### 2. `add-linglong-robot`

该分支在 `fix-local-command-import` 基础上加入灵龙机器人模型资源，但尚未加入可训练任务。

主要修改：

- 加入 `LingLong2.0.urdf`。
- 加入 `LingLong_L1_V1.4.urdf`。
- 加入灵龙机器人各刚体对应的 STL 网格文件。

代表提交：

- `fb92c5e`：`Add LingLong robot URDF and meshes`

适用场景：

- 单独检查 URDF、关节、惯量、碰撞体和外观网格。
- 对机器人模型资源进行修改，但不希望同时改动训练任务。

注意：

- 该分支只有模型资源，不能直接使用 `Locomotion-Flat-LingLong-30Dof-v0` 训练。

### 3. `add-linglong-locomotion-task`

这是目前用于灵龙 30 自由度机器人强化学习训练的工作分支，继承 `add-linglong-robot` 的全部内容。

主要修改：

- 新增灵龙 30 自由度 Isaac Lab 机器人配置和执行器分组。
- 新增任务注册：`Locomotion-Flat-LingLong-30Dof-v0`。
- 保留原有注册任务，不修改 L7 任务名称和行为。
- 新增灵龙环境配置、观测、动作、奖励、终止条件和 RSL-RL PPO 配置。
- 新增 `scripts/check_linglong_asset.py`，用于检查 URDF 导入、刚体数量、关节数量和基本物理稳定性。
- 修复多躯干接触终止条件的张量形状问题：同时检查 `base_link`、`waist_yaw_link`、`waist_pitch_link` 时，为每个并行环境返回一个终止布尔值。

代表提交：

- `a694375`：新增灵龙 locomotion 训练任务。
- `dd2ea61`：修复灵龙多刚体接触终止条件。

当前已注册任务：

- `Locomotion-Flat-L7_29Dof-v0`
- `Locomotion-Flat-LingLong-30Dof-v0`
- `Tracking-Flat-L7_29Dof-v0`

当前验证状态：

- URDF 成功导入：31 个刚体、30 个关节。
- 仿真数据保持有限值，未发现 NaN、关节爆炸或零件分离。
- 无策略的中立姿态 PD 检查中，机器人会在数秒后摔倒；这是未训练状态，不代表训练任务异常。
- 64 个并行环境、2 个 PPO iteration 的烟雾测试已通过。
- 烟雾测试累计运行 3072 个 timesteps，训练正常结束，退出码为 0。
- Actor 输出维度为 30，与机器人 30 个受控关节一致。

注意：

- 2 轮烟雾测试只证明“环境能够启动、采样、计算奖励并更新网络”，不能证明机器人已经学会站立或行走。
- 正式长时间训练前，建议先测试 256、512、1024 等环境数量，选择 RTX 2080 Ti 不溢出显存且吞吐量较高的配置。
- 目前灵龙任务仍处于独立开发分支，尚未合并到默认分支。

## 三、常用切换命令

查看当前分支：

```bash
git branch --show-current
git status --short
```

切换到灵龙训练分支并拉取最新修改：

```bash
git switch add-linglong-locomotion-task
git pull --ff-only origin add-linglong-locomotion-task
```

切换到只包含灵龙模型资源的分支：

```bash
git switch add-linglong-robot
```

切换回默认基础分支：

```bash
git switch fix-local-command-import
```

## 四、当前建议

现阶段继续在 `add-linglong-locomotion-task` 分支上完成灵龙机器人的环境数量性能测试、短程训练和可视化验证。确认训练稳定、奖励趋势合理且机器人行为正常后，再决定是否合并到默认分支。
