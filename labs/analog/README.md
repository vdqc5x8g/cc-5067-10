# 同观测不同原音 · 削顶教学台

这套教学音轨要回答一个问题：**仅凭削顶后的采样，能证明什么、不能证明什么？**

## 一句话结论

两段不同的原始声，在同一低阈值 `T` 下产生**逐样本相同**的削顶观测；未削区原样保留。
把阈值抬到 `T↑`、恰好露出一个原本不可见的峰样本后，仍然与观测一致的候选集合会**严格缩小**——
被排除的一组延拓随之变化，而未受损的小信号段**逐样本不动**。

## 方法选择：为什么是构造，不是直接削干声

需求要求在“削已知干声”与“构造观测相同的多种原音”之间，**据实测选定**。仓库里的
`declipper.lab.methods.compare_construction_methods` 对两者做了同一个阈值下的逐样本测量：

| 方法 | 观测是否逐样本相同 | 最大逐样本差 | 不同样本数 |
|---|---|---:|---:|
| A. 直接削两段不同的自然干声变体 | 否 | 0.202 | 1599 / 1600 |
| B. 显式构造核心样值 | **是** | 0.000 | 0 / 1600 |

实测结论：直接削真实干声的不同变体无法保证观测一致；显式构造核心采样可以在削顶后得到
完全相同的观测。仓库据此选用 **方法 B**（`lab_report.json` 的
`construction_method_measurement.chosen == "construct_identical_observation"`）。

## 运行

```bash
# 生成成对练习音、观测、真值、参考区间与报告
python3 -m declipper lab build --spec labs/analog/paired_clip.spec.json --out examples/lab

# 只打印抬阈值盲比与排除结果
python3 -m declipper lab report

# 图形界面
python3 -m declipper panel --open-browser   # 打开 /lab
```

## 产物目录与真值隔离

```text
examples/lab/
├── observation/                    # 可交给重建器
│   ├── T0.observation.f32.wav
│   └── T1.observation.f32.wav
├── truth/                          # 密封：重建器从不读取
│   ├── low.truth.f32.wav
│   ├── high.truth.f32.wav
│   └── decoy-*.truth.f32.wav
├── reference/
│   ├── intact_intervals.json       # 未受损区样本区间
│   └── pair_observations.json      # 逐样本相同性证明
├── blind/                          # 引擎输出（输入只有观测）
│   ├── T0.conservative.f32.wav ...
├── spec.json                       # 可编辑合成源
└── lab_report.json
```

**隔离规则（由代码结构保证）：**

- `declipper.lab.engine.blind_reconstruct(observation, sample_rate, threshold, ...)`
  的参数里没有 `truth` / `clean` / `candidate`——它只接收观测数组。
- 真值只在 `declipper.lab.evaluation` 中用于**事后**评分与候选排除，绝不进入重建器。
- 测试 `test_engine_output_independent_of_hidden_truth` 会改变隐藏真值、保持观测不变，
  断言盲重建输出逐样本不变，以运行时证明引擎没有偷看真值。
- Web 接口默认返回的 payload **不含任何真值波形或逐候选误差**；只有显式点击
  “揭示真值” 才会附带 `truth` / `truth_audit`，且注明仅供事后核对。

## 可编辑合成源

`labs/analog/paired_clip.spec.json` 控制谐音、瞬态与阈值；界面同样可调。

| 字段 | 作用 |
|---|---|
| `body.fundamental_hz` / `body.harmonics` | 主体音高与各次谐波（幅度、相位、衰减） |
| `body.amplitude` / `body.decay` / `body.attack_seconds` | 主体包络 |
| `core.length` | 无法辨认的平台长度 |
| `core.low_anchors` / `core.high_anchors` | 两条不同原音的核心样值锚点（升余弦插值） |
| `core.reveal_position` / `core.reveal_value` | 抬阈值时唯一露出的样本 |
| `clip_threshold` (`T`) / `reveal_threshold` (`T↑`) | 低阈值与抬升阈值，必须 `0 < T < T↑` |
| `decoy_value` | 在 `T↑` 下被排除的诱饵候选峰值，必须 `> T↑` |

校验会在生成前拒绝任何无法满足“恰好露出一个样本”“诱饵必被排除”等性质的参数，
并给出具体是哪个核心样本越界。

## 界面能做什么

1. 拖动/微调谐音、瞬态、阈值，点“重新合成并盲比”，服务端真实重算。
2. 在 `T` 与 `T↑` 之间切换观测视图；蓝色平台标出**无法辨认的采样**，橙色竖线标出
   **抬阈值新露出的那一个样本**。
3. 切换 **保守修补** 与 **先验延拓** 两种重建假设；两者都重新削顶，播放
   “注入的猜测 `x − clip(x)`” 与“两种假设之差”，直观听到观测无法区分的那部分。
4. 查看抬阈值前后的候选集合，以及被排除的一组；确认小信号未受损区逐样本不变。
5. 需要时点“揭示真值”，才显示各候选与原音的事后误差与试听。

## 核心不变量（自动检查）

`lab_report.json.checks` 与 `threshold_lift_comparison(...).checks`：

- `candidate_set_strictly_shrank`：抬阈值后一致候选集合严格变小；
- `exactly_one_sample_newly_revealed`：新露出的样本恰好一个；
- `small_signal_unchanged_sample_for_sample`：未受损区在 `T` 与 `T↑` 下逐样本不变；
- `reconstruction_sees_only_observation`：重建输入只有观测。
