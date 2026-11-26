# Declipper Prototype — 约束式削波重建与母带试听面板

初版假设已知对称硬削波阈值 `T`。它不是“把顶端圆滑一下”的插值器，也不靠统一降低音量掩盖缺失峰形；求解器把未削样本作为等式约束，把饱和区作为幅度不等式，在明确的平滑/带限/副麦先验下求解潜在波形。

本仓库只使用 Python 3.10+ 标准库；WAV 读写、FFT、最小二乘、投影梯度 QP、HTTP 服务和前端均无需安装第三方包。

## 1. 图形入口（主交付）

```bash
python3 -m declipper demo     # 可选：重新生成 examples/generated 下的 WAV/JSON
python3 -m declipper panel    # 默认 http://127.0.0.1:8765
```

也可以自动打开浏览器：

```bash
python3 -m declipper panel --open-browser
```

浏览器打开 <http://127.0.0.1:8765> 后可以：

1. 选择内置合成弦音或打开自己的 WAV；
2. 在波形上按住指针框选削波区；
3. 选择自动、保守、谐波延拓或双轨模式；
4. 查看停止原因、迭代次数、约束违反、再削波误差、未削样本误差、边界曲率和峰高范围；
5. 在结果区叠加查看重建波形，切换主解/保守解/延拓解/双轨解；
6. 用两个音频播放器或下载 32-bit float WAV 做声音对照；
7. 可选运行守卫响度归一，查看未知峰高为何不能被当成真实余量；周期歧义时也可勾选“强制歧义时延”，结果会保持未认证并被归一守卫拦截正向补偿；
8. 加载不同削波程度评估、周期偏移检查和失败样例报告。

所有恢复和 WAV 编码都由本地 Python 服务实际运算；前端只负责框选、绘图和播放返回的音频。请求异常、阈值错误、时延歧义等会显示在面板顶部红色提示区。

## 2. 命令行仅作为辅助

```bash
# 生成全部演示声音、评估和失败样例
python3 -m declipper demo --out examples/generated

# 单轨恢复并导出 32-bit float WAV
python3 -m declipper restore input.wav \
  --threshold 0.34 \
  --start-sample 204 --end-sample 211 \
  --output restored.f32.wav \
  --report restored.json

# 正确副麦可自动识别时延；只有唯一相关峰时才给认证双轨单解
python3 -m declipper restore clipped.wav \
  --threshold 0.34 \
  --start-sample 204 --end-sample 211 \
  --secondary secondary.wav \
  --output dual.f32.wav

# 强制使用未认证时延时，必须显式承担风险
python3 -m declipper restore clipped.wav --threshold 0.34 \
  --secondary secondary.wav --shift 99 \
  --force-unverified-dual \
  --output forced.f32.wav

# 周期偏移 + 响度归一安全检查
python3 -m declipper period-check --out examples/generated
```

## 3. 数学约束与“能证明什么”

硬削波模型为：

\[
y_n=\operatorname{clip}(x_n;-T,T).
\]

因此：

- 若 `-T < y_n < T`，则 `x_n = y_n`，求解器不改动这些样本；
- 若 `y_n = T`，则 `x_n >= T`；
- 若 `y_n = -T`，则 `x_n <= -T`；
- 将负平台变换为 `h_n=-x_n` 后，两类平台都满足 `h_n >= T`。

求解器在平台变量上最小化显式先验能量：

\[
E(x)=w_1\sum (x_{n+1}-x_n)^2
+w_2\sum (x_{n-1}-2x_n+x_{n+1})^2
+w_t\sum (x_n-t_n)^2
+w_r\sum (x_n-\operatorname{sgn}(y_n)T)^2.
\]

其中：

- 一阶/二阶差分是最小斜率/最小曲率先验；
- `t_n` 可以是局部正弦拟合的谐波延拓模板，也可以是已对齐、增益校准的副麦模板；
- 盒约束投影保证饱和区幅度不等式；
- 谱投影梯度求解器记录迭代数、停止原因、投影梯度和最大约束违反。

### 峰高范围必须分层表述

面板和 JSON 同时给出：

1. **观测可证明范围**：`[T, +∞)`。  
   削平的顶部没有保留原峰值的全部信息；仅凭硬削波样本不能证明有限上界。
2. **先验峰高外包络**：根据未削样本经验斜率/二阶差分上限，构造左右可达包络，得到有限但较宽的 `[T, U_prior]`。这不是观测证明。
3. **最小努力点估计**：约束 QP 的峰高，可能明显低于真值。
4. **认证双轨区间**：只有副麦时延唯一可识别且对齐时，才把副麦作为强模板求单一受约束解，并依据对齐残差给经验峰高上界。

## 4. 内置场景

`python3 -m declipper demo` 会构造两段真实峰值不同的合成弦音，并施加同一阈值 `T=0.34`：

| 轨道 | 真实全局峰值（仅构造/评估知道） | 选中的 8 个样本 | 单轨点估计 | 单轨先验上界 | 认证双轨峰值 |
|---|---:|---|---:|---:|---:|
| Amber 低峰值弦音 | 0.72 | 八个 `+0.34` | 0.607 | 1.84 | — |
| Silver 高峰值弦音 | 1.18 | 八个 `+0.34` | 0.747 | 2.84 | **1.176** |

这八个样本逐样本完全相同，说明同一个短平台本身无法区分两段弦音的真实峰高。低峰轨道的点估计偏低、高峰轨道的点估计也偏低，但宽先验范围覆盖真值；加入时延唯一的未削副麦后，高峰轨道得到接近真值的单一受约束解。

## 5. 副麦周期歧义与响度归一器

双轨对齐只在相关峰唯一、相关性足够高且没有近似等值得周期候选时标记为 `identified/certified`。精确周期信号会产生相隔一个或多个周期的等值相关峰，此时状态为 `ambiguous_delay`，后端不会导出认证双轨单解。

`period-check` 会强制选择偏移一个宏观周期的错误时延，并把这个未认证结果传给两种归一策略：

- **朴素点峰值策略（仅作对照）**：把推测峰高当成真实余量，给出约 `+1.0 dB` 的补偿，自以为有约 `4.2 dB` 以上余量；把同一增益事后施加到干净真值，真实峰值升至约 `1.329`，出现 19 个超过 `0.98` ceiling 的样本、15 个超过数字满刻度的样本。
- **守卫策略（默认）**：`certified_headroom_db = null`，不把未认证峰高用于正向补偿，增益不超过 `1.0`，并报告被阻止的 makeup gain。

这不是说守卫归一能恢复缺失峰形；它只是拒绝把无法断定的峰高误当成真实余量。

## 6. 不同削波程度的实际评估

干净波形只在 `evaluate_restoration` 中用于事后评估，不传入 `restore_signal`。同一音乐峰周期在不同阈值下的本地平台指标（生成于 `examples/generated/evaluation/evaluation_report.json`）：

| T | 修复前 RMSE | 修复后 RMSE | 区 SNR dB | 上下文 LSD dB | 先验范围覆盖真值 |
|---:|---:|---:|---:|---:|:---:|
| 0.55 | 0.462 | 0.394 | 8.04 | 5.89 | 是 |
| 0.40 | 0.524 | 0.288 | 9.83 | 5.40 | 是 |
| 0.27 | 0.573 | 0.197 | 12.29 | 7.26 | 是 |
| 0.16 | 0.668 | 0.249 | 10.28 | 8.34 | 是 |

“听感”不能只由 RMSE 代表，因此面板可直接 A/B 听削波版与重建版；报告中另给上下文对数谱距离作为客观代理。正式主观听感仍需人耳试听，本仓库不声称 LSD 等同于听感测试。

## 7. 明确的失败样例

### 超长连续饱和区

`examples/generated/failure_long_plateau/`

- 41 个连续正平台样本，超过由采样率和局部周期得到的可靠先验长度 16；
- 同时导出：
  - `long_conservative.f32.wav`：最小先验能量的保守解；
  - `long_continuation_speculative.f32.wav`：谐波延拓推测解；
- 报告包含两者的真值误差、再削波误差、未削样本误差、边界检查和优化器日志。

该样例中严重削波使谐波幅度本身欠定，延拓模板反而低估峰形；保守解的区 RMSE 更好，但这不是普遍保证。面板会明确警告两个解必须并列看待，不能把其中一个包装成“找回原声”。

### 副麦时延不可识别

`examples/generated/failure_ambiguous_delay/`

- 精确周期音的相关峰在 `61, 21, 101, ...` 等周期偏移处近似等值；
- JSON 状态为 `ambiguous_delay`；
- 策略是拒绝认证双轨单解，只能给单轨范围，或由用户强制偏移并永久标记为未认证。

## 8. 产物目录

```text
examples/generated/
├── manifest.json
├── demo_report.json
├── *.clipped.f32.wav
├── *.single_track_restored.f32.wav
├── *.dual_restored.f32.wav
├── *.unclipped_secondary.f32.wav
├── evaluation/
│   ├── evaluation_report.json
│   └── degree_t*.wav
├── period_offset/
│   ├── period_offset_check.json
│   ├── wrong_period_alignment_restored.f32.wav
│   ├── naive_loudness_master.f32.wav
│   └── guarded_loudness_master.f32.wav
└── failure_ambiguous_delay/ + failure_long_plateau/
```

重建母带默认写 32-bit float WAV，因为真实潜在峰可能超过 `1.0`；这避免在导出阶段再次被 16-bit PCM 硬截断。若需要普通 PCM16 试听文件，可使用 CLI 的 `--pcm16 --safe-preview-gain`，但那只是监听转换，不是修复算法的一部分。

## 9. 测试

```bash
python3 -m unittest discover -s tests -v
```

覆盖：

- WAV float 往返；
- 平台检测；
- 未削样本零误差、再削波零失配和不等式违反；
- 先验范围覆盖合成真值；
- 唯一/周期歧义时延；
- 守卫响度归一；
- 长平台双候选与警告；
- 非法阈值类型化错误。

## 10. 仍然存在的限制

- 初版假设阈值精确且对称；阈值未知、非对称、量化噪声和麦克风削波需要扩展模型。
- 经验斜率/曲率包络给出的是先验外包络，不是严格声学证明。
- 谐波延拓只适用于局部稳定的带限/谐波素材；长平台可能欠定。
- 副麦假设采样率一致、可做标量增益校准且没有复杂运动/滤波失配。
- 没有任何算法能从任意严重削波中保证恢复原始峰形；观测中不存在的信息只能由先验猜测，并必须展示不确定性与失败样例。
