【目标】
用你已有的绿色贴表面点（ScanSurfaceSnapper 的结果）直接触发 MRMotifs 的 ShockWaveEffect.prefab，
复用“球爆开/冲击波扫描房间”的视觉，同时避免和原来的扔球+二次环境射线判定逻辑冲突。

============================
一、为什么这是最佳桥接（避免逻辑矛盾）
============================
1) 绿球已经完成“贴表面/选爆点”
   -> 不需要再让 ShockWaveOrbMotif 做一遍环境 raycast，否则会出现双重判定、位置不一致。

2) 直接复用 ShockWaveEffect.prefab
   -> 该 prefab 内含 ShockWaveEffectMotif，会自动扩张扫描球并销毁自己。
   -> 视觉链路（Depth API + shader）保持 MRMotifs 原样，不破坏深度效果。

3) 桥接器只做“翻译层”
   -> 读取绿球位置/法线（可选）
   -> 实例化 shockwave effect
   -> 可选播放音效与 pitch ramp
   -> 不接管参考系、不接管表面吸附、不接管 chunk/网格系统

============================
二、场景桥接方案（推荐）
============================
保留：
- SessionReferenceFrame（固定参考系）
- ScanWorkCenter（黄色调度中心）
- ScanSurfaceSnapper（绿色贴表面点）
- [MR Motif] EnvironmentDepth（深度矩阵/helper/全局深度链路）
- ShockWaveEffect.prefab（MRMotifs 原 prefabs）

暂时停用（避免重复触发/重复判定）：
- [MR Motif] Orb Spawner（先 Disable）
- ShockWaveOrbMotif 的发射/吸附路径（先不参与）

新增：
- 一个空物体：ScanShockwaveBridge（建议放在 SessionReferenceFrame 下）
- 挂脚本：ScanShockwaveBridge.cs

============================
三、Inspector 绑定（一步一步）
============================
1) 在场景中创建空物体：ScanShockwaveBridge
2) 挂上脚本 ScanShockwaveBridge.cs

3) 绑定字段：
- sessionReferenceFrame -> 拖你的 SessionReferenceFrame
- surfaceSnapPoint      -> 拖绿色 marker（ScanSurfaceSnapper 输出点）
- surfaceNormalSource   -> 可先也拖绿色 marker（没有单独法线源就留空/同上）
- shockWaveEffectPrefab -> 拖 MRMotifs 的 ShockWaveEffect.prefab 上的 AudioSource 组件（注意不是拖纯 GameObject，如果 Inspector 类型是 AudioSource）
- scanTriggerClip       -> 可选（想覆盖原 prefab 的音效时再填）
- validityIndicator     -> 可选（如果你有“只有 snap 有效才亮/显示”的对象，把它拖进来）

4) 建议初始参数：
- upMode = AlignToSurfaceSnapUp
- alignForwardToReferenceForward = true
- spawnOffsetAlongUp = 0.01 ~ 0.02
- parentMode = NoneWorldSpace（推荐先用世界空间，最不容易受父物体移动影响）
- enableManualKey = true
- manualTriggerKey = G
- cooldown = 0.25
- requiredStableTime = 0.12
- stablePositionTolerance = 0.01
- requireSurfaceSnapPointActive = true
- requireStableBeforeTrigger = true

============================
四、如何验证“链路通顺”（冒烟测试）
============================
1) 运行场景，确保绿色 marker 能稳定贴到地面/床面/台阶面
2) 按 G（或调用 TriggerShockwave()）
3) 观察：冲击波是否从绿球附近爆开，并扫描环境深度边界
4) 把绿球移动到不同高度（如床上），再次触发
5) 观察：爆点高度是否跟随绿球，而不是回到原来的 MRMotifs orb 位置

通过标准（建议）
- 地面触发正确
- 高处触发正确
- 连续多次触发不漂移
- 停用 Orb Spawner 后仍然能触发冲击波（说明桥接独立成立）

============================
五、常见问题与排查
============================
Q1: 按键没反应
- 确认 shockWaveEffectPrefab 已绑定（AudioSource 类型）
- 确认 surfaceSnapPoint 已绑定
- 确认 cooldown 未卡住
- 先把 requireStableBeforeTrigger=false 测一下（排除稳定门控）

Q2: 爆点位置正确但冲击波方向怪
- 先试 upMode = AlignToReferenceUp
- 或 upMode = KeepPrefabRotation（仅验证位置）

Q3: 爆点埋进表面看不清
- 调大 spawnOffsetAlongUp（如 0.02~0.03）

Q4: 似乎还是在原来的 orb 逻辑触发
- 先禁用 [MR Motif] Orb Spawner
- 确认你按的是桥接器按键，而不是原场景的手柄按钮逻辑

============================
六、后续升级路线（不推翻当前桥接）
============================
阶段A（现在）：绿球 -> ShockWaveEffect（验证链路）
阶段B：绿球 -> 自定义 ScanCoverDriver + shader（通用扫描参数）
阶段C：扫描后留痕 -> chunk/网格化覆盖 -> 持久化/anchor 参考系

这意味着当前桥接器不是废代码，而是“过渡层”，用于确保你后续升级不会伤筋动骨。
