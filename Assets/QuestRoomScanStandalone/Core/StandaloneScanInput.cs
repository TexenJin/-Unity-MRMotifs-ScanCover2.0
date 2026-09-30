using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// QRS 独立测试链的右手柄输入：
    ///   扳机       = 开始 / 继续采集共享 TSDF
    ///   A          = InfiniTAM 档封口完整原因账、冻结 TSDF 并排空网格尾随；即时外壳档中切换世界锁定定格
    ///   Y          = 在 64³ / 32³ / 16³ 只读回放档之间循环
    ///   B          = 只导出并清空当前档，不清共享 TSDF
    ///   右摇杆按下 = 一键回到真实 TSDF 的生产纸皮观察档；
    ///   右握把+右摇杆按下 = 纸皮→支撑真值→三角粗皮→HERA 的诊断视图循环。
    ///   右摇杆方向+按下 = 追责/性能热键：上=实时轨 / 左=胶冻↔原冻（仅空卷） / 下=融合 20↔10Hz / 右=深度预处理(双边+缘洗)
    ///   左摇杆按下 = 支撑真值档切拓扑审计/外皮片实体；纸网合流档切纸主网格/旧真边对照
    ///   右握把+左摇杆按下 = 只切线框前的整面深度预绘（网格仍显示）
    ///   左摇杆上+按下 = 源头时序滤波开关（盯墙养绿 A/B 热键，HUD 闸行 时开/时关 回显）
    ///   左摇杆下+按下 = 第一阶段纯白几何 / 原状态色切换（仅显示层）
    ///   左摇杆右+按下 = 32³融合管理块线框开关（青稳/黄热/红双热/洋红已解冻）
    ///   左摇杆左+按下 = 空卷时切换胶冻 / 原冻（与右摇杆左同义，便于实机操作）
    ///   左握把+X = 平台前处理 / QRS 后处理双路逐帧采集开关
    ///   普通 X     = 裁决海→BB点云→即时外壳→种子平面→壳纸合流→合流仅纸→关
    /// 每次按键给一下短震动作为反馈。
    /// </summary>
    public class StandaloneScanInput : MonoBehaviour
    {
        [SerializeField, Tooltip("接收输入的手柄")]
        private OVRInput.Controller controller = OVRInput.Controller.RTouch;

        private float _prevTrigger;

        private void Update()
        {
            var scanner = StandaloneRoomScanner.Instance;
            if (scanner == null) return;

            // 扳机：模拟量上升沿判定。
            // Meta OpenXR 后端不报 Button.PrimaryIndexTrigger 的数字态（扳机量轴正常，
            // 数字点击永远是 false），所以必须用轴阈值 + 上升沿，不能用 GetDown。
            float trig = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, controller);
            bool triggerDown = trig > 0.75f && _prevTrigger <= 0.75f;
            _prevTrigger = trig;

            // 扳机：开始 / 继续
            if (triggerDown)
            {
                scanner.NotifyInput("扳机");
                if (!scanner.IsScanning && !scanner.IsChunkAbFrozen)
                {
                    _ = scanner.StartScanningAsync();
                    Pulse();
                }
            }

            // A：即时壳保持原定格语义；InfiniTAM 生产档先封口完整原因账，
            // 再冻结 TSDF、排空分块提取/提交队列并写出尾随验证小票。
            if (OVRInput.GetDown(OVRInput.Button.One, controller))
            {
                scanner.NotifyInput("A键");
                if (scanner.TryToggleInstantShellFreeze())
                {
                    Pulse();
                }
                else if (scanner.TryBeginInfiniTamMeshTailValidation())
                {
                    Pulse(0.5f, 0.5f);
                }
                else if (scanner.IsChunkAbExperimentEnabled)
                {
                    scanner.FreezeChunkAbTsdf();
                    Pulse();
                }
                else if (scanner.IsScanning)
                {
                    scanner.PauseScanning();
                    Pulse();
                }
            }

            // B：种面“待清卷”是一次明确的生产换卷事务，优先级高于
            // HERA/A-B 派生档导出；否则场景常驻 enableFrozenChunkAbExperiment
            // 会把 B 送进尚未冻结的回放分支并静默返回，HUD 要求按 B 却毫无反应。
            // 其他时候仍保持原语义：A/B 模式只导出并清当前派生档。
            if (OVRInput.GetDown(OVRInput.Button.Two, controller))
            {
                scanner.NotifyInput("B键");
                bool seedNeedsClear = scanner.DepthCapture != null &&
                                      scanner.DepthCapture.SeedPlaneAwaitClear;
                if (seedNeedsClear)
                {
                    scanner.NotifyInput("B键：保存旧账并为种面清卷");
                    scanner.StopAndClearScan();
                    Pulse(0.5f, 0.5f);
                }
                else if (scanner.IsChunkAbExperimentEnabled)
                {
                    scanner.ExportAndClearActiveChunkAbGear();
                    Pulse(0.5f, 0.5f);
                }
                else if (scanner.IsScanning || scanner.HasStarted)
                {
                    scanner.StopAndClearScan();
                    Pulse(0.5f, 0.5f);
                }
            }

            // 右摇杆：按下=单色/状态着色；**推方向再按下**=性能二分热键（实机：
            // 采集 15-24fps GPU U 91%，冻结满显示 73fps——猪在采集链路 GPU 侧）：
            //   上=实时轨开关（提取+过滤+回读churn）
            //   左=原冻→胶冻→裁冻准入（只允许尚未开扫的空卷切换；纸皮档不变）
            //   下=融合 20↔10Hz（TSDF 积分量减半）
            if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstick, controller))
            {
                Vector2 stick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, controller);
                if (stick.y > 0.5f)
                {
                    scanner.NotifyInput("摇杆上");
                    scanner.ToggleLiveTrack();
                }
                else if (stick.x < -0.5f)
                {
                    scanner.NotifyInput("摇杆左");
                    scanner.ToggleGunGelGuardedFusionExperiment();
                }
                else if (stick.x > 0.5f)
                {
                    scanner.NotifyInput("摇杆右");
                    scanner.ToggleDepthPreprocessing();
                }
                else if (stick.y < -0.5f)
                {
                    scanner.NotifyInput("摇杆下");
                    scanner.ToggleIntegrationRate();
                }
                else
                {
                    // 无组合键永远回真实生产纸皮，避免录屏误落在 HERA/粗皮/点层。
                    // 原来的路线诊断循环保留在右握把组合键，不影响故障排查能力。
                    float rightGrip = OVRInput.Get(
                        OVRInput.RawAxis1D.RHandTrigger,
                        OVRInput.Controller.RTouch);
                    bool rightGripHeld =
                        OVRInput.Get(OVRInput.RawButton.RHandTrigger,
                                     OVRInput.Controller.RTouch) ||
                        rightGrip > 0.35f;
                    if (!rightGripHeld)
                        scanner.ShowProductionPaperView();
                    else if (scanner.IsChunkAbFrozen)
                        scanner.ToggleChunkAbDisplayMode();
                    else
                        scanner.ToggleMeshDisplay();
                }
                Pulse();
            }

            // 左手 Y：仅冻结后切换当前回放档，三档不并行常驻。
            if (OVRInput.GetDown(OVRInput.RawButton.Y))
            {
                scanner.NotifyInput("Y键");
                scanner.CycleChunkAbGear();
                Pulse();
            }

            // 左手 X：呼出/收起融合前 BB 反投影留痕；开启时只显示反投影点。
            // 左握把+X：最小双路逐帧采集（平台前处理 + 同帧 QRS 后处理），不改显示。
            if (OVRInput.GetDown(OVRInput.RawButton.X))
            {
                // 不走 PrimaryHandTrigger 虚拟重映射：Quest/OpenXR 实机曾出现
                // 握把已按住但读数仍为 0，导致组合键误落到普通 X 点阵分支。
                // 直接读左握把物理 RawButton，并用 RawAxis 0.35 作双保险。
                float leftGrip = OVRInput.Get(
                    OVRInput.RawAxis1D.LHandTrigger,
                    OVRInput.Controller.LTouch);
                bool leftGripHeld =
                    OVRInput.Get(OVRInput.RawButton.LHandTrigger, OVRInput.Controller.LTouch) ||
                    leftGrip > 0.35f;
                if (leftGripHeld)
                {
                    scanner.NotifyInput($"左握把+X 握{leftGrip:0.00}");
                    scanner.TogglePairedDepthFrameCapture();
                }
                else
                {
                    // 普通 X：关 → 枪胶裁决海 → BB 反投影 → 即时外壳 →
                    // 种子平面 → 壳纸合流 → 合流仅纸 → 关。
                    scanner.ToggleCoverageMarkers();
                    // ToggleCoverageMarkers 内部会写一次提示，因此诊断握值必须放在
                    // 它之后，确保实机截图能看见而不是被“BB反投影”提示覆盖。
                    scanner.NotifyInput($"X点阵 握{leftGrip:0.00}");
                }
                Pulse();
            }

            // 左摇杆按下：线框/实体切换。右握把+左摇杆按下：
            // 只切线框前的整面深度预绘，可见线框与所有后台路径不变。走全局 shader 开关（ApplyDisplayMode
            // SetGlobalFloat），对增量 HERA 页同样生效——帧率二分实验专用：
            // 切实体后帧率跳升=线框边缘检测的填充开销是主猪。
            // 左摇杆推上再按下=源头时序滤波开关（盯墙养绿 A/B：同墙同段实时切换，
            // 关闭侧=pre-时序滤波基线；左方向同样循环原冻→胶冻→裁冻）。
            if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstick, OVRInput.Controller.LTouch))
            {
                Vector2 lstick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, OVRInput.Controller.LTouch);
                if (lstick.y > 0.5f)
                {
                    scanner.NotifyInput("左摇杆上");
                    scanner.ToggleTemporalFilter();
                }
                else if (lstick.y < -0.5f)
                {
                    scanner.NotifyInput("左摇杆下");
                    scanner.ToggleGeometryTruthView();
                }
                else if (lstick.x > 0.5f)
                {
                    scanner.NotifyInput("左摇杆右");
                    scanner.ToggleManagementBlockWireOverlay();
                }
                else if (lstick.x < -0.5f)
                {
                    scanner.NotifyInput("左摇杆左");
                    scanner.ToggleGunGelGuardedFusionExperiment();
                }
                else
                {
                    float rightGrip = OVRInput.Get(
                        OVRInput.RawAxis1D.RHandTrigger,
                        OVRInput.Controller.RTouch);
                    bool rightGripHeld =
                        OVRInput.Get(OVRInput.RawButton.RHandTrigger,
                                     OVRInput.Controller.RTouch) ||
                        rightGrip > 0.35f;
                    if (rightGripHeld)
                    {
                        scanner.NotifyInput($"右握把+左摇杆 握{rightGrip:0.00}");
                        scanner.ToggleRearWireDepthPrepass();
                    }
                    else
                    {
                        scanner.NotifyInput("左摇杆");
                        scanner.ToggleWireframe();
                    }
                }
                Pulse();
            }
        }

        private void Pulse(float frequency = 0.3f, float amplitude = 0.3f)
        {
            OVRInput.SetControllerVibration(frequency, amplitude, controller);
        }
    }
}
