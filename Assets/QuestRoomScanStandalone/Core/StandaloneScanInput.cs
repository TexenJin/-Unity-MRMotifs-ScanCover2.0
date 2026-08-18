using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// QRS 独立测试链的右手柄输入：
    ///   扳机       = 开始 / 继续采集共享 TSDF
    ///   A          = 冻结共享 TSDF（先隐藏采集覆盖片）
    ///   Y          = 在 64³ / 32³ / 16³ 只读回放档之间循环
    ///   B          = 只导出并清空当前档，不清共享 TSDF
    ///   右摇杆按下 = 冻结回放后：单色线框/好坏空状态着色；扫描中：网格显示开/关（只藏绘制，
    ///                融合/提取/精修后台照跑——满屏网视角的帧率二分闸，判光栅化压力用。
    ///                注意 A/B 实验旗下满屏网=增量 HERA 所画，开关切的就是它）
    ///   右摇杆方向+按下 = 性能二分热键：上=实时轨 / 左=冻结调度器 / 下=融合 20↔10Hz / 右=深度预处理(双边+缘洗)
    ///   左摇杆按下 = 线框 / 实体切换（全局 shader 开关，帧率二分用）
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

            // A：冻结同一份 TSDF；片状采集标记必须先隐藏。
            if (OVRInput.GetDown(OVRInput.Button.One, controller))
            {
                scanner.NotifyInput("A键");
                if (scanner.IsChunkAbExperimentEnabled)
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

            // B：A/B 模式只导出并清当前档；绝不清共享 TSDF。
            if (OVRInput.GetDown(OVRInput.Button.Two, controller))
            {
                scanner.NotifyInput("B键");
                if (scanner.IsChunkAbExperimentEnabled)
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
            //   左=冻结调度器开关（2s 全体积普查+票箱回读）
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
                    scanner.ToggleFrozenBlockSupervisor();
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
                    // 着色切换只在冻结回放后有意义（ToggleChunkAbDisplayMode 未冻结
                    // 直接早退=空转）；扫描中这颗键让给帧率二分总闸"显开/显关"。
                    if (scanner.IsChunkAbFrozen)
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

            // 左手 X：呼出/收起采集点阵（单帧真值判官对照层）。
            if (OVRInput.GetDown(OVRInput.RawButton.X))
            {
                scanner.NotifyInput("X键");
                scanner.ToggleCoverageMarkers();
                Pulse();
            }

            // 左摇杆按下：线框/实体切换。走全局 shader 开关（ApplyDisplayMode
            // SetGlobalFloat），对增量 HERA 页同样生效——帧率二分实验专用：
            // 切实体后帧率跳升=线框边缘检测的填充开销是主猪。
            if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstick, OVRInput.Controller.LTouch))
            {
                scanner.NotifyInput("左摇杆");
                scanner.ToggleWireframe();
                Pulse();
            }
        }

        private void Pulse(float frequency = 0.3f, float amplitude = 0.3f)
        {
            OVRInput.SetControllerVibration(frequency, amplitude, controller);
        }
    }
}
