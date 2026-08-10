using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// QRS 独立测试链的右手柄输入：
    ///   扳机     = 开始扫描 / 暂停后继续
    ///   A        = 暂停（数据保留）
    ///   B        = 保存累计账并清空（下一次扳机 = 全新扫描、全新账）
    ///   摇杆按下 = 线框 / 顶点色实体 切换
    ///   左手 X   = 生产提取 / 严格观测边影子提取切换
    ///   左手 Y   = 原生产网格 A / 候选生产网格 B 切换
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
                if (!scanner.IsScanning)
                {
                    _ = scanner.StartScanningAsync();
                    Pulse();
                }
            }

            // A：暂停
            if (OVRInput.GetDown(OVRInput.Button.One, controller))
            {
                scanner.NotifyInput("A键");
                if (scanner.IsScanning)
                {
                    scanner.PauseScanning();
                    Pulse();
                }
            }

            // B：先导出累计时间序列，再清空网格/体积/融合计数/记账会话
            if (OVRInput.GetDown(OVRInput.Button.Two, controller))
            {
                scanner.NotifyInput("B键");
                if (scanner.IsScanning || scanner.HasStarted)
                {
                    scanner.StopAndClearScan();
                    Pulse(0.5f, 0.5f);
                }
            }

            // 右摇杆按下：线框 / 实体 切换（真数字键，GetDown 可靠；
            // 对应 QRS 原版 X 键 CycleRenderMode 的二态精简版）
            if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstick, controller))
            {
                scanner.ToggleWireframe();
                Pulse();
            }

            // 左手 Y 不再切换前台网格。严格候选链已经正式成为并锁定为生产网格。
        }

        private void Pulse(float frequency = 0.3f, float amplitude = 0.3f)
        {
            OVRInput.SetControllerVibration(frequency, amplitude, controller);
        }
    }
}
