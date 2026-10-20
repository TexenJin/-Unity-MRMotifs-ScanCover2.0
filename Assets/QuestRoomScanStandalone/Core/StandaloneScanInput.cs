using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Minimal operator controls. Display state has exactly two modes:
    /// original production mesh, or that same mesh plus immutable same-candidate
    /// extraction provenance. No legacy display carousel remains reachable.
    /// </summary>
    public class StandaloneScanInput : MonoBehaviour
    {
        [SerializeField, Tooltip("接收输入的手柄")]
        private OVRInput.Controller controller = OVRInput.Controller.RTouch;

        private float _previousTrigger;

        private void Update()
        {
            StandaloneRoomScanner scanner = StandaloneRoomScanner.Instance;
            if (scanner == null) return;

            float trigger = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger,
                controller);
            bool triggerDown = trigger > 0.75f && _previousTrigger <= 0.75f;
            _previousTrigger = trigger;

            if (triggerDown)
            {
                scanner.NotifyInput("扳机");
                if (!scanner.IsScanning)
                    _ = scanner.StartScanningAsync();
                Pulse();
            }

            if (OVRInput.GetDown(OVRInput.Button.One, controller))
            {
                scanner.NotifyInput("A键");
                scanner.ToggleOperatorFreeze();
                Pulse();
            }

            if (OVRInput.GetDown(OVRInput.Button.Two, controller))
            {
                scanner.NotifyInput("B键");
                if (scanner.IsScanning || scanner.HasStarted)
                    scanner.StopAndClearScan();
                Pulse(0.5f, 0.5f);
            }

            if (OVRInput.GetDown(OVRInput.RawButton.X))
            {
                scanner.NotifyInput("X键");
                scanner.ToggleMeshLayerEvidenceView();
                Pulse();
            }

        }

        private void Pulse(float amplitude = 0.3f, float duration = 0.08f)
        {
            OVRInput.SetControllerVibration(1f, amplitude, controller);
            Invoke(nameof(StopPulse), duration);
        }

        private void StopPulse()
        {
            OVRInput.SetControllerVibration(0f, 0f, controller);
        }
    }
}
