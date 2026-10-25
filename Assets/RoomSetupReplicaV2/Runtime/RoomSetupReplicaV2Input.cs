using UnityEngine;

/// <summary>
/// Right trigger starts/pauses live capture; A records an exact input tape;
/// Y replays it; B clears; X toggles the replica mesh.
/// </summary>
public sealed class RoomSetupReplicaV2Input : MonoBehaviour
{
    [SerializeField] private OVRInput.Controller controller =
        OVRInput.Controller.RTouch;
    private float _previousTrigger;

    private void Update()
    {
        RoomSetupReplicaV2Scanner scanner = RoomSetupReplicaV2Scanner.Instance;
        if (scanner == null)
            return;
        float trigger = OVRInput.Get(
            OVRInput.Axis1D.PrimaryIndexTrigger, controller);
        bool triggerDown = trigger > 0.75f && _previousTrigger <= 0.75f;
        _previousTrigger = trigger;
        if (triggerDown)
        {
            scanner.ToggleScanning();
            Pulse();
        }
        if (OVRInput.GetDown(OVRInput.Button.Two, controller))
        {
            scanner.ClearReplica();
            Pulse(0.5f, 0.12f);
        }
        if (OVRInput.GetDown(OVRInput.RawButton.A))
        {
            scanner.ToggleRecording();
            Pulse(0.5f, 0.12f);
        }
        if (OVRInput.GetDown(OVRInput.RawButton.Y))
        {
            scanner.ReplayLastTape();
            Pulse(0.5f, 0.12f);
        }
        if (OVRInput.GetDown(OVRInput.RawButton.X))
        {
            RoomSetupReplicaV2Reconstruction reconstruction =
                scanner.Reconstruction;
            if (reconstruction != null)
                reconstruction.SetPreviewVisible(
                    !reconstruction.PreviewVisible);
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
