using UnityEngine;

/// <summary>
/// Right trigger starts/stops coverage and recording together; A replays the
/// last tape; B clears; X toggles the replica mesh.
/// </summary>
public sealed class RoomSetupReplicaV2Input : MonoBehaviour
{
    [SerializeField] private OVRInput.Controller controller =
        OVRInput.Controller.RTouch;
    [SerializeField, Range(0.05f, 0.5f)]
    private float actionDebounceSeconds = 0.25f;
    private float _previousTrigger;
    private float _nextActionTime;

    private void Update()
    {
        RoomSetupReplicaV2Scanner scanner = RoomSetupReplicaV2Scanner.Instance;
        if (scanner == null)
            return;
        float trigger = OVRInput.Get(
            OVRInput.Axis1D.PrimaryIndexTrigger, controller);
        bool triggerDown = trigger > 0.75f && _previousTrigger <= 0.75f;
        _previousTrigger = trigger;
        if (triggerDown && AcceptAction())
        {
            scanner.ToggleCoverageAndRecording();
            Pulse();
        }
        if (OVRInput.GetDown(OVRInput.Button.Two, controller) &&
            AcceptAction())
        {
            scanner.ClearReplica();
            Pulse(0.5f, 0.12f);
        }
        if (OVRInput.GetDown(OVRInput.RawButton.A) && AcceptAction())
        {
            scanner.ReplayLastTape();
            Pulse(0.5f, 0.12f);
        }
        if (OVRInput.GetDown(OVRInput.RawButton.X) && AcceptAction())
        {
            RoomSetupReplicaV2Reconstruction reconstruction =
                scanner.Reconstruction;
            if (reconstruction != null)
                reconstruction.SetPreviewVisible(
                    !reconstruction.PreviewVisible);
            Pulse();
        }
    }

    private bool AcceptAction()
    {
        if (Time.unscaledTime < _nextActionTime)
            return false;
        _nextActionTime = Time.unscaledTime +
                          Mathf.Max(0.05f, actionDebounceSeconds);
        return true;
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
