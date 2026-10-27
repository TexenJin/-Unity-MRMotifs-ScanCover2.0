using UnityEngine;

/// <summary>
/// Room Setup replica input. Right trigger toggles the paired live coverage
/// and recording session; A replays the fixed tape, B clears the replica, and
/// X toggles preview visibility. Replay-only annotation is disabled while live
/// controls are open so the same trigger cannot paint and start capture.
/// </summary>
[DefaultExecutionOrder(-15)]
public sealed class RoomSetupReplicaV2Input : MonoBehaviour
{
    [SerializeField] private OVRInput.Controller controller =
        OVRInput.Controller.RTouch;
    [SerializeField] private bool replayOnlyMode = false;
    [SerializeField, Range(0.05f, 0.5f)]
    private float actionDebounceSeconds = 0.25f;
    private float _previousTrigger;
    private float _nextActionTime;

    private void Awake()
    {
        RoomSetupReplicaV2ReplayProbe probe =
            GetComponent<RoomSetupReplicaV2ReplayProbe>();
        if (replayOnlyMode)
        {
            // Existing serialized V2 scenes predate the probe. Add the
            // read-only annotator without requiring a scene rebuild.
            if (probe == null)
                probe = gameObject.AddComponent<RoomSetupReplicaV2ReplayProbe>();
            probe.enabled = true;
        }
        else if (probe != null)
        {
            probe.enabled = false;
        }
    }

    private void Update()
    {
        RoomSetupReplicaV2Scanner scanner = RoomSetupReplicaV2Scanner.Instance;
        if (scanner == null)
            return;
        if (OVRInput.GetDown(OVRInput.RawButton.A) && AcceptAction())
        {
            scanner.ReplayLastTape();
            Pulse(0.5f, 0.12f);
        }
        if (replayOnlyMode)
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
