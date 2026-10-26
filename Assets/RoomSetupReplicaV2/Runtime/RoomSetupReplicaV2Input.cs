using UnityEngine;

/// <summary>
/// Replay-lab input. A replays the fixed tape. The separate read-only replay
/// probe owns right-trigger paint/erase and B-stroke undo after replay; live
/// capture controls remain disabled by replayOnlyMode.
/// </summary>
public sealed class RoomSetupReplicaV2Input : MonoBehaviour
{
    [SerializeField] private OVRInput.Controller controller =
        OVRInput.Controller.RTouch;
    [SerializeField] private bool replayOnlyMode = true;
    [SerializeField, Range(0.05f, 0.5f)]
    private float actionDebounceSeconds = 0.25f;
    private float _previousTrigger;
    private float _nextActionTime;

    private void Awake()
    {
        // Existing serialized V2 scenes predate the replay probe. Add the
        // read-only annotator at runtime so rebuilding the scene is optional.
        if (GetComponent<RoomSetupReplicaV2ReplayProbe>() == null)
            gameObject.AddComponent<RoomSetupReplicaV2ReplayProbe>();
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
