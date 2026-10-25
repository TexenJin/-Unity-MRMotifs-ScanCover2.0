using UnityEngine;

/// <summary>
/// Replay-only V1 comparator: A replays V2's last Stage-space tape; B clears;
/// X toggles the comparator mesh. It cannot overwrite the shared tape.
/// </summary>
public sealed class RoomSetupReplicaV1ReplayInput : MonoBehaviour
{
    [SerializeField] private OVRInput.Controller controller =
        OVRInput.Controller.RTouch;
    [SerializeField, Range(0.05f, 0.5f)]
    private float actionDebounceSeconds = 0.25f;
    private float _nextActionTime;

    private void Update()
    {
        RoomSetupReplicaV1ReplayScanner scanner = RoomSetupReplicaV1ReplayScanner.Instance;
        if (scanner == null)
            return;
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
            RoomSetupReplicaV1ReplayReconstruction reconstruction =
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
