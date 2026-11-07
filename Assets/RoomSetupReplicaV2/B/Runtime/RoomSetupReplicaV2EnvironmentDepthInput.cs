using System.Collections;
using UnityEngine;

/// <summary>Minimal controls for the sequential Environment Depth B app.</summary>
[DefaultExecutionOrder(-15)]
public sealed class RoomSetupReplicaV2EnvironmentDepthInput : MonoBehaviour
{
    [SerializeField] private OVRInput.Controller controller =
        OVRInput.Controller.RTouch;
    [SerializeField, Range(0.05f, 0.5f)]
    private float actionDebounceSeconds = 0.25f;

    private float _previousTrigger;
    private float _nextActionTime;
    private Coroutine _hapticsStop;

    private void Update()
    {
        RoomSetupReplicaV2EnvironmentDepthScanner scanner =
            RoomSetupReplicaV2EnvironmentDepthScanner.Instance;
        if (scanner == null)
            return;

        float trigger = OVRInput.Get(
            OVRInput.Axis1D.PrimaryIndexTrigger, controller);
        bool triggerDown = trigger > 0.75f &&
                           _previousTrigger <= 0.75f;
        _previousTrigger = trigger;
        if (triggerDown && AcceptAction())
        {
            scanner.ToggleScanning();
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
            scanner.ToggleReplicaPreview();
            Pulse();
        }
        if (OVRInput.GetDown(OVRInput.RawButton.Y) && AcceptAction())
        {
            scanner.ToggleUniqueEdgeDisplay();
            Pulse(0.45f, 0.10f);
        }
        if (OVRInput.GetDown(
                OVRInput.Button.PrimaryThumbstick, controller) &&
            AcceptAction() && scanner.FreezeCurrentReplica())
        {
            Pulse(0.55f, 0.14f);
        }
    }

    private bool AcceptAction()
    {
        if (Time.unscaledTime < _nextActionTime)
            return false;
        _nextActionTime = Time.unscaledTime + actionDebounceSeconds;
        return true;
    }

    private void Pulse(float amplitude = 0.35f,
                       float seconds = 0.08f)
    {
        OVRInput.SetControllerVibration(1f, amplitude,
            OVRInput.Controller.RTouch);
        if (_hapticsStop != null)
            StopCoroutine(_hapticsStop);
        _hapticsStop = StartCoroutine(StopPulse(seconds));
    }

    private IEnumerator StopPulse(float seconds)
    {
        yield return new WaitForSecondsRealtime(seconds);
        OVRInput.SetControllerVibration(
            0f, 0f, OVRInput.Controller.RTouch);
        _hapticsStop = null;
    }
}
