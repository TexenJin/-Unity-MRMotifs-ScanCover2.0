using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.XR;

public class ScanCoverCollisionVerifier : MonoBehaviour
{
    public enum ControllerButton
    {
        None,
        PrimaryButton,
        SecondaryButton,
        Trigger,
        Grip,
        ThumbstickClick,
        Menu
    }

    [Header("Refs")]
    public Transform spawnFrom;                 // CenterEye / controller
    public LayerMask collisionMask = ~0;        // Everything
    public bool drawRay = true;

    [Header("Input (New Input System - XR Controller)")]
    public bool preferRightHandController = true;
    public ControllerButton dropButton = ControllerButton.PrimaryButton;
    public ControllerButton rayHoldButton = ControllerButton.Trigger;

    [Header("Drop Test (Rigidbody)")]
    public float dropForwardMeters = 0.6f;
    public float dropUpMeters = 0.8f;
    public float ballRadius = 0.04f;            // 4cm
    public float ballMass = 0.2f;
    public float initialDownVelocity = 0.0f;

    [Header("Raycast Test (Query)")]
    public float rayLength = 8.0f;

    [Header("Logs")]
    public bool logRayHits = true;
    public bool logCollisions = true;

    void Reset()
    {
        if (!spawnFrom && Camera.main) spawnFrom = Camera.main.transform;
    }

    void Update()
    {
        if (!spawnFrom) return;

        if (GetActionDown(dropButton))
            SpawnDropBall();

        bool doRay = (rayHoldButton == ControllerButton.None) || GetActionHeld(rayHoldButton);
        if (doRay)
            DoRaycast();
    }

    private bool GetActionDown(ControllerButton button)
    {
        if (button == ControllerButton.None) return false;

        var xr = GetPreferredXRController();
        var xrControl = ResolveButton(xr, button);
        if (xrControl != null) return xrControl.wasPressedThisFrame;

        return GetGamepadButtonDown(button);
    }

    private bool GetActionHeld(ControllerButton button)
    {
        if (button == ControllerButton.None) return false;

        var xr = GetPreferredXRController();
        var xrControl = ResolveButton(xr, button);
        if (xrControl != null) return xrControl.isPressed;

        return GetGamepadButtonHeld(button);
    }

    private XRController GetPreferredXRController()
    {
        XRController fallback = null;

        foreach (var device in InputSystem.devices)
        {
            if (device is not XRController xr || !device.enabled) continue;

            if (fallback == null) fallback = xr;

            bool isRight = HasUsage(device, "RightHand");
            bool isLeft = HasUsage(device, "LeftHand");
            if (preferRightHandController && isRight) return xr;
            if (!preferRightHandController && isLeft) return xr;
        }

        return fallback;
    }

    private static bool HasUsage(InputDevice device, string usageName)
    {
        for (int i = 0; i < device.usages.Count; i++)
        {
            if (device.usages[i].ToString() == usageName) return true;
        }
        return false;
    }

    private static ButtonControl ResolveButton(XRController controller, ControllerButton button)
    {
        if (controller == null) return null;

        string controlName = button switch
        {
            ControllerButton.PrimaryButton => "primaryButton",
            ControllerButton.SecondaryButton => "secondaryButton",
            ControllerButton.Trigger => "triggerPressed",
            ControllerButton.Grip => "gripPressed",
            ControllerButton.ThumbstickClick => "primary2DAxisClick",
            ControllerButton.Menu => "menuButton",
            _ => null
        };

        if (string.IsNullOrEmpty(controlName)) return null;
        return controller.TryGetChildControl<ButtonControl>(controlName);
    }

    private static bool GetGamepadButtonDown(ControllerButton button)
    {
        var pad = Gamepad.current;
        if (pad == null) return false;

        return button switch
        {
            ControllerButton.PrimaryButton => pad.buttonSouth.wasPressedThisFrame,
            ControllerButton.SecondaryButton => pad.buttonEast.wasPressedThisFrame,
            ControllerButton.Trigger => pad.rightTrigger.wasPressedThisFrame,
            ControllerButton.Grip => pad.leftTrigger.wasPressedThisFrame,
            ControllerButton.ThumbstickClick => pad.rightStickButton.wasPressedThisFrame,
            ControllerButton.Menu => pad.startButton.wasPressedThisFrame,
            _ => false
        };
    }

    private static bool GetGamepadButtonHeld(ControllerButton button)
    {
        var pad = Gamepad.current;
        if (pad == null) return false;

        return button switch
        {
            ControllerButton.PrimaryButton => pad.buttonSouth.isPressed,
            ControllerButton.SecondaryButton => pad.buttonEast.isPressed,
            ControllerButton.Trigger => pad.rightTrigger.isPressed,
            ControllerButton.Grip => pad.leftTrigger.isPressed,
            ControllerButton.ThumbstickClick => pad.rightStickButton.isPressed,
            ControllerButton.Menu => pad.startButton.isPressed,
            _ => false
        };
    }

    void SpawnDropBall()
    {
        // Spawn test sphere
        var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        go.name = $"[CollisionTestBall]_{Time.frameCount}";
        go.transform.localScale = Vector3.one * (ballRadius * 2f);

        // Forward offset + up offset
        Vector3 p = spawnFrom.position + spawnFrom.forward * dropForwardMeters + Vector3.up * dropUpMeters;
        go.transform.position = p;

        // Collider
        var sc = go.GetComponent<SphereCollider>();
        sc.isTrigger = false;

        var rb = go.AddComponent<Rigidbody>();
        rb.mass = ballMass;
        rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
        rb.interpolation = RigidbodyInterpolation.Interpolate;
        if (initialDownVelocity != 0f)
            rb.linearVelocity = Vector3.down * Mathf.Abs(initialDownVelocity);

        // Collision callback
        var reporter = go.AddComponent<CollisionReporter>();
        reporter.owner = this;

        if (logCollisions)
            Debug.Log($"[CollisionVerifier] Spawn ball at {p:F3}");
    }

    void DoRaycast()
    {
        Ray r = new Ray(spawnFrom.position, spawnFrom.forward);
        if (Physics.Raycast(r, out RaycastHit hit, rayLength, collisionMask, QueryTriggerInteraction.Ignore))
        {
            if (drawRay) Debug.DrawLine(r.origin, hit.point, Color.green, 0f, false);

            if (logRayHits)
            {
                Debug.Log($"[RayHit] {hit.collider.name} dist={hit.distance:F3} " +
                          $"pt={hit.point:F3} n={hit.normal:F3}");
            }
        }
        else
        {
            if (drawRay) Debug.DrawLine(r.origin, r.origin + r.direction * rayLength, Color.red, 0f, false);
        }
    }

    // Collision callback on spawned sphere
    public class CollisionReporter : MonoBehaviour
    {
        public ScanCoverCollisionVerifier owner;

        void OnCollisionEnter(Collision c)
        {
            if (owner != null && owner.logCollisions)
            {
                var cp = c.GetContact(0);
                Debug.Log($"[BallCollision] hit={c.collider.name} pt={cp.point:F3} n={cp.normal:F3} " +
                          $"relV={c.relativeVelocity.magnitude:F3}");
            }
        }
    }
}
