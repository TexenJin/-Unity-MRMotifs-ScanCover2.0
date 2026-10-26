using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Read-only replay annotation probe. After a replay has produced its frozen
/// final mesh, the right controller ray selects visible defect triangles.
/// Labels are bound to the exported mesh SHA-256 and Quest Stage coordinates.
/// This component never changes fusion, meshing, cleanup, or publication.
/// </summary>
[DefaultExecutionOrder(20)]
[DisallowMultipleComponent]
[RequireComponent(typeof(RoomSetupReplicaV2Reconstruction))]
public sealed class RoomSetupReplicaV2ReplayProbe : MonoBehaviour
{
    private enum PaintMode : byte
    {
        None,
        Add,
        Erase
    }

    [Serializable]
    private sealed class ProbeLabel
    {
        public int ordinal;
        public int stroke;
        public string classification;
        public string utc;
        public float realtimeSeconds;
        public int triangle;
        public float distanceMeters;
        public Vector3 barycentric;
        public Vector3 hitStage;
        public Vector3 normalStage;
        public Vector3 triangleAStage;
        public Vector3 triangleBStage;
        public Vector3 triangleCStage;
        public Vector3 rayOriginStage;
        public Vector3 rayDirectionStage;
        public Vector3 controllerStagePosition;
        public Quaternion controllerStageRotation;
        public Vector3 headStagePosition;
        public Quaternion headStageRotation;
        public int replayedPairs;
    }

    [Serializable]
    private sealed class ProbeReceipt
    {
        public int schemaVersion = 2;
        public string kind = "RoomSetupReplicaV2HumanTriangleLabels";
        public string coordinateSpace = "QuestStage";
        public string meshPath;
        public string meshSha256;
        public int meshTriangles;
        public string tapePath;
        public string createdUtc;
        public string updatedUtc;
        public string controls =
            "HoldRightTrigger=PaintOrEraseFromStartFace;" +
            "B=UndoLastStroke;A=Replay";
        public List<ProbeLabel> labels = new List<ProbeLabel>();
    }

    private sealed class StrokeDelta
    {
        public int stroke;
        public PaintMode mode;
        public readonly List<ProbeLabel> added = new List<ProbeLabel>();
        public readonly List<ProbeLabel> removed = new List<ProbeLabel>();
        public int Changes => added.Count + removed.Count;
    }

    [Header("Controller annotation")]
    [SerializeField] private OVRInput.Controller controller =
        OVRInput.Controller.RTouch;
    [SerializeField, Range(1f, 12f)] private float rayLengthMeters = 8f;
    [SerializeField, Range(0.01f, 0.2f)]
    private float raycastIntervalSeconds = 0.033f;
    [SerializeField, Range(0.5f, 0.95f)]
    private float triggerPressThreshold = 0.75f;
    [SerializeField] private Color idleRayColor =
        new Color(1f, 0.72f, 0.08f, 0.9f);
    [SerializeField] private Color hitRayColor =
        new Color(0.05f, 0.95f, 1f, 0.95f);
    [SerializeField] private Color markedTriangleColor =
        new Color(1f, 0.08f, 0.55f, 0.92f);

    public string ReceiptPath => Path.Combine(
        Application.persistentDataPath,
        "RoomSetupReplicaV2",
        "last-replay-human-labels.json");
    public int LabelCount => _receipt != null ? _receipt.labels.Count : 0;

    private readonly HashSet<int> _labelledTriangles = new HashSet<int>();
    private readonly HashSet<int> _strokeTriangles = new HashSet<int>();
    private readonly Stack<StrokeDelta> _strokeHistory =
        new Stack<StrokeDelta>();
    private readonly List<Vector3> _highlightVertices =
        new List<Vector3>(192);
    private readonly List<int> _highlightTriangles = new List<int>(192);

    private RoomSetupReplicaV2Reconstruction _reconstruction;
    private OVRCameraRig _rig;
    private Transform _controllerAnchor;
    private Transform _headAnchor;
    private LineRenderer _ray;
    private GameObject _reticle;
    private GameObject _highlightObject;
    private Mesh _highlightMesh;
    private Material _rayMaterial;
    private Material _highlightMaterial;
    private ProbeReceipt _receipt;
    private string _activeMeshSha = string.Empty;
    private float _previousTrigger;
    private float _nextRaycastTime;
    private float _nextAutosaveTime;
    private bool _hasHit;
    private bool _readyLogged;
    private bool _receiptDirty;
    private PaintMode _paintMode;
    private int _nextStroke = 1;
    private int _nextOrdinal = 1;
    private StrokeDelta _activeStroke;
    private RoomSetupReplicaV2Reconstruction.ReplayProbeHit _hit;

    private void Awake()
    {
        _reconstruction = GetComponent<RoomSetupReplicaV2Reconstruction>();
        EnsureVisuals();
    }

    private void Update()
    {
        ResolveAnchors();
        float trigger = OVRInput.Get(
            OVRInput.Axis1D.PrimaryIndexTrigger, controller);
        bool triggerPressed = trigger >= triggerPressThreshold;
        bool triggerDown = triggerPressed &&
                           _previousTrigger < triggerPressThreshold;
        bool triggerUp = !triggerPressed &&
                         _previousTrigger >= triggerPressThreshold;
        _previousTrigger = trigger;

        if (_reconstruction == null ||
            !_reconstruction.CanProbeFinalMesh ||
            _controllerAnchor == null)
        {
            if (_paintMode != PaintMode.None)
                EndStroke();
            SetVisualsActive(false);
            _hasHit = false;
            _readyLogged = false;
            return;
        }

        BeginMeshSessionIfNeeded();
        SetVisualsActive(true);
        if (!_readyLogged)
        {
            _readyLogged = true;
            Debug.Log(
                "[RoomSetupReplicaV2Probe] ready: hold right trigger to " +
                "paint from an unselected face or erase from a selected " +
                "face; B undoes the latest stroke. receipt=" + ReceiptPath);
        }

        if (Time.unscaledTime >= _nextRaycastTime)
        {
            _nextRaycastTime = Time.unscaledTime +
                               Mathf.Max(0.01f, raycastIntervalSeconds);
            var ray = new Ray(
                _controllerAnchor.position,
                _controllerAnchor.forward);
            _hasHit = _reconstruction.TryRaycastFinalMesh(
                ray, rayLengthMeters, out _hit);
            UpdateRay(ray);
        }

        if (triggerPressed && _paintMode == PaintMode.None && _hasHit)
            BeginStroke(_hit);
        else if (triggerPressed && _paintMode != PaintMode.None && _hasHit)
            ApplyStroke(_hit);
        else if (triggerDown && !_hasHit)
            Pulse(0.18f, 0.06f);

        if (triggerUp && _paintMode != PaintMode.None)
            EndStroke();
        if (_receiptDirty && Time.unscaledTime >= _nextAutosaveTime)
        {
            SaveReceipt();
            _nextAutosaveTime = Time.unscaledTime + 0.25f;
        }
        if (_paintMode == PaintMode.None &&
            OVRInput.GetDown(OVRInput.RawButton.B, controller))
            UndoLastStroke();
    }

    private void BeginMeshSessionIfNeeded()
    {
        string sha = _reconstruction.LastReplayMeshSha256 ?? string.Empty;
        if (_receipt != null && string.Equals(
                _activeMeshSha, sha, StringComparison.Ordinal))
            return;
        _activeMeshSha = sha;
        _labelledTriangles.Clear();
        _strokeTriangles.Clear();
        _strokeHistory.Clear();
        _paintMode = PaintMode.None;
        _activeStroke = null;
        _nextStroke = 1;
        _nextOrdinal = 1;
        _receipt = LoadMatchingReceipt(sha);
        if (_receipt == null)
        {
            _receipt = new ProbeReceipt
            {
                meshPath = _reconstruction.LastReplayMeshPath,
                meshSha256 = sha,
                meshTriangles = _reconstruction.FinalReplayTriangleCount,
                tapePath = _reconstruction.TapePath,
                createdUtc = DateTime.UtcNow.ToString("O"),
                updatedUtc = DateTime.UtcNow.ToString("O")
            };
        }
        for (int i = 0; i < _receipt.labels.Count; i++)
        {
            ProbeLabel label = _receipt.labels[i];
            _labelledTriangles.Add(label.triangle);
            _nextStroke = Mathf.Max(_nextStroke, label.stroke + 1);
            _nextOrdinal = Mathf.Max(_nextOrdinal, label.ordinal + 1);
        }
        RebuildHighlight();
        SaveReceipt();
    }

    private ProbeReceipt LoadMatchingReceipt(string meshSha)
    {
        try
        {
            if (!File.Exists(ReceiptPath))
                return null;
            ProbeReceipt receipt = JsonUtility.FromJson<ProbeReceipt>(
                File.ReadAllText(ReceiptPath));
            if (receipt == null || receipt.labels == null ||
                !string.Equals(
                    receipt.meshSha256, meshSha,
                    StringComparison.OrdinalIgnoreCase) ||
                receipt.meshTriangles !=
                _reconstruction.FinalReplayTriangleCount)
                return null;
            Debug.Log(
                $"[RoomSetupReplicaV2Probe] resumed " +
                $"labels={receipt.labels.Count} receipt={ReceiptPath}");
            return receipt;
        }
        catch (Exception exception)
        {
            Debug.LogWarning(
                "[RoomSetupReplicaV2Probe] could not resume receipt: " +
                exception.Message);
            return null;
        }
    }

    private void BeginStroke(
        RoomSetupReplicaV2Reconstruction.ReplayProbeHit hit)
    {
        if (_receipt == null || _paintMode != PaintMode.None)
            return;
        _paintMode = _labelledTriangles.Contains(hit.triangle)
            ? PaintMode.Erase
            : PaintMode.Add;
        _strokeTriangles.Clear();
        _activeStroke = new StrokeDelta
        {
            stroke = _nextStroke++,
            mode = _paintMode
        };
        ApplyStroke(hit);
        Pulse(_paintMode == PaintMode.Add ? 0.35f : 0.20f, 0.05f);
    }

    private void ApplyStroke(
        RoomSetupReplicaV2Reconstruction.ReplayProbeHit hit)
    {
        if (_receipt == null || _activeStroke == null ||
            !_strokeTriangles.Add(hit.triangle))
            return;
        bool changed = _paintMode == PaintMode.Add
            ? AddLabel(hit, _activeStroke.stroke)
            : RemoveLabel(hit.triangle);
        if (!changed)
            return;
        _receipt.updatedUtc = DateTime.UtcNow.ToString("O");
        _receiptDirty = true;
        RebuildHighlight();
    }

    private bool AddLabel(
        RoomSetupReplicaV2Reconstruction.ReplayProbeHit hit,
        int stroke)
    {
        if (_receipt == null || !_labelledTriangles.Add(hit.triangle))
            return false;

        var label = new ProbeLabel
        {
            ordinal = _nextOrdinal++,
            stroke = stroke,
            classification = "VisibleDefect",
            utc = DateTime.UtcNow.ToString("O"),
            realtimeSeconds = Time.realtimeSinceStartup,
            triangle = hit.triangle,
            distanceMeters = hit.distanceMeters,
            barycentric = hit.barycentric,
            hitStage = hit.stagePoint,
            normalStage = hit.stageNormal,
            triangleAStage = hit.stageA,
            triangleBStage = hit.stageB,
            triangleCStage = hit.stageC,
            rayOriginStage = _reconstruction.ReplayWorldToStagePoint(
                _controllerAnchor.position),
            rayDirectionStage =
                _reconstruction.ReplayWorldToStageDirection(
                    _controllerAnchor.forward),
            controllerStagePosition =
                _reconstruction.ReplayWorldToStagePoint(
                    _controllerAnchor.position),
            controllerStageRotation = StageRotation(_controllerAnchor),
            headStagePosition = _headAnchor != null
                ? _reconstruction.ReplayWorldToStagePoint(
                    _headAnchor.position)
                : Vector3.zero,
            headStageRotation = _headAnchor != null
                ? StageRotation(_headAnchor)
                : Quaternion.identity,
            replayedPairs = _reconstruction.LastReplayPairCount
        };
        _receipt.labels.Add(label);
        _activeStroke.added.Add(label);
        return true;
    }

    private bool RemoveLabel(int triangle)
    {
        if (_receipt == null || !_labelledTriangles.Contains(triangle))
            return false;
        for (int i = _receipt.labels.Count - 1; i >= 0; i--)
        {
            if (_receipt.labels[i].triangle != triangle)
                continue;
            ProbeLabel removed = _receipt.labels[i];
            _receipt.labels.RemoveAt(i);
            _labelledTriangles.Remove(triangle);
            _activeStroke.removed.Add(removed);
            return true;
        }
        return false;
    }

    private void EndStroke()
    {
        if (_activeStroke != null && _activeStroke.Changes > 0)
        {
            _strokeHistory.Push(_activeStroke);
            SaveReceipt();
            Pulse(_activeStroke.mode == PaintMode.Add ? 0.65f : 0.4f, 0.08f);
            Debug.Log(
                $"[RoomSetupReplicaV2Probe] stroke=" +
                $"{_activeStroke.stroke} mode={_activeStroke.mode} " +
                $"changes={_activeStroke.Changes} " +
                $"selected={_receipt.labels.Count} receipt={ReceiptPath}");
        }
        _activeStroke = null;
        _paintMode = PaintMode.None;
        _strokeTriangles.Clear();
    }

    private void UndoLastStroke()
    {
        if (_receipt == null || _strokeHistory.Count == 0)
            return;
        StrokeDelta stroke = _strokeHistory.Pop();
        for (int i = 0; i < stroke.added.Count; i++)
        {
            int triangle = stroke.added[i].triangle;
            for (int j = _receipt.labels.Count - 1; j >= 0; j--)
                if (_receipt.labels[j].triangle == triangle)
                    _receipt.labels.RemoveAt(j);
            _labelledTriangles.Remove(triangle);
        }
        for (int i = 0; i < stroke.removed.Count; i++)
        {
            ProbeLabel label = stroke.removed[i];
            if (_labelledTriangles.Add(label.triangle))
                _receipt.labels.Add(label);
        }
        _receipt.updatedUtc = DateTime.UtcNow.ToString("O");
        _receiptDirty = true;
        RebuildHighlight();
        SaveReceipt();
        Pulse(0.35f, 0.08f);
        Debug.Log(
            $"[RoomSetupReplicaV2Probe] undid stroke={stroke.stroke} " +
            $"mode={stroke.mode} selected={_receipt.labels.Count}");
    }

    private void SaveReceipt()
    {
        if (_receipt == null)
            return;
        try
        {
            string directory = Path.GetDirectoryName(ReceiptPath);
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);
            File.WriteAllText(
                ReceiptPath, JsonUtility.ToJson(_receipt, true));
            _receiptDirty = false;
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "[RoomSetupReplicaV2Probe] cannot write receipt: " +
                exception);
        }
    }

    private void ResolveAnchors()
    {
        if (_rig == null)
            _rig = FindAnyObjectByType<OVRCameraRig>();
        if (_rig == null)
            return;
        _controllerAnchor = _rig.rightControllerAnchor;
        _headAnchor = _rig.centerEyeAnchor;
    }

    private Quaternion StageRotation(Transform source)
    {
        Vector3 forward = _reconstruction.ReplayWorldToStageDirection(
            source.forward);
        Vector3 up = _reconstruction.ReplayWorldToStageDirection(source.up);
        return Vector3.Cross(forward, up).sqrMagnitude > 0.000001f
            ? Quaternion.LookRotation(forward, up)
            : Quaternion.identity;
    }

    private void UpdateRay(Ray source)
    {
        if (_ray == null)
            return;
        Vector3 end = _hasHit
            ? _hit.worldPoint
            : source.GetPoint(rayLengthMeters);
        _ray.SetPosition(0, source.origin);
        _ray.SetPosition(1, end);
        SetMaterialColor(
            _rayMaterial, _hasHit ? hitRayColor : idleRayColor);
        if (_reticle != null)
        {
            _reticle.SetActive(_hasHit);
            if (_hasHit)
            {
                _reticle.transform.position =
                    _hit.worldPoint + _hit.worldNormal * 0.004f;
                _reticle.transform.rotation = Quaternion.LookRotation(
                    _hit.worldNormal,
                    Mathf.Abs(Vector3.Dot(_hit.worldNormal, Vector3.up)) >
                    0.95f ? Vector3.forward : Vector3.up);
            }
        }
    }

    private void EnsureVisuals()
    {
        Shader shader = Resources.Load<Shader>(
            "RoomSetupReplicaV2Preview");
        if (shader == null)
            shader = Shader.Find("Genesis/RoomSetupReplicaV2Preview");
        if (shader == null)
        {
            Debug.LogError(
                "[RoomSetupReplicaV2Probe] preview shader unavailable");
            return;
        }
        _rayMaterial = new Material(shader)
        {
            name = "Room Setup Replica V2 Probe Ray"
        };
        _highlightMaterial = new Material(shader)
        {
            name = "Room Setup Replica V2 Probe Highlight"
        };
        SetMaterialColor(_rayMaterial, idleRayColor);
        SetMaterialColor(_highlightMaterial, markedTriangleColor);

        _ray = gameObject.AddComponent<LineRenderer>();
        _ray.name = "Room Setup Replica V2 Probe Ray";
        _ray.useWorldSpace = true;
        _ray.positionCount = 2;
        _ray.widthMultiplier = 0.006f;
        _ray.numCapVertices = 2;
        _ray.shadowCastingMode = ShadowCastingMode.Off;
        _ray.receiveShadows = false;
        _ray.sharedMaterial = _rayMaterial;

        _reticle = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        _reticle.name = "Room Setup Replica V2 Probe Reticle";
        _reticle.transform.localScale = Vector3.one * 0.025f;
        Collider collider = _reticle.GetComponent<Collider>();
        if (collider != null)
            Destroy(collider);
        MeshRenderer reticleRenderer =
            _reticle.GetComponent<MeshRenderer>();
        if (reticleRenderer != null)
        {
            reticleRenderer.shadowCastingMode = ShadowCastingMode.Off;
            reticleRenderer.receiveShadows = false;
            reticleRenderer.sharedMaterial = _highlightMaterial;
        }

        _highlightObject = new GameObject(
            "Room Setup Replica V2 Human Labels");
        MeshFilter filter = _highlightObject.AddComponent<MeshFilter>();
        MeshRenderer renderer =
            _highlightObject.AddComponent<MeshRenderer>();
        renderer.shadowCastingMode = ShadowCastingMode.Off;
        renderer.receiveShadows = false;
        renderer.sharedMaterial = _highlightMaterial;
        _highlightMesh = new Mesh
        {
            name = "Room Setup Replica V2 Human Label Mesh",
            indexFormat = IndexFormat.UInt32
        };
        _highlightMesh.MarkDynamic();
        filter.sharedMesh = _highlightMesh;
        SetVisualsActive(false);
    }

    private void RebuildHighlight()
    {
        if (_highlightMesh == null)
            return;
        Transform stage = _reconstruction != null
            ? _reconstruction.ReplayStageTransform
            : null;
        if (_highlightObject != null && stage != null &&
            _highlightObject.transform.parent != stage)
        {
            _highlightObject.transform.SetParent(stage, false);
            _highlightObject.transform.localPosition = Vector3.zero;
            _highlightObject.transform.localRotation = Quaternion.identity;
            _highlightObject.transform.localScale = Vector3.one;
        }
        _highlightVertices.Clear();
        _highlightTriangles.Clear();
        if (_receipt != null)
        {
            for (int i = 0; i < _receipt.labels.Count; i++)
            {
                ProbeLabel label = _receipt.labels[i];
                int first = _highlightVertices.Count;
                Vector3 offset = label.normalStage * 0.004f;
                _highlightVertices.Add(label.triangleAStage + offset);
                _highlightVertices.Add(label.triangleBStage + offset);
                _highlightVertices.Add(label.triangleCStage + offset);
                _highlightTriangles.Add(first);
                _highlightTriangles.Add(first + 1);
                _highlightTriangles.Add(first + 2);
            }
        }
        _highlightMesh.Clear(false);
        _highlightMesh.SetVertices(_highlightVertices);
        _highlightMesh.SetTriangles(_highlightTriangles, 0, false);
        _highlightMesh.RecalculateBounds();
        _highlightMesh.UploadMeshData(false);
    }

    private void SetVisualsActive(bool active)
    {
        if (_ray != null)
            _ray.enabled = active;
        if (_reticle != null)
            _reticle.SetActive(active && _hasHit);
        if (_highlightObject != null)
            _highlightObject.SetActive(active);
    }

    private static void SetMaterialColor(Material material, Color color)
    {
        if (material != null && material.HasProperty("_BaseColor"))
            material.SetColor("_BaseColor", color);
    }

    private void Pulse(float amplitude, float duration)
    {
        OVRInput.SetControllerVibration(1f, amplitude, controller);
        CancelInvoke(nameof(StopPulse));
        Invoke(nameof(StopPulse), duration);
    }

    private void StopPulse()
    {
        OVRInput.SetControllerVibration(0f, 0f, controller);
    }

    private void OnDestroy()
    {
        if (_paintMode != PaintMode.None)
            EndStroke();
        else if (_receiptDirty)
            SaveReceipt();
        StopPulse();
        if (_reticle != null) Destroy(_reticle);
        if (_highlightObject != null) Destroy(_highlightObject);
        if (_highlightMesh != null) Destroy(_highlightMesh);
        if (_rayMaterial != null) Destroy(_rayMaterial);
        if (_highlightMaterial != null) Destroy(_highlightMaterial);
    }
}
