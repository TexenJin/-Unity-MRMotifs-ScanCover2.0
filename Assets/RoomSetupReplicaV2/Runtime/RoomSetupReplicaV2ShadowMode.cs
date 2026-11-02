using System;
using System.Collections.Generic;
using System.IO;
using Unity.Collections;
using UnityEngine;

/// <summary>
/// Read-only Quest shadow observer for the Room Setup replica input path.
///
/// It consumes the exact stereo packet already decoded by the reconstruction,
/// derives stereo/temporal/edge/layer evidence, and writes receipts.  It never
/// changes depth acceptance, TSDF state, mesh candidates, or publication.
/// </summary>
[DisallowMultipleComponent]
public sealed class RoomSetupReplicaV2ShadowMode : MonoBehaviour
{
    private const string ReceiptSchema =
        "roomsetup-replica-v2-quest-shadow-frame-v1";
    private const string SessionSchema =
        "roomsetup-replica-v2-quest-shadow-session-v1";

    [Header("Quest shadow mode (read-only)")]
    [SerializeField] private bool shadowModeEnabled = false;
    [SerializeField, Range(1, 8)] private int pixelStride = 4;
    [SerializeField, Range(1, 120)] private int flushEveryPairs = 10;

    [Header("Explicit experimental hypotheses")]
    [SerializeField] private float edgeSpreadMeters = 0.12f;
    [SerializeField] private float stereoBaseMeters = 0.035f;
    [SerializeField] private float stereoDepthScale = 0.012f;
    [SerializeField] private float temporalBaseMeters = 0.045f;
    [SerializeField] private float temporalDepthScale = 0.012f;
    [SerializeField] private float layerAgreementMeters = 0.075f;
    [SerializeField, Range(2, 8)] private int promotionPairs = 3;

    public bool ShadowModeEnabled => shadowModeEnabled;
    public int ObservedPairCount { get; private set; }
    public int LastSampleCount { get; private set; }
    public int LastRiskySampleCount { get; private set; }
    public int LastChallengeCount { get; private set; }
    public int LastPromotionCount { get; private set; }
    public float LastProcessingMilliseconds { get; private set; }
    public string ReceiptPath => Path.Combine(
        Application.persistentDataPath,
        "RoomSetupReplicaV2", "shadow", "quest-shadow-receipts.jsonl");
    public string CompactStatus =>
        $"shadow p={ObservedPairCount:D5} sample={LastSampleCount:D6} " +
        $"risk={LastRiskySampleCount:D5} challenge={LastChallengeCount:D5} " +
        $"promote={LastPromotionCount:D4} ms={LastProcessingMilliseconds:F2}";

    private HistorySample[] _leftHistory;
    private HistorySample[] _rightHistory;
    private HistorySample[] _leftNext;
    private HistorySample[] _rightNext;
    private ProjectedHistory[] _leftProjected;
    private ProjectedHistory[] _rightProjected;
    private int _width;
    private int _height;
    private int _expected;
    private StreamWriter _writer;
    private bool _writeFailureLogged;
    private string _sessionId;

    private struct HistorySample
    {
        public bool valid;
        public Vector3 dominant;
        public Vector3 challenger;
        public int challengerStreak;
    }

    private struct ProjectedHistory
    {
        public bool valid;
        public float projectiveDepth;
        public HistorySample sample;
    }

    [Serializable]
    private sealed class ShadowThresholds
    {
        public int pixelStride;
        public float edgeSpreadMeters;
        public float stereoBaseMeters;
        public float stereoDepthScale;
        public float temporalBaseMeters;
        public float temporalDepthScale;
        public float layerAgreementMeters;
        public int promotionPairs;
    }

    [Serializable]
    private sealed class ShadowSessionReceipt
    {
        public string schema;
        public string kind;
        public string sessionId;
        public string createdUtc;
        public string mode;
        public string mutationContract;
        public ShadowThresholds thresholds;
    }

    [Serializable]
    private sealed class ShadowFrameReceipt
    {
        public string schema;
        public string kind;
        public string sessionId;
        public string createdUtc;
        public string source;
        public int observedPair;
        public int platformFrame;
        public double captureTime;
        public int width;
        public int height;
        public int sampled;
        public int directValid;
        public int nativeGradientAccepted;
        public int edgeRisk;
        public int stereoAvailable;
        public int stereoAgreement;
        public int stereoConflict;
        public int temporalAvailable;
        public int temporalAgreement;
        public int temporalConflict;
        public int layerAgreement;
        public int layerChallenge;
        public int wouldIntegrate;
        public int wouldDefer;
        public int wouldPromote;
        public int wouldRetireExisting;
        public int missingStereo;
        public int missingTemporal;
        public int missingDirect;
        public float processingMilliseconds;
        public bool affectedReconstruction;
    }

    public void ResetShadowSession()
    {
        CloseWriter();
        _sessionId = null;
        ObservedPairCount = 0;
        LastSampleCount = 0;
        LastRiskySampleCount = 0;
        LastChallengeCount = 0;
        LastPromotionCount = 0;
        LastProcessingMilliseconds = 0f;
        ClearHistory();
    }

    public void FlushReceipts()
    {
        FlushWriter();
    }

    public void ObservePair(
        int platformFrame,
        double captureTime,
        bool replay,
        int width,
        int height,
        NativeArray<Color> leftWorld,
        NativeArray<Color> rightWorld,
        float[] leftDepth,
        float[] rightDepth,
        byte[] leftAccepted,
        byte[] rightAccepted,
        Matrix4x4 leftWorldToClip,
        Matrix4x4 rightWorldToClip)
    {
        if (!shadowModeEnabled || width <= 0 || height <= 0)
            return;
        double processingStart = Time.realtimeSinceStartupAsDouble;
        int expected = width * height;
        if (leftWorld.Length < expected || rightWorld.Length < expected ||
            leftDepth == null || rightDepth == null ||
            leftDepth.Length < expected || rightDepth.Length < expected ||
            leftAccepted == null || rightAccepted == null ||
            leftAccepted.Length < expected || rightAccepted.Length < expected)
            return;

        EnsureHistory(width, height);
        ProjectHistory(_leftHistory, _leftProjected, leftWorldToClip);
        ProjectHistory(_rightHistory, _rightProjected, rightWorldToClip);

        var receipt = new ShadowFrameReceipt
        {
            schema = ReceiptSchema,
            kind = "frame",
            sessionId = EnsureSession(),
            createdUtc = DateTime.UtcNow.ToString("O"),
            source = replay ? "replay" : "live",
            observedPair = ++ObservedPairCount,
            platformFrame = platformFrame,
            captureTime = captureTime,
            width = width,
            height = height,
            affectedReconstruction = false
        };

        Array.Clear(_leftNext, 0, _leftNext.Length);
        Array.Clear(_rightNext, 0, _rightNext.Length);
        ObserveEye(
            leftWorld, rightWorld, leftDepth, rightDepth, leftAccepted,
            _leftProjected, _leftNext, rightWorldToClip, receipt);
        ObserveEye(
            rightWorld, leftWorld, rightDepth, leftDepth, rightAccepted,
            _rightProjected, _rightNext, leftWorldToClip, receipt);
        Swap(ref _leftHistory, ref _leftNext);
        Swap(ref _rightHistory, ref _rightNext);

        LastSampleCount = receipt.sampled;
        LastRiskySampleCount = receipt.edgeRisk + receipt.stereoConflict +
                               receipt.temporalConflict;
        LastChallengeCount = receipt.layerChallenge;
        LastPromotionCount = receipt.wouldPromote;
        receipt.wouldRetireExisting = receipt.wouldPromote;
        receipt.processingMilliseconds = (float)(
            (Time.realtimeSinceStartupAsDouble - processingStart) * 1000d);
        LastProcessingMilliseconds = receipt.processingMilliseconds;
        WriteLine(JsonUtility.ToJson(receipt));
        if (flushEveryPairs > 0 &&
            ObservedPairCount % flushEveryPairs == 0)
            FlushWriter();
    }

    private void ObserveEye(
        NativeArray<Color> world,
        NativeArray<Color> otherWorld,
        float[] depth,
        float[] otherDepth,
        byte[] accepted,
        ProjectedHistory[] projected,
        HistorySample[] next,
        Matrix4x4 otherWorldToClip,
        ShadowFrameReceipt receipt)
    {
        int stride = Mathf.Max(1, pixelStride);
        for (int y = 0; y < _height; y += stride)
        for (int x = 0; x < _width; x += stride)
        {
            int index = x + y * _width;
            receipt.sampled++;
            float currentDepth = depth[index];
            if (!(currentDepth > 0f) || !float.IsFinite(currentDepth))
            {
                receipt.missingDirect++;
                receipt.wouldDefer++;
                continue;
            }
            receipt.directValid++;
            if (accepted[index] != 0)
                receipt.nativeGradientAccepted++;

            Vector3 current = DecodeWorld(world[index]);
            if (!Finite(current))
            {
                receipt.missingDirect++;
                receipt.wouldDefer++;
                continue;
            }

            bool edgeAvailable = TryLocalSpread(
                depth, x, y, out float spread);
            bool edgeRisk = edgeAvailable && spread >= edgeSpreadMeters;
            if (edgeRisk)
                receipt.edgeRisk++;

            bool stereoAvailable = TryProjectedResidual(
                current, otherWorldToClip, otherWorld, otherDepth,
                out float stereoResidual);
            float stereoTolerance = stereoBaseMeters +
                                    stereoDepthScale * currentDepth;
            bool stereoAgreement = stereoAvailable &&
                                   stereoResidual <= stereoTolerance;
            bool stereoConflict = stereoAvailable &&
                                  stereoResidual > stereoTolerance;
            if (stereoAvailable)
            {
                receipt.stereoAvailable++;
                if (stereoAgreement) receipt.stereoAgreement++;
                else receipt.stereoConflict++;
            }
            else
            {
                receipt.missingStereo++;
            }

            ProjectedHistory prior = projected[index];
            bool temporalAvailable = prior.valid;
            float temporalResidual = temporalAvailable
                ? Vector3.Distance(current, prior.sample.dominant)
                : 0f;
            float temporalTolerance = temporalBaseMeters +
                                      temporalDepthScale * currentDepth;
            bool temporalAgreement = temporalAvailable &&
                                     temporalResidual <= temporalTolerance;
            bool temporalConflict = temporalAvailable &&
                                    temporalResidual > temporalTolerance;
            if (temporalAvailable)
            {
                receipt.temporalAvailable++;
                if (temporalAgreement) receipt.temporalAgreement++;
                else receipt.temporalConflict++;
            }
            else
            {
                receipt.missingTemporal++;
            }

            HistorySample state;
            if (!prior.valid)
            {
                state = new HistorySample
                {
                    valid = true,
                    dominant = current
                };
                if (!edgeRisk && !stereoConflict)
                    receipt.wouldIntegrate++;
                else
                    receipt.wouldDefer++;
                next[index] = state;
                continue;
            }

            if (Vector3.Distance(current, prior.sample.dominant) <=
                layerAgreementMeters)
            {
                receipt.layerAgreement++;
                state = prior.sample;
                state.valid = true;
                state.challenger = default;
                state.challengerStreak = 0;
                if (!edgeRisk && !stereoConflict)
                    receipt.wouldIntegrate++;
                else
                    receipt.wouldDefer++;
                next[index] = state;
                continue;
            }

            receipt.layerChallenge++;
            state = prior.sample;
            state.valid = true;
            bool supportedChallenge = stereoAgreement && !edgeRisk;
            if (!supportedChallenge)
            {
                receipt.wouldDefer++;
                next[index] = state;
                continue;
            }

            if (state.challengerStreak > 0 &&
                Vector3.Distance(current, state.challenger) <=
                layerAgreementMeters)
            {
                state.challengerStreak++;
            }
            else
            {
                state.challenger = current;
                state.challengerStreak = 1;
            }

            if (state.challengerStreak >= Mathf.Max(2, promotionPairs))
            {
                state.dominant = state.challenger;
                state.challenger = default;
                state.challengerStreak = 0;
                receipt.wouldPromote++;
            }
            else
            {
                receipt.wouldDefer++;
            }
            next[index] = state;
        }
    }

    private void ProjectHistory(
        HistorySample[] history,
        ProjectedHistory[] projected,
        Matrix4x4 worldToClip)
    {
        Array.Clear(projected, 0, projected.Length);
        if (history == null)
            return;
        int stride = Mathf.Max(1, pixelStride);
        for (int y = 0; y < _height; y += stride)
        for (int x = 0; x < _width; x += stride)
        {
            HistorySample sample = history[x + y * _width];
            if (!sample.valid || !Finite(sample.dominant))
                continue;
            if (!TryProject(
                    sample.dominant, worldToClip,
                    out int px, out int py, out float projectiveDepth))
                continue;
            // Shadow samples live on a coarse lattice. Snap temporal
            // reprojections to that same lattice; otherwise a valid prior
            // sample usually lands between sampled pixels and is falsely
            // reported as missing history.
            px = Mathf.Clamp(
                Mathf.RoundToInt(px / (float)stride) * stride,
                0, ((_width - 1) / stride) * stride);
            py = Mathf.Clamp(
                Mathf.RoundToInt(py / (float)stride) * stride,
                0, ((_height - 1) / stride) * stride);
            int index = px + py * _width;
            if (!projected[index].valid ||
                projectiveDepth < projected[index].projectiveDepth)
            {
                projected[index] = new ProjectedHistory
                {
                    valid = true,
                    projectiveDepth = projectiveDepth,
                    sample = sample
                };
            }
        }
    }

    private bool TryProjectedResidual(
        Vector3 current,
        Matrix4x4 otherWorldToClip,
        NativeArray<Color> otherWorld,
        float[] otherDepth,
        out float residual)
    {
        residual = 0f;
        if (!TryProject(
                current, otherWorldToClip,
                out int px, out int py, out _))
            return false;
        float best = float.PositiveInfinity;
        for (int oy = -1; oy <= 1; oy++)
        for (int ox = -1; ox <= 1; ox++)
        {
            int x = px + ox;
            int y = py + oy;
            if (x < 0 || x >= _width || y < 0 || y >= _height)
                continue;
            int index = x + y * _width;
            if (!(otherDepth[index] > 0f))
                continue;
            Vector3 other = DecodeWorld(otherWorld[index]);
            if (!Finite(other))
                continue;
            best = Mathf.Min(best, Vector3.Distance(current, other));
        }
        if (!float.IsFinite(best))
            return false;
        residual = best;
        return true;
    }

    private bool TryLocalSpread(
        float[] depth, int x, int y, out float spread)
    {
        spread = 0f;
        if (x <= 0 || x >= _width - 1 || y <= 0 || y >= _height - 1)
            return false;
        int index = x + y * _width;
        float center = depth[index];
        float minimum = center;
        float maximum = center;
        int[] neighbours =
        {
            index - 1, index + 1, index - _width, index + _width
        };
        for (int i = 0; i < neighbours.Length; i++)
        {
            float value = depth[neighbours[i]];
            if (!(value > 0f) || !float.IsFinite(value))
                return false;
            minimum = Mathf.Min(minimum, value);
            maximum = Mathf.Max(maximum, value);
        }
        spread = maximum - minimum;
        return true;
    }

    private bool TryProject(
        Vector3 world,
        Matrix4x4 worldToClip,
        out int x,
        out int y,
        out float projectiveDepth)
    {
        x = y = 0;
        projectiveDepth = 0f;
        Vector4 clip = worldToClip * new Vector4(
            world.x, world.y, world.z, 1f);
        if (!float.IsFinite(clip.w) || clip.w <= 0.00001f)
            return false;
        float u = clip.x / clip.w * 0.5f + 0.5f;
        float v = clip.y / clip.w * 0.5f + 0.5f;
        if (!float.IsFinite(u) || !float.IsFinite(v) ||
            u < 0f || u >= 1f || v < 0f || v >= 1f)
            return false;
        x = Mathf.Clamp((int)(u * _width), 0, _width - 1);
        y = Mathf.Clamp((int)(v * _height), 0, _height - 1);
        projectiveDepth = clip.w;
        return true;
    }

    private void EnsureHistory(int width, int height)
    {
        int expected = width * height;
        if (_width == width && _height == height &&
            _leftHistory != null && _leftHistory.Length == expected)
            return;
        _width = width;
        _height = height;
        _expected = expected;
        _leftHistory = new HistorySample[expected];
        _rightHistory = new HistorySample[expected];
        _leftNext = new HistorySample[expected];
        _rightNext = new HistorySample[expected];
        _leftProjected = new ProjectedHistory[expected];
        _rightProjected = new ProjectedHistory[expected];
    }

    private void ClearHistory()
    {
        if (_leftHistory != null) Array.Clear(_leftHistory, 0, _leftHistory.Length);
        if (_rightHistory != null) Array.Clear(_rightHistory, 0, _rightHistory.Length);
        if (_leftNext != null) Array.Clear(_leftNext, 0, _leftNext.Length);
        if (_rightNext != null) Array.Clear(_rightNext, 0, _rightNext.Length);
        if (_leftProjected != null) Array.Clear(_leftProjected, 0, _leftProjected.Length);
        if (_rightProjected != null) Array.Clear(_rightProjected, 0, _rightProjected.Length);
    }

    private string EnsureSession()
    {
        if (!string.IsNullOrEmpty(_sessionId))
            return _sessionId;
        _sessionId = DateTime.UtcNow.ToString("yyyyMMddTHHmmssfffZ") + "-" +
                     Guid.NewGuid().ToString("N").Substring(0, 8);
        var session = new ShadowSessionReceipt
        {
            schema = SessionSchema,
            kind = "session",
            sessionId = _sessionId,
            createdUtc = DateTime.UtcNow.ToString("O"),
            mode = "read-only pre-TSDF stereo temporal layer shadow",
            mutationContract =
                "observes decoded evidence only; never mutates depth, TSDF, mesh, or publication",
            thresholds = new ShadowThresholds
            {
                pixelStride = Mathf.Max(1, pixelStride),
                edgeSpreadMeters = edgeSpreadMeters,
                stereoBaseMeters = stereoBaseMeters,
                stereoDepthScale = stereoDepthScale,
                temporalBaseMeters = temporalBaseMeters,
                temporalDepthScale = temporalDepthScale,
                layerAgreementMeters = layerAgreementMeters,
                promotionPairs = Mathf.Max(2, promotionPairs)
            }
        };
        WriteLine(JsonUtility.ToJson(session));
        return _sessionId;
    }

    private void WriteLine(string json)
    {
        if (string.IsNullOrEmpty(json))
            return;
        try
        {
            if (_writer == null)
            {
                string directory = Path.GetDirectoryName(ReceiptPath);
                if (!string.IsNullOrEmpty(directory))
                    Directory.CreateDirectory(directory);
                _writer = new StreamWriter(ReceiptPath, true);
            }
            _writer.WriteLine(json);
        }
        catch (Exception exception)
        {
            if (!_writeFailureLogged)
            {
                _writeFailureLogged = true;
                Debug.LogWarning(
                    "[RoomSetupReplicaV2Shadow] receipt write failed: " +
                    exception.Message);
            }
            CloseWriter();
        }
    }

    private void FlushWriter()
    {
        try
        {
            _writer?.Flush();
        }
        catch (Exception exception)
        {
            if (!_writeFailureLogged)
            {
                _writeFailureLogged = true;
                Debug.LogWarning(
                    "[RoomSetupReplicaV2Shadow] receipt flush failed: " +
                    exception.Message);
            }
            CloseWriter();
        }
    }

    private void CloseWriter()
    {
        if (_writer == null)
            return;
        try { _writer.Dispose(); }
        catch { }
        _writer = null;
    }

    private static Vector3 DecodeWorld(Color encoded)
    {
        return new Vector3(encoded.r, encoded.g, encoded.b);
    }

    private static void Swap<T>(ref T left, ref T right)
    {
        T value = left;
        left = right;
        right = value;
    }

    private static bool Finite(Vector3 value)
    {
        return float.IsFinite(value.x) &&
               float.IsFinite(value.y) &&
               float.IsFinite(value.z);
    }

    private void OnApplicationPause(bool paused)
    {
        if (paused)
            FlushWriter();
    }

    private void OnApplicationQuit()
    {
        FlushWriter();
        CloseWriter();
    }

    private void OnDestroy()
    {
        FlushWriter();
        CloseWriter();
    }
}
