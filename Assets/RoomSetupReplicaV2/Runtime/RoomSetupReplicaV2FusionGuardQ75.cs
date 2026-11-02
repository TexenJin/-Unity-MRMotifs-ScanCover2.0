using System;
using System.Globalization;
using Unity.Collections;
using UnityEngine;

/// <summary>
/// Production evaluator for the frozen FusionGuard V0.3 q75 challenger
/// policy. It does not gate ordinary first-surface fusion. It answers only
/// whether one stereo pair is strong enough to challenge an already mature
/// TSDF sign.
/// </summary>
[DisallowMultipleComponent]
public sealed class RoomSetupReplicaV2FusionGuardQ75 : MonoBehaviour
{
    public const float MaximumRisk = 0.6311358213424683f;
    public const float MaximumStereoAnchorSeparationMeters = 0.05f;

    private const string ModelResource =
        "FusionGuard/fusionguard-v03-real-sensor-multiscene-frozen-model";
    private const int ContinuousFeatureCount = 20;
    private const int ExpectedCoefficientCount = 31;

    private static readonly string[] ContinuousFeatureNames =
    {
        "currentDepthMeters",
        "temporal1AbsMeters",
        "temporal2AbsMeters",
        "temporalPersistentMinMeters",
        "temporalMaxMeters",
        "temporalChangeMeters",
        "temporalPositiveMeters",
        "temporalNegativeMeters",
        "temporalDisparity1",
        "temporalDisparity2",
        "pointDistance1Meters",
        "gradientMeters",
        "gradientDepthNormalized",
        "nearJumpMeters",
        "farJumpMeters",
        "edgeSpanMeters",
        "secondOrderMeters",
        "disparityEdge",
        "neighborSupport",
        "pixelMotion1"
    };

    public bool ModelReady => _modelReady;
    public long QualifiedVoxelObservations { get; private set; }
    public long DeferredVoxelObservations { get; private set; }

    private sealed class Snapshot
    {
        public Vector3[] world;
        public byte[] valid;
        public Matrix4x4 worldToClip;
        public Vector3 eye;
        public bool available;
    }

    private readonly Snapshot[,] _history =
        new Snapshot[2, 2]
        {
            { new Snapshot(), new Snapshot() },
            { new Snapshot(), new Snapshot() }
        };

    private float[] _coefficients;
    private float[][] _knots;
    private bool _modelReady;
    private bool _modelFailureLogged;
    private int _width;
    private int _height;
    private int _expected;
    private NativeArray<Color> _leftWorld;
    private NativeArray<Color> _rightWorld;
    private float[] _leftDepth;
    private float[] _rightDepth;
    private byte[] _leftAccepted;
    private byte[] _rightAccepted;
    private Matrix4x4 _leftWorldToClip;
    private Matrix4x4 _rightWorldToClip;
    private Vector3 _leftEye;
    private Vector3 _rightEye;
    private float[] _leftRisk;
    private float[] _rightRisk;
    private byte[] _leftRiskState;
    private byte[] _rightRiskState;
    private bool _pairActive;
    private bool _commitActive;
    private int _commitEye;
    private int _commitCursor;
    // Evaluation is main-thread and non-reentrant. Reuse these arrays so a
    // first scan over many unique pixels does not create per-pixel garbage.
    private readonly float[] _rawFeatures =
        new float[ContinuousFeatureCount];
    private readonly float[] _percentiles =
        new float[ContinuousFeatureCount];

    public void ResetSession()
    {
        _pairActive = false;
        _commitActive = false;
        _commitEye = 0;
        _commitCursor = 0;
        QualifiedVoxelObservations = 0;
        DeferredVoxelObservations = 0;
        for (int eye = 0; eye < 2; eye++)
        for (int age = 0; age < 2; age++)
        {
            Snapshot snapshot = _history[eye, age];
            snapshot.available = false;
            if (snapshot.valid != null)
                Array.Clear(snapshot.valid, 0, snapshot.valid.Length);
        }
    }

    public void BeginPair(
        int width,
        int height,
        NativeArray<Color> leftWorld,
        NativeArray<Color> rightWorld,
        float[] leftDepth,
        float[] rightDepth,
        byte[] leftAccepted,
        byte[] rightAccepted,
        Matrix4x4 leftWorldToClip,
        Matrix4x4 rightWorldToClip,
        Vector3 leftEye,
        Vector3 rightEye)
    {
        // A new pair is never allowed to overwrite a partially committed
        // history snapshot. The reconstruction owner keeps the input pair
        // single-flight and finishes CommitPairSlice before FinalizePair.
        _commitActive = false;
        _commitEye = 0;
        _commitCursor = 0;
        EnsureModel();
        _pairActive = _modelReady && width > 0 && height > 0;
        if (!_pairActive)
            return;
        _width = width;
        _height = height;
        _expected = width * height;
        _leftWorld = leftWorld;
        _rightWorld = rightWorld;
        _leftDepth = leftDepth;
        _rightDepth = rightDepth;
        _leftAccepted = leftAccepted;
        _rightAccepted = rightAccepted;
        _leftWorldToClip = leftWorldToClip;
        _rightWorldToClip = rightWorldToClip;
        _leftEye = leftEye;
        _rightEye = rightEye;
        EnsureArray(ref _leftRisk, _expected);
        EnsureArray(ref _rightRisk, _expected);
        EnsureArray(ref _leftRiskState, _expected);
        EnsureArray(ref _rightRiskState, _expected);
        Array.Clear(_leftRiskState, 0, _expected);
        Array.Clear(_rightRiskState, 0, _expected);
        EnsureHistoryArrays();
    }

    /// <summary>
    /// Requires both eyes to observe the queried voxel with the same direct
    /// sign, a surface-anchor separation no larger than 5 cm, and frozen V0.3
    /// risk no larger than the q75 operating threshold in both eyes.
    /// </summary>
    public bool TryQualifyVoxelObservation(
        int sourceEye,
        int sourcePixel,
        Vector3 voxelWorld,
        out Vector3 stereoAnchor,
        out sbyte directSign)
    {
        stereoAnchor = default;
        directSign = 0;
        if (!_pairActive || sourceEye < 0 || sourceEye > 1 ||
            sourcePixel < 0 || sourcePixel >= _expected ||
            !Finite(voxelWorld))
        {
            DeferredVoxelObservations++;
            return false;
        }

        int otherEye = 1 - sourceEye;
        if (!TryProjectPixel(
                voxelWorld, WorldToClip(otherEye), out int otherPixel,
                out float otherVoxelDepth))
        {
            DeferredVoxelObservations++;
            return false;
        }
        if (!TryProjectPixel(
                voxelWorld, WorldToClip(sourceEye), out _,
                out float sourceVoxelDepth))
        {
            DeferredVoxelObservations++;
            return false;
        }
        // The source pixel is the exact floor-projected index already used by
        // IntegrateChunk; do not re-quantize it a second time at pixel edges.
        if (Accepted(sourceEye)[sourcePixel] == 0 ||
            Accepted(otherEye)[otherPixel] == 0)
        {
            DeferredVoxelObservations++;
            return false;
        }

        float sourceResidual = Depth(sourceEye)[sourcePixel] -
                               sourceVoxelDepth;
        float otherResidual = Depth(otherEye)[otherPixel] -
                              otherVoxelDepth;
        if (!float.IsFinite(sourceResidual) ||
            !float.IsFinite(otherResidual) ||
            Mathf.Abs(sourceResidual) < 0.025f ||
            Mathf.Abs(otherResidual) < 0.025f)
        {
            DeferredVoxelObservations++;
            return false;
        }
        sbyte sourceSign = sourceResidual > 0f ? (sbyte)1 : (sbyte)-1;
        sbyte otherSign = otherResidual > 0f ? (sbyte)1 : (sbyte)-1;
        if (sourceSign != otherSign)
        {
            DeferredVoxelObservations++;
            return false;
        }

        Vector3 sourceAnchor = World(sourceEye, sourcePixel);
        Vector3 otherAnchor = World(otherEye, otherPixel);
        if (!Finite(sourceAnchor) || !Finite(otherAnchor) ||
            Vector3.Distance(sourceAnchor, otherAnchor) >
                MaximumStereoAnchorSeparationMeters ||
            !TryRisk(sourceEye, sourcePixel, out float sourceRisk) ||
            !TryRisk(otherEye, otherPixel, out float otherRisk) ||
            Mathf.Max(sourceRisk, otherRisk) > MaximumRisk)
        {
            DeferredVoxelObservations++;
            return false;
        }

        stereoAnchor = (sourceAnchor + otherAnchor) * 0.5f;
        directSign = sourceSign;
        QualifiedVoxelObservations++;
        return true;
    }

    /// <summary>
    /// Returns the current stereo surface anchor without asking the q75 model
    /// to classify it. This is used only after a mature voxel has already
    /// detected a large same-sign TSDF jump: both eyes must still agree that
    /// the challenger is one physical depth layer before its short temporal
    /// latch may advance.
    /// </summary>
    public bool TryGetStereoSurfaceAnchor(
        int sourceEye,
        int sourcePixel,
        out Vector3 stereoAnchor)
    {
        stereoAnchor = default;
        if (!_pairActive || sourceEye < 0 || sourceEye > 1 ||
            sourcePixel < 0 || sourcePixel >= _expected ||
            Accepted(sourceEye)[sourcePixel] == 0)
            return false;

        Vector3 sourceAnchor = World(sourceEye, sourcePixel);
        if (!Finite(sourceAnchor))
            return false;
        int otherEye = 1 - sourceEye;
        if (!TryProjectPixel(
                sourceAnchor, WorldToClip(otherEye), out int otherPixel,
                out _) ||
            Accepted(otherEye)[otherPixel] == 0)
            return false;
        Vector3 otherAnchor = World(otherEye, otherPixel);
        if (!Finite(otherAnchor) ||
            Vector3.Distance(sourceAnchor, otherAnchor) >
                MaximumStereoAnchorSeparationMeters)
            return false;
        stereoAnchor = (sourceAnchor + otherAnchor) * 0.5f;
        return true;
    }

    public void CommitPair()
    {
        if (!_pairActive)
            return;
        BeginCommitPair();
        while (!CommitPairSlice(double.PositiveInfinity))
        {
        }
    }

    public void BeginCommitPair()
    {
        if (!_pairActive || _commitActive)
            return;
        _commitActive = true;
        _commitEye = 0;
        _commitCursor = 0;
        RotateCommitEye(0);
    }

    /// <summary>
    /// Copies the accepted stereo history without creating one unbounded
    /// 2 x width x height main-thread spike. Returns true once both eyes are
    /// committed. The resulting snapshots are byte-for-byte equivalent to
    /// the previous eager CommitPair implementation.
    /// </summary>
    public bool CommitPairSlice(double deadline)
    {
        if (!_pairActive)
            return true;
        if (!_commitActive)
            BeginCommitPair();

        while (_commitEye < 2)
        {
            Snapshot target = _history[_commitEye, 0];
            NativeArray<Color> source = _commitEye == 0
                ? _leftWorld
                : _rightWorld;
            float[] sourceDepth = Depth(_commitEye);
            while (_commitCursor < _expected)
            {
                int index = _commitCursor++;
                Vector3 point = DecodeWorld(source[index]);
                bool valid = sourceDepth[index] > 0f && Finite(point);
                target.valid[index] = valid ? (byte)1 : (byte)0;
                target.world[index] = valid ? point : default;
                if ((_commitCursor & 255) == 0 &&
                    Time.realtimeSinceStartupAsDouble >= deadline)
                    return false;
            }
            target.worldToClip = WorldToClip(_commitEye);
            target.eye = Eye(_commitEye);
            target.available = true;
            _commitEye++;
            _commitCursor = 0;
            if (_commitEye < 2)
                RotateCommitEye(_commitEye);
        }

        _commitActive = false;
        _pairActive = false;
        return true;
    }

    private void RotateCommitEye(int eye)
    {
        Snapshot recycled = _history[eye, 1];
        _history[eye, 1] = _history[eye, 0];
        _history[eye, 0] = recycled;
        EnsureSnapshot(recycled);
    }

    private bool TryRisk(int eye, int index, out float risk)
    {
        float[] cache = eye == 0 ? _leftRisk : _rightRisk;
        byte[] state = eye == 0 ? _leftRiskState : _rightRiskState;
        if (state[index] != 0)
        {
            risk = cache[index];
            return state[index] == 1;
        }
        bool valid = EvaluateRisk(eye, index, out risk);
        cache[index] = risk;
        state[index] = valid ? (byte)1 : (byte)2;
        return valid;
    }

    private bool EvaluateRisk(int eye, int index, out float risk)
    {
        risk = float.NaN;
        if (Accepted(eye)[index] == 0)
            return false;
        Vector3 current = World(eye, index);
        if (!Finite(current))
            return false;
        Vector3 eyePosition = Eye(eye);
        float currentRange = Vector3.Distance(current, eyePosition);
        if (!(currentRange > 0f) || !float.IsFinite(currentRange))
            return false;

        int x = index % _width;
        int y = index / _width;
        float[] raw = _rawFeatures;
        raw[0] = currentRange;

        bool temporal1 = TryTemporal(
            current, x, y, _history[eye, 0], out float signed1,
            out float point1, out float observed1, out float expected1,
            out float pixelMotion1);
        bool temporal2 = TryTemporal(
            current, x, y, _history[eye, 1], out float signed2,
            out _, out float observed2, out float expected2, out _);

        raw[1] = temporal1 ? Mathf.Abs(signed1) : float.NaN;
        raw[2] = temporal2 ? Mathf.Abs(signed2) : float.NaN;
        bool bothTemporal = temporal1 && temporal2;
        raw[3] = bothTemporal
            ? Mathf.Min(Mathf.Abs(signed1), Mathf.Abs(signed2))
            : float.NaN;
        raw[4] = bothTemporal
            ? Mathf.Max(Mathf.Abs(signed1), Mathf.Abs(signed2))
            : float.NaN;
        raw[5] = bothTemporal ? Mathf.Abs(signed1 - signed2) : float.NaN;
        raw[6] = temporal1 ? Mathf.Max(signed1, 0f) : float.NaN;
        raw[7] = temporal1 ? Mathf.Max(-signed1, 0f) : float.NaN;
        raw[8] = temporal1
            ? Mathf.Abs(SafeReciprocal(observed1) -
                        SafeReciprocal(expected1))
            : float.NaN;
        raw[9] = temporal2
            ? Mathf.Abs(SafeReciprocal(observed2) -
                        SafeReciprocal(expected2))
            : float.NaN;
        raw[10] = temporal1 ? point1 : float.NaN;

        EdgeFeatures edges = CalculateEdges(eye, x, y, currentRange);
        raw[11] = edges.gradient;
        raw[12] = float.IsFinite(edges.gradient)
            ? edges.gradient / Mathf.Max(currentRange * currentRange, 0.01f)
            : float.NaN;
        raw[13] = edges.nearJump;
        raw[14] = edges.farJump;
        raw[15] = edges.span;
        raw[16] = edges.secondOrder;
        raw[17] = edges.disparityEdge;
        raw[18] = edges.support;
        raw[19] = temporal1 ? pixelMotion1 : float.NaN;

        float[] percentile = _percentiles;
        for (int feature = 0; feature < ContinuousFeatureCount; feature++)
            percentile[feature] = Percentile(raw[feature], _knots[feature]);

        float value = _coefficients[0];
        for (int feature = 0; feature < ContinuousFeatureCount; feature++)
            value += _coefficients[1 + feature] * percentile[feature];
        bool signFlip = bothTemporal &&
                        ((signed1 < 0f) != (signed2 < 0f));
        value += _coefficients[21] * (signFlip ? 1f : 0f);
        value += _coefficients[22] * (temporal1 ? 0f : 1f);
        value += _coefficients[23] * (temporal2 ? 0f : 1f);
        value += _coefficients[24] *
                 (float.IsFinite(edges.gradient) ? 0f : 1f);
        value += _coefficients[25] * percentile[3] * percentile[17];
        value += _coefficients[26] * (signFlip ? percentile[4] : 0f);
        value += _coefficients[27] * percentile[6] * percentile[13];
        value += _coefficients[28] * percentile[7] * percentile[14];
        value += _coefficients[29] * percentile[16] * percentile[1];
        value += _coefficients[30] * percentile[0] * percentile[12];
        if (!float.IsFinite(value))
            return false;
        risk = value;
        return true;
    }

    private struct EdgeFeatures
    {
        public float gradient;
        public float nearJump;
        public float farJump;
        public float span;
        public float secondOrder;
        public float disparityEdge;
        public float support;
    }

    private EdgeFeatures CalculateEdges(
        int eye, int x, int y, float center)
    {
        var result = new EdgeFeatures
        {
            gradient = float.NaN,
            span = float.NaN,
            secondOrder = float.NaN
        };
        float minimum = float.PositiveInfinity;
        float maximum = float.NegativeInfinity;
        float sum = 0f;
        int count = 0;
        float left = RangeAt(eye, x - 1, y);
        float right = RangeAt(eye, x + 1, y);
        float up = RangeAt(eye, x, y - 1);
        float down = RangeAt(eye, x, y + 1);
        float[] neighbours = { left, right, up, down };
        for (int i = 0; i < neighbours.Length; i++)
        {
            float neighbour = neighbours[i];
            if (!(neighbour > 0f) || !float.IsFinite(neighbour))
                continue;
            float delta = neighbour - center;
            result.nearJump = Mathf.Max(result.nearJump, Mathf.Max(delta, 0f));
            result.farJump = Mathf.Max(result.farJump, Mathf.Max(-delta, 0f));
            result.disparityEdge = Mathf.Max(
                result.disparityEdge,
                Mathf.Abs(SafeReciprocal(neighbour) -
                          SafeReciprocal(center)));
            minimum = Mathf.Min(minimum, neighbour);
            maximum = Mathf.Max(maximum, neighbour);
            sum += neighbour;
            count++;
        }
        result.support = count / 4f;
        if (count > 0)
        {
            result.span = maximum - minimum;
            result.secondOrder = Mathf.Abs(center - sum / count);
        }
        // Frozen V0.3 used backward horizontal/vertical differences.
        if (left > 0f && up > 0f)
        {
            float dx = Mathf.Abs(center - left);
            float dy = Mathf.Abs(center - up);
            result.gradient = Mathf.Sqrt(dx * dx + dy * dy);
        }
        return result;
    }

    private bool TryTemporal(
        Vector3 current,
        int sourceX,
        int sourceY,
        Snapshot snapshot,
        out float signed,
        out float pointDistance,
        out float observedRange,
        out float expectedRange,
        out float pixelMotion)
    {
        signed = pointDistance = observedRange = expectedRange =
            pixelMotion = float.NaN;
        if (snapshot == null || !snapshot.available ||
            !TryProjectPixel(
                current, snapshot.worldToClip, out int targetIndex,
                out _))
            return false;
        if (snapshot.valid[targetIndex] == 0)
            return false;
        Vector3 observed = snapshot.world[targetIndex];
        if (!Finite(observed))
            return false;
        observedRange = Vector3.Distance(observed, snapshot.eye);
        expectedRange = Vector3.Distance(current, snapshot.eye);
        signed = observedRange - expectedRange;
        pointDistance = Vector3.Distance(observed, current);
        int targetX = targetIndex % _width;
        int targetY = targetIndex / _width;
        float dx = targetX - sourceX;
        float dy = targetY - sourceY;
        pixelMotion = Mathf.Sqrt(dx * dx + dy * dy);
        return float.IsFinite(signed) && float.IsFinite(pointDistance);
    }

    private float RangeAt(int eye, int x, int y)
    {
        if (x < 0 || x >= _width || y < 0 || y >= _height)
            return float.NaN;
        int index = x + y * _width;
        if (!(Depth(eye)[index] > 0f))
            return float.NaN;
        Vector3 point = World(eye, index);
        return Finite(point) ? Vector3.Distance(point, Eye(eye)) : float.NaN;
    }

    private bool TryProjectPixel(
        Vector3 world,
        Matrix4x4 worldToClip,
        out int index,
        out float projectiveDepth)
    {
        index = -1;
        projectiveDepth = 0f;
        Vector4 clip = worldToClip * new Vector4(
            world.x, world.y, world.z, 1f);
        if (!float.IsFinite(clip.w) || clip.w <= 0.000001f)
            return false;
        float u = clip.x / clip.w * 0.5f + 0.5f;
        float v = clip.y / clip.w * 0.5f + 0.5f;
        if (!float.IsFinite(u) || !float.IsFinite(v) ||
            u < 0f || u >= 1f || v < 0f || v >= 1f)
            return false;
        int x = Mathf.Clamp(Mathf.FloorToInt(u * _width), 0, _width - 1);
        int y = Mathf.Clamp(Mathf.FloorToInt(v * _height), 0, _height - 1);
        index = x + y * _width;
        projectiveDepth = clip.w;
        return true;
    }

    private void EnsureModel()
    {
        if (_modelReady || _modelFailureLogged)
            return;
        TextAsset asset = Resources.Load<TextAsset>(ModelResource);
        if (asset == null)
        {
            LogModelFailure("frozen model resource is missing");
            return;
        }
        try
        {
            _coefficients = ParseArray(asset.text, "coefficients", 0);
            _knots = new float[ContinuousFeatureCount][];
            int calibrationStart = asset.text.IndexOf(
                "\"empiricalPercentileKnots\"", StringComparison.Ordinal);
            if (calibrationStart < 0)
                throw new FormatException("missing empiricalPercentileKnots");
            for (int i = 0; i < ContinuousFeatureCount; i++)
                _knots[i] = ParseArray(
                    asset.text, ContinuousFeatureNames[i], calibrationStart);
            if (_coefficients.Length != ExpectedCoefficientCount)
                throw new FormatException(
                    $"coefficient count {_coefficients.Length}");
            for (int i = 0; i < _knots.Length; i++)
            {
                if (_knots[i] == null || _knots[i].Length < 2)
                    throw new FormatException(
                        $"invalid knots for {ContinuousFeatureNames[i]}");
            }
            _modelReady = true;
        }
        catch (Exception exception)
        {
            LogModelFailure(exception.Message);
        }
    }

    private void LogModelFailure(string reason)
    {
        _modelFailureLogged = true;
        Debug.LogError(
            "[RoomSetupReplicaV2] q75 production route disabled: " + reason);
    }

    private static float[] ParseArray(string json, string key, int start)
    {
        int keyIndex = json.IndexOf(
            "\"" + key + "\"", start, StringComparison.Ordinal);
        if (keyIndex < 0)
            throw new FormatException("missing " + key);
        int open = json.IndexOf('[', keyIndex);
        int close = json.IndexOf(']', open + 1);
        if (open < 0 || close < 0)
            throw new FormatException("malformed " + key);
        string[] tokens = json.Substring(open + 1, close - open - 1)
            .Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
        var values = new float[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
            values[i] = float.Parse(
                tokens[i], NumberStyles.Float, CultureInfo.InvariantCulture);
        return values;
    }

    private static float Percentile(float value, float[] knots)
    {
        if (!float.IsFinite(value) || knots == null || knots.Length < 2)
            return 0f;
        if (value < knots[0])
            return 0f;
        int last = knots.Length - 1;
        if (value > knots[last])
            return 1f;
        int low = 0;
        int high = knots.Length;
        while (low < high)
        {
            int mid = low + ((high - low) >> 1);
            if (knots[mid] < value)
                low = mid + 1;
            else
                high = mid;
        }
        int right = Mathf.Clamp(low, 0, last);
        if (knots[right] == value)
            return right / (float)last;
        int left = right - 1;
        while (left > 0 && knots[left - 1] == knots[left])
            left--;
        while (right < last && knots[right] == knots[left])
            right++;
        float span = knots[right] - knots[left];
        if (!(span > 0f))
            return left / (float)last;
        float t = Mathf.Clamp01((value - knots[left]) / span);
        return Mathf.Lerp(left / (float)last, right / (float)last, t);
    }

    private void EnsureHistoryArrays()
    {
        for (int eye = 0; eye < 2; eye++)
        for (int age = 0; age < 2; age++)
            EnsureSnapshot(_history[eye, age]);
    }

    private void EnsureSnapshot(Snapshot snapshot)
    {
        if (snapshot.world == null || snapshot.world.Length != _expected)
        {
            snapshot.world = new Vector3[_expected];
            snapshot.valid = new byte[_expected];
            snapshot.available = false;
        }
    }

    private NativeArray<Color> WorldArray(int eye) =>
        eye == 0 ? _leftWorld : _rightWorld;
    private float[] Depth(int eye) => eye == 0 ? _leftDepth : _rightDepth;
    private byte[] Accepted(int eye) =>
        eye == 0 ? _leftAccepted : _rightAccepted;
    private Matrix4x4 WorldToClip(int eye) =>
        eye == 0 ? _leftWorldToClip : _rightWorldToClip;
    private Vector3 Eye(int eye) => eye == 0 ? _leftEye : _rightEye;
    private Vector3 World(int eye, int index) =>
        DecodeWorld(WorldArray(eye)[index]);

    private static Vector3 DecodeWorld(Color encoded) =>
        new Vector3(encoded.r, encoded.g, encoded.b);

    private static float SafeReciprocal(float value) =>
        value > 0f && float.IsFinite(value) ? 1f / value : float.NaN;

    private static bool Finite(Vector3 value) =>
        float.IsFinite(value.x) && float.IsFinite(value.y) &&
        float.IsFinite(value.z);

    private static void EnsureArray(ref float[] array, int length)
    {
        if (array == null || array.Length != length)
            array = new float[length];
    }

    private static void EnsureArray(ref byte[] array, int length)
    {
        if (array == null || array.Length != length)
            array = new byte[length];
    }
}
