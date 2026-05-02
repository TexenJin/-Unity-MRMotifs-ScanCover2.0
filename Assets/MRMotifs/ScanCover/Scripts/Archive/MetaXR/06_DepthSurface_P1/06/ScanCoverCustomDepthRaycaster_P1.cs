using System;
using System.Reflection;
using Meta.XR.EnvironmentDepth;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
[DefaultExecutionOrder(-48)]
public class ScanCoverCustomDepthRaycaster_P1 : MonoBehaviour
{
    public enum DepthEyeMode
    {
        Left = 0,
        Right = 1,
        Both = 2,
    }

    private static readonly int EnvironmentDepthTextureId = Shader.PropertyToID("_EnvironmentDepthTexture");
    private static readonly int EnvironmentDepthTextureSizeId = Shader.PropertyToID("_EnvironmentDepthTextureSize");
    private static readonly int CopiedDepthTextureId = Shader.PropertyToID("_CopiedDepthTexture");
    private static readonly int EnvironmentDepthZBufferParamsId = Shader.PropertyToID("_EnvironmentDepthZBufferParams");
    private const int NumEyes = 2;
    public const int TextureSize = 128;

    [Header("Refs")]
    public EnvironmentDepthManager depthManager;

    [Header("Depth Copy")]
    public bool warmUpRaycast = true;
    public DepthEyeMode eye = DepthEyeMode.Right;

    [Header("Debug")]
    public bool debugLog = false;

    public bool IsDepthTextureAvailable => _isDepthTextureAvailable;

    private ComputeShader _copyShader;
    private ComputeBuffer _computeBuffer;
    private NativeArray<float> _depthTexturePixels;
    private NativeArray<float> _gpuRequestBuffer;
    private AsyncGPUReadbackRequest? _currentGpuReadbackRequest;
    private bool _isDepthTextureAvailable;
    private readonly Matrix4x4[] _matrixVP = new Matrix4x4[NumEyes];
    private readonly Matrix4x4[] _matrixV = new Matrix4x4[NumEyes];
    private readonly Matrix4x4[] _matrixVPInv = new Matrix4x4[NumEyes];
    private readonly Plane[][] _camFrustumPlanes = { new Plane[6], new Plane[6] };
    private Vector4 _environmentDepthZBufferParams;
    private int _currentEyeIndex;
    private Matrix4x4 _worldToTrackingSpaceMatrix = Matrix4x4.identity;

    private FieldInfo _fiFrameDescriptors;
    private MethodInfo _miGetTrackingSpaceWorldToLocalMatrix;
    private Array _cachedFrameDescArray;

    private void Awake()
    {
        ResolveRefs();
        BindReflection();
        InitializeResources();
        SetEye((int)eye);
    }

    private void OnEnable()
    {
        ResolveRefs();
        BindReflection();
        InitializeResources();
    }

    private void Update()
    {
        ResolveRefs();
        BindReflection();
        UpdateTextureCopyRequest();
        CreateTextureCopyRequestIfNeeded();
    }

    private void OnDisable()
    {
        InvalidateDepthTexture();
    }

    private void OnDestroy()
    {
        if (_currentGpuReadbackRequest.HasValue && !_currentGpuReadbackRequest.Value.done)
            _currentGpuReadbackRequest.Value.WaitForCompletion();

        if (_computeBuffer != null)
            _computeBuffer.Dispose();
        if (_depthTexturePixels.IsCreated)
            _depthTexturePixels.Dispose();
        if (_gpuRequestBuffer.IsCreated)
            _gpuRequestBuffer.Dispose();
    }

    public void SetEye(int eyeIndex)
    {
        if (eyeIndex <= 0)
        {
            eye = DepthEyeMode.Left;
            _currentEyeIndex = 0;
            return;
        }

        if (eyeIndex == 1)
        {
            eye = DepthEyeMode.Right;
            _currentEyeIndex = 1;
            return;
        }

        eye = DepthEyeMode.Both;
        _currentEyeIndex = 0;
    }

    public Vector2Int WorldPosToNonNormalizedTextureCoords02(Vector3 worldPos)
    {
        var clipPos = _matrixVP[_currentEyeIndex] * new Vector4(worldPos.x, worldPos.y, worldPos.z, 1f);
        var uv = (new Vector2(clipPos.x, clipPos.y) / clipPos.w + Vector2.one) * 0.5f;
        return new Vector2Int(
            Mathf.Clamp((int)(uv.x * TextureSize), 0, TextureSize - 1),
            Mathf.Clamp((int)(uv.y * TextureSize), 0, TextureSize - 1));
    }

    public float SampleDepthTexture02(Vector2Int texCoord)
    {
        if (!_depthTexturePixels.IsCreated)
            return 0f;

        return _depthTexturePixels[
            texCoord.x + texCoord.y * TextureSize + TextureSize * TextureSize * _currentEyeIndex];
    }

    public Vector3 WorldPosAtDepthTexCoord02(Vector2Int texCoord)
    {
        float linearDepth = SampleDepthTexture02(texCoord);
        if (linearDepth <= 0f)
            return new Vector3(float.NaN, float.NaN, float.NaN);

        float clipDepth = _environmentDepthZBufferParams.x / linearDepth - _environmentDepthZBufferParams.y;
        const float oneOverSize = 1f / TextureSize;
        var clipPos = new Vector4(
            texCoord.x * oneOverSize * 2f - 1f,
            texCoord.y * oneOverSize * 2f - 1f,
            clipDepth,
            1f);

        Vector4 worldH = _matrixVPInv[_currentEyeIndex] * clipPos;
        return worldH.w != 0f ? (Vector3)(worldH / worldH.w) : new Vector3(float.NaN, float.NaN, float.NaN);
    }

    public float WorldPosToLinearDepth02(Vector3 worldPos)
    {
        var viewPos = _matrixV[_currentEyeIndex] * new Vector4(worldPos.x, worldPos.y, worldPos.z, 1f);
        return -viewPos.z;
    }

    public Vector3 ReconstructNormal02(Vector2Int texCoord)
    {
        float centerDepth = SampleDepthTexture02(texCoord);
        if (centerDepth <= 0f)
            return Vector3.zero;

        var centerWorld = WorldPosAtDepthTexCoord02(texCoord);
        if (!IsFinite(centerWorld))
            return Vector3.zero;

        var horDeriv = ClosestDerivativeToAdjacentExtrapolations(texCoord, new Vector2Int(1, 0), centerDepth, centerWorld);
        var verDeriv = ClosestDerivativeToAdjacentExtrapolations(texCoord, new Vector2Int(0, 1), centerDepth, centerWorld);
        if (horDeriv.sqrMagnitude <= 1e-8f || verDeriv.sqrMagnitude <= 1e-8f)
            return Vector3.zero;

        return -Vector3.Normalize(Vector3.Cross(horDeriv, verDeriv));
    }

    public Vector2Int NormalizedToDepthTexCoord(Vector2 uv)
    {
        int x = Mathf.Clamp(Mathf.RoundToInt(uv.x * TextureSize), 0, TextureSize - 1);
        int y = Mathf.Clamp(Mathf.RoundToInt((1f - uv.y) * TextureSize), 0, TextureSize - 1);
        return new Vector2Int(x, y);
    }

    private void ResolveRefs()
    {
        if (!depthManager)
            depthManager = GetComponent<EnvironmentDepthManager>();
    }

    private void InitializeResources()
    {
        if (_computeBuffer != null)
            return;

        _copyShader = Resources.Load<ComputeShader>("CopyDepthTexture");
        Assert.IsNotNull(_copyShader, "Compute shader 'CopyDepthTexture' not found in Resources.");

        int numPixels = TextureSize * TextureSize * NumEyes;
        _computeBuffer = new ComputeBuffer(numPixels, sizeof(float));
        _depthTexturePixels = new NativeArray<float>(numPixels, Allocator.Persistent);
        _gpuRequestBuffer = new NativeArray<float>(numPixels, Allocator.Persistent);
    }

    private void BindReflection()
    {
        if (depthManager == null)
            return;

        Type t = depthManager.GetType();
        if (_fiFrameDescriptors == null)
            _fiFrameDescriptors = t.GetField("frameDescriptors", BindingFlags.Instance | BindingFlags.NonPublic);
        if (_miGetTrackingSpaceWorldToLocalMatrix == null)
            _miGetTrackingSpaceWorldToLocalMatrix = t.GetMethod("GetTrackingSpaceWorldToLocalMatrix", BindingFlags.Instance | BindingFlags.NonPublic);
    }

    private void CreateTextureCopyRequestIfNeeded()
    {
        if (_currentGpuReadbackRequest.HasValue)
            return;
        if (!warmUpRaycast || depthManager == null || !depthManager.enabled || !depthManager.IsDepthAvailable)
        {
            InvalidateDepthTexture();
            return;
        }

        RenderTexture depthTexture = Shader.GetGlobalTexture(EnvironmentDepthTextureId) as RenderTexture;
        if (depthTexture == null)
            return;

        if (!TryReadFrameDescriptors(out _cachedFrameDescArray) || _cachedFrameDescArray == null || _cachedFrameDescArray.Length < NumEyes)
        {
            InvalidateDepthTexture();
            return;
        }

        _worldToTrackingSpaceMatrix = GetTrackingSpaceWorldToLocalMatrix();
        _copyShader.SetTexture(0, EnvironmentDepthTextureId, depthTexture);
        _copyShader.SetFloat(EnvironmentDepthTextureSizeId, depthTexture.width);
        _environmentDepthZBufferParams = Shader.GetGlobalVector(EnvironmentDepthZBufferParamsId);
        _copyShader.SetVector(EnvironmentDepthZBufferParamsId, _environmentDepthZBufferParams);
        _copyShader.SetBuffer(0, CopiedDepthTextureId, _computeBuffer);
        _copyShader.Dispatch(0, 1, 1, 1);

        _currentGpuReadbackRequest = AsyncGPUReadback.RequestIntoNativeArray(ref _gpuRequestBuffer, _computeBuffer);
    }

    private void UpdateTextureCopyRequest()
    {
        if (!_currentGpuReadbackRequest.HasValue || !_currentGpuReadbackRequest.Value.done)
            return;

        if (_currentGpuReadbackRequest.Value.hasError)
        {
            if (debugLog)
                Debug.LogError("[ScanCoverCustomDepthRaycaster_P1] AsyncGPUReadback request error.");
        }
        else
        {
            (_depthTexturePixels, _gpuRequestBuffer) = (_gpuRequestBuffer, _depthTexturePixels);
            UpdateMatricesFromFrameDescriptors();
            _isDepthTextureAvailable = true;
        }

        _currentGpuReadbackRequest = null;
    }

    private void UpdateMatricesFromFrameDescriptors()
    {
        if (_cachedFrameDescArray == null || _cachedFrameDescArray.Length < NumEyes)
            return;

        for (int i = 0; i < NumEyes; i++)
        {
            object frameDesc = _cachedFrameDescArray.GetValue(i);
            CalculateDepthCameraMatrices(frameDesc, out Matrix4x4 proj, out Matrix4x4 view);
            view *= _worldToTrackingSpaceMatrix;
            _matrixV[i] = view;
            _matrixVP[i] = proj * view;
            GeometryUtility.CalculateFrustumPlanes(_matrixVP[i], _camFrustumPlanes[i]);
            _matrixVPInv[i] = _matrixVP[i].inverse;
        }
    }

    private bool TryReadFrameDescriptors(out Array frameDescriptors)
    {
        frameDescriptors = null;
        if (_fiFrameDescriptors == null || depthManager == null)
            return false;

        frameDescriptors = _fiFrameDescriptors.GetValue(depthManager) as Array;
        return frameDescriptors != null;
    }

    private Matrix4x4 GetTrackingSpaceWorldToLocalMatrix()
    {
        if (_miGetTrackingSpaceWorldToLocalMatrix != null && depthManager != null)
        {
            object result = _miGetTrackingSpaceWorldToLocalMatrix.Invoke(depthManager, null);
            if (result is Matrix4x4 m)
                return m;
        }

        if (depthManager != null && depthManager.CustomTrackingSpace != null)
            return depthManager.CustomTrackingSpace.worldToLocalMatrix;
        return Matrix4x4.identity;
    }

    private static void CalculateDepthCameraMatrices(object frameDesc, out Matrix4x4 projMatrix, out Matrix4x4 viewMatrix)
    {
        float left = GetFrameFloat(frameDesc, "fovLeftAngleTangent");
        float right = GetFrameFloat(frameDesc, "fovRightAngleTangent");
        float bottom = GetFrameFloat(frameDesc, "fovDownAngleTangent");
        float top = GetFrameFloat(frameDesc, "fovTopAngleTangent");
        float near = GetFrameFloat(frameDesc, "nearZ");
        float far = GetFrameFloat(frameDesc, "farZ");
        Vector3 poseLocation = GetFrameVector3(frameDesc, "createPoseLocation");
        Quaternion poseRotation = GetFrameQuaternion(frameDesc, "createPoseRotation");

        float x = 2f / (right + left);
        float y = 2f / (top + bottom);
        float a = (right - left) / (right + left);
        float b = (top - bottom) / (top + bottom);
        float c;
        float d;
        if (float.IsInfinity(far))
        {
            c = -1f;
            d = -2f * near;
        }
        else
        {
            c = -(far + near) / (far - near);
            d = -(2f * far * near) / (far - near);
        }

        projMatrix = new Matrix4x4
        {
            m00 = x,
            m01 = 0f,
            m02 = a,
            m03 = 0f,
            m10 = 0f,
            m11 = y,
            m12 = b,
            m13 = 0f,
            m20 = 0f,
            m21 = 0f,
            m22 = c,
            m23 = d,
            m30 = 0f,
            m31 = 0f,
            m32 = -1f,
            m33 = 0f
        };

        viewMatrix = Matrix4x4.TRS(poseLocation, poseRotation, new Vector3(1f, 1f, -1f)).inverse;
    }

    private static float GetFrameFloat(object frameDesc, string fieldName)
    {
        FieldInfo fi = frameDesc.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        return fi != null ? (float)fi.GetValue(frameDesc) : 0f;
    }

    private static Vector3 GetFrameVector3(object frameDesc, string fieldName)
    {
        FieldInfo fi = frameDesc.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        return fi != null ? (Vector3)fi.GetValue(frameDesc) : Vector3.zero;
    }

    private static Quaternion GetFrameQuaternion(object frameDesc, string fieldName)
    {
        FieldInfo fi = frameDesc.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        return fi != null ? (Quaternion)fi.GetValue(frameDesc) : Quaternion.identity;
    }

    private void InvalidateDepthTexture()
    {
        _isDepthTextureAvailable = false;
    }

    private Vector3 ClosestDerivativeToAdjacentExtrapolations(Vector2Int texCoord, Vector2Int axis, float centerDepth, Vector3 centerWorld)
    {
        float d0 = SampleDepthTexture02(texCoord - axis);
        float d1 = SampleDepthTexture02(texCoord + axis);
        float d2 = SampleDepthTexture02(texCoord - axis * 2);
        float d3 = SampleDepthTexture02(texCoord + axis * 2);

        var ext0 = new Vector2(
            Mathf.Abs(Extrapolate(d0, d2) - centerDepth),
            Mathf.Abs(Extrapolate(d1, d3) - centerDepth));

        return ext0.x > ext0.y
            ? WorldPosAtDepthTexCoord02(texCoord + axis) - centerWorld
            : centerWorld - WorldPosAtDepthTexCoord02(texCoord - axis);
    }

    private static float Extrapolate(float d1, float d2)
    {
        float denom = 2f * d2 - d1;
        if (Mathf.Abs(denom) < 1e-6f)
            return 0f;
        return d1 * d2 / denom;
    }

    private static bool IsFinite(Vector3 p)
    {
        return !float.IsNaN(p.x) && !float.IsNaN(p.y) && !float.IsNaN(p.z) &&
               !float.IsInfinity(p.x) && !float.IsInfinity(p.y) && !float.IsInfinity(p.z);
    }
}
