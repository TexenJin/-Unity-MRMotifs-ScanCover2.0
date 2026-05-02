using System;
using System.Reflection;
using Meta.XR.EnvironmentDepth;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;

namespace MyProject.XR
{
    public enum Eye
    {
        Left = 0,
        Right = 1,
        Both = 2,
    }

    public enum DepthRaycastResult
    {
        Success = 0,
        NotReady = 1,
        NoHit = 2,
        RayOccluded = 3,
        HitPointOccluded = 4,
        RayOutsideOfDepthCameraFrustum = 5,
    }

    [AddComponentMenu("XR/Depth/Custom Raycaster")]
    [DefaultExecutionOrder(-48)]
    [DisallowMultipleComponent]
    public class CustomEnvironmentDepthRaycaster : MonoBehaviour
    {
        private static readonly int EnvironmentDepthTextureId = Shader.PropertyToID("_EnvironmentDepthTexture");
        private static readonly int EnvironmentDepthTextureSizeId = Shader.PropertyToID("_EnvironmentDepthTextureSize");
        private static readonly int CopiedDepthTextureId = Shader.PropertyToID("_CopiedDepthTexture");
        private static readonly int EnvironmentDepthZBufferParamsId = Shader.PropertyToID("_EnvironmentDepthZBufferParams");
        private const int NumEyes = 2;

        public const int TextureSize = 128;

        [SerializeField] private Eye eye = Eye.Right;
        [SerializeField] private bool warmUpRaycast = true;
        [SerializeField] private bool debugLog = false;

        [Header("Dependencies")]
        public EnvironmentDepthManager depthManager;

        private ComputeShader _copyShader;
        private ComputeBuffer _computeBuffer;
        private NativeArray<float> _depthTexturePixels;
        private NativeArray<float> _gpuRequestBuffer;
        private AsyncGPUReadbackRequest? _currentGpuReadbackRequest;
        private bool _isDepthTextureAvailable;
        private Eye _lastEye;
        private RenderTexture _updatedDepthTexture;

        private readonly Matrix4x4[] _matrixVP = new Matrix4x4[NumEyes];
        private readonly Matrix4x4[] _matrixV = new Matrix4x4[NumEyes];
        private readonly Matrix4x4[] _matrixVPInv = new Matrix4x4[NumEyes];
        private readonly Plane[][] _camFrustumPlanes = { new Plane[6], new Plane[6] };
        private Vector4 _environmentDepthZBufferParams;
        private int _currentEyeIndex;
        private Matrix4x4 _worldToTrackingSpaceMatrix = Matrix4x4.identity;

        private FieldInfo _fiFrameDescriptors;
        private MethodInfo _miGetTrackingSpaceWorldToLocalMatrix;
        private EventInfo _eiOnDepthTextureUpdate;
        private MethodInfo _miDepthTextureUpdateAdd;
        private MethodInfo _miDepthTextureUpdateRemove;
        private Array _cachedFrameDescArray;
        private Delegate _depthTextureUpdateHandler;

        public bool IsDepthTextureAvailable => _isDepthTextureAvailable;

        private void Awake()
        {
            ResolveRefs();
            BindReflection();
            InitializeResources();
            SetEye(eye);
            _lastEye = eye;
            SubscribeDepthUpdates(true);
        }

        private void OnEnable()
        {
            ResolveRefs();
            BindReflection();
            InitializeResources();
            SetEye(eye);
            _lastEye = eye;
            SubscribeDepthUpdates(true);
        }

        private void Update()
        {
            ResolveRefs();
            BindReflection();

            if (_lastEye != eye)
            {
                SetEye(eye);
                _lastEye = eye;
            }

            UpdateTextureCopyRequest();
            CreateTextureCopyRequestIfNeeded();
        }

        private void OnDisable()
        {
            SubscribeDepthUpdates(false);
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

        private void OnValidate()
        {
            SetEye(eye);
            _lastEye = eye;
        }

        public void SetEye(Eye selectedEye)
        {
            eye = selectedEye;
            switch (selectedEye)
            {
                case Eye.Left:
                    _currentEyeIndex = 0;
                    break;
                case Eye.Right:
                    _currentEyeIndex = 1;
                    break;
                default:
                    _currentEyeIndex = 0;
                    break;
            }
        }

        public void SetEye(int eyeIndex)
        {
            if (eyeIndex <= 0)
            {
                SetEye(Eye.Left);
                return;
            }

            if (eyeIndex == 1)
            {
                SetEye(Eye.Right);
                return;
            }

            SetEye(Eye.Both);
        }

        public Vector2Int WorldPosToNonNormalizedTextureCoords02(Vector3 worldPos)
        {
            var clipPos = _matrixVP[_currentEyeIndex] * new Vector4(worldPos.x, worldPos.y, worldPos.z, 1f);
            if (Mathf.Abs(clipPos.w) < 1e-6f)
                return new Vector2Int(-1, -1);

            var uv = (new Vector2(clipPos.x, clipPos.y) / clipPos.w + Vector2.one) * 0.5f;
            return new Vector2Int(
                Mathf.Clamp(Mathf.FloorToInt(uv.x * TextureSize), 0, TextureSize - 1),
                Mathf.Clamp(Mathf.FloorToInt(uv.y * TextureSize), 0, TextureSize - 1));
        }

        public float SampleDepthTexture02(Vector2Int texCoord)
        {
            if (!_depthTexturePixels.IsCreated || !IsInBounds02(texCoord))
                return 0f;

            return _depthTexturePixels[
                texCoord.x + texCoord.y * TextureSize + TextureSize * TextureSize * _currentEyeIndex];
        }

        public Vector3 WorldPosAtDepthTexCoord02(Vector2Int texCoord)
        {
            float linearDepth = SampleDepthTexture02(texCoord);
            if (linearDepth <= 0f)
                return InvalidVector();

            float clipDepth = _environmentDepthZBufferParams.x / linearDepth - _environmentDepthZBufferParams.y;
            const float oneOverSize = 1f / TextureSize;
            var clipPos = new Vector4(
                (texCoord.x + 0.5f) * oneOverSize * 2f - 1f,
                (texCoord.y + 0.5f) * oneOverSize * 2f - 1f,
                clipDepth,
                1f);

            Vector4 worldH = _matrixVPInv[_currentEyeIndex] * clipPos;
            return Mathf.Abs(worldH.w) > 1e-6f ? (Vector3)(worldH / worldH.w) : InvalidVector();
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

            Vector3 centerWorld = WorldPosAtDepthTexCoord02(texCoord);
            if (!IsFinite(centerWorld))
                return Vector3.zero;

            Vector3 horDeriv = ClosestDerivativeToAdjacentExtrapolations02(texCoord, new Vector2Int(1, 0), centerDepth, centerWorld);
            Vector3 verDeriv = ClosestDerivativeToAdjacentExtrapolations02(texCoord, new Vector2Int(0, 1), centerDepth, centerWorld);
            if (horDeriv.sqrMagnitude <= 1e-8f || verDeriv.sqrMagnitude <= 1e-8f)
                return Vector3.zero;

            return -Vector3.Normalize(Vector3.Cross(horDeriv, verDeriv));
        }

        public bool ReconstructNormalAtWorldPos02(Vector3 position, out Vector3 normal, out float confidence)
        {
            normal = Vector3.zero;
            confidence = 0f;

            Vector2Int tc = WorldPosToNonNormalizedTextureCoords02(position);
            if (!IsInBounds02(tc))
                return false;

            Vector3 n = ReconstructNormal02(tc);
            if (n.sqrMagnitude <= 1e-8f)
                return false;

            normal = n.normalized;
            confidence = 1f;
            return true;
        }

        public Vector2Int NormalizedToDepthTexCoord(Vector2 uv)
        {
            int x = Mathf.Clamp(Mathf.FloorToInt(uv.x * TextureSize), 0, TextureSize - 1);
            int y = Mathf.Clamp(Mathf.FloorToInt((1f - uv.y) * TextureSize), 0, TextureSize - 1);
            return new Vector2Int(x, y);
        }

        public (DepthRaycastResult status, Vector3 position, int eyeIndex) Raycast02(
            Ray ray,
            float maxDistance,
            Eye selectedEye,
            bool allowOccluded)
        {
            if (!_isDepthTextureAvailable)
                return (DepthRaycastResult.NotReady, default, 0);

            if (selectedEye != Eye.Both)
                return GetResult(selectedEye == Eye.Left ? 0 : 1);

            var left = GetResult(0);
            if (left.status == DepthRaycastResult.Success)
                return left;

            var right = GetResult(1);
            if (right.status == DepthRaycastResult.Success)
                return right;

            if (left.status == DepthRaycastResult.HitPointOccluded &&
                right.status == DepthRaycastResult.HitPointOccluded)
            {
                return Vector3.Distance(ray.origin, left.position) >
                       Vector3.Distance(ray.origin, right.position)
                    ? left
                    : right;
            }

            return left;

            (DepthRaycastResult status, Vector3 position, int eyeIndex) GetResult(int idx)
            {
                Ray localRay = ray;
                float localMaxDistance = maxDistance;

                if (!ClampRayOriginToCamFrustumPlanes02(ref localRay, _camFrustumPlanes[idx], ref localMaxDistance))
                    return (DepthRaycastResult.RayOutsideOfDepthCameraFrustum, default, idx);

                Plane nearPlane = _camFrustumPlanes[idx][4];
                if (Vector3.Dot(localRay.direction, nearPlane.normal) < 0f &&
                    nearPlane.Raycast(localRay, out float nearDistance) &&
                    localMaxDistance > nearDistance)
                {
                    localMaxDistance = nearDistance;
                }

                DepthRaycastResult status = RaycastInternal02(localRay, out Vector3 hitPos, localMaxDistance, idx, allowOccluded);
                return (status, hitPos, idx);
            }
        }

        public DepthRaycastResult RaycastInternal02(
            Ray ray,
            out Vector3 position,
            float maxDistance,
            int eyeIndex,
            bool allowOccluded)
        {
            position = default;
            if (maxDistance < 0.01f)
                return DepthRaycastResult.NoHit;

            _currentEyeIndex = eyeIndex;
            Vector3 origin = ray.origin;
            Vector3 dir = ray.direction;
            Vector3 end = origin + dir * maxDistance;

            Vector2Int projOrig = WorldPosToNonNormalizedTextureCoords02(origin);
            if (!IsInBounds02(projOrig))
                return DepthRaycastResult.RayOutsideOfDepthCameraFrustum;

            if (!allowOccluded)
            {
                float rayDepth = WorldPosToLinearDepth02(origin);
                float envDepth = SampleDepthTexture02(projOrig);
                if (envDepth > 0f && rayDepth > envDepth)
                    return DepthRaycastResult.RayOccluded;
            }

            Vector2Int projEnd = WorldPosToNonNormalizedTextureCoords02(end);
            int dx = projEnd.x - projOrig.x;
            int dy = projEnd.y - projOrig.y;
            int steps = Math.Max(Math.Abs(dx), Math.Abs(dy));

            if (steps == 0)
            {
                float envDepth = SampleDepthTexture02(projOrig);
                float rayDepth = WorldPosToLinearDepth02(origin);
                if (envDepth > 0f && rayDepth < envDepth && WorldPosToLinearDepth02(end) > envDepth)
                {
                    position = origin + dir * envDepth;
                    return DepthRaycastResult.Success;
                }

                return DepthRaycastResult.NoHit;
            }

            float invStart = InverseSafe(WorldPosToLinearDepth02(origin));
            float invEnd = InverseSafe(WorldPosToLinearDepth02(end));
            float stepX = dx / (float)steps;
            float stepY = dy / (float)steps;
            float invDelta = (invEnd - invStart) / steps;
            float cx = projOrig.x;
            float cy = projOrig.y;
            float invDepth = invStart;
            bool seenEmpty = false;

            for (int i = 0; i <= steps; i++)
            {
                Vector2Int tc = new Vector2Int((int)cx, (int)cy);
                if (!IsInBounds02(tc))
                    return DepthRaycastResult.RayOutsideOfDepthCameraFrustum;

                float envDepth = SampleDepthTexture02(tc);
                if (envDepth > 0f)
                {
                    float rayDepth = InverseSafe(invDepth);
                    if (!seenEmpty)
                    {
                        seenEmpty = envDepth > rayDepth;
                    }
                    else if (envDepth <= rayDepth)
                    {
                        Vector2Int prevTc = new Vector2Int((int)(cx - stepX), (int)(cy - stepY));
                        float prevDepth = SampleDepthTexture02(prevTc);
                        Vector3 wp1 = WorldPosAtDepthTexCoord02(prevTc);
                        Vector3 wp2 = WorldPosAtDepthTexCoord02(tc);
                        position = ClosestPointOnFirstRay02(origin, dir, wp1, wp2 - wp1);
                        return prevDepth - envDepth > 0.3f
                            ? DepthRaycastResult.HitPointOccluded
                            : DepthRaycastResult.Success;
                    }
                }

                cx += stepX;
                cy += stepY;
                invDepth += invDelta;
            }

            return seenEmpty ? DepthRaycastResult.NoHit : DepthRaycastResult.RayOccluded;
        }

        public static Vector3 ClosestPointOnFirstRay02(
            Vector3 p1, Vector3 d1, Vector3 p2, Vector3 d2)
        {
            Vector3 v3 = p2 - p1;
            Vector3 cross12 = Vector3.Cross(d1, d2);
            Vector3 cross32 = Vector3.Cross(v3, d2);
            float denom = Vector3.Dot(cross12, cross12);
            if (denom <= 1e-8f)
                return p1;
            float s = Vector3.Dot(cross32, cross12) / denom;
            return p1 + d1 * s;
        }

        public static bool ClampRayOriginToCamFrustumPlanes02(
            ref Ray ray, Plane[] planes, ref float maxDistance)
        {
            if (GeometryUtility.TestPlanesAABB(planes, new Bounds(ray.origin, Vector3.zero)))
                return true;

            for (int i = 0; i < 5; i++)
            {
                if (planes[i].Raycast(ray, out float distance))
                {
                    const float tolerance = 0.01f;
                    if (GeometryUtility.TestPlanesAABB(
                            planes,
                            new Bounds(ray.GetPoint(distance + tolerance), Vector3.zero)))
                    {
                        maxDistance -= distance;
                        if (maxDistance <= 0f)
                            return false;
                        ray.origin = ray.GetPoint(distance);
                        return true;
                    }
                }
            }

            return false;
        }

        public static bool IsInBounds02(Vector2Int tc)
        {
            return tc.x >= 0 && tc.x < TextureSize && tc.y >= 0 && tc.y < TextureSize;
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

            Type depthManagerType = depthManager.GetType();
            if (_fiFrameDescriptors == null)
                _fiFrameDescriptors = depthManagerType.GetField("frameDescriptors", BindingFlags.Instance | BindingFlags.NonPublic);
            if (_miGetTrackingSpaceWorldToLocalMatrix == null)
                _miGetTrackingSpaceWorldToLocalMatrix = depthManagerType.GetMethod("GetTrackingSpaceWorldToLocalMatrix", BindingFlags.Instance | BindingFlags.NonPublic);
            if (_eiOnDepthTextureUpdate == null)
                _eiOnDepthTextureUpdate = depthManagerType.GetEvent("onDepthTextureUpdate", BindingFlags.Instance | BindingFlags.NonPublic);
            if (_miDepthTextureUpdateAdd == null && _eiOnDepthTextureUpdate != null)
                _miDepthTextureUpdateAdd = _eiOnDepthTextureUpdate.GetAddMethod(true);
            if (_miDepthTextureUpdateRemove == null && _eiOnDepthTextureUpdate != null)
                _miDepthTextureUpdateRemove = _eiOnDepthTextureUpdate.GetRemoveMethod(true);
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

            RenderTexture depthTexture = _updatedDepthTexture;
            if (depthTexture == null)
                depthTexture = Shader.GetGlobalTexture(EnvironmentDepthTextureId) as RenderTexture;
            if (depthTexture == null)
                return;
            _updatedDepthTexture = null;

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
                    Debug.LogError("[CustomEnvironmentDepthRaycaster] AsyncGPUReadback request error.");
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
                if (result is Matrix4x4 matrix)
                    return matrix;
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
            FieldInfo field = frameDesc.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
            return field != null ? (float)field.GetValue(frameDesc) : 0f;
        }

        private static Vector3 GetFrameVector3(object frameDesc, string fieldName)
        {
            FieldInfo field = frameDesc.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
            return field != null ? (Vector3)field.GetValue(frameDesc) : Vector3.zero;
        }

        private static Quaternion GetFrameQuaternion(object frameDesc, string fieldName)
        {
            FieldInfo field = frameDesc.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
            return field != null ? (Quaternion)field.GetValue(frameDesc) : Quaternion.identity;
        }

        private void InvalidateDepthTexture()
        {
            _isDepthTextureAvailable = false;
            _updatedDepthTexture = null;
        }

        private void SubscribeDepthUpdates(bool subscribe)
        {
            if (depthManager == null)
                return;

            BindReflection();
            if (_eiOnDepthTextureUpdate == null || _miDepthTextureUpdateAdd == null || _miDepthTextureUpdateRemove == null)
                return;

            if (_depthTextureUpdateHandler == null)
            {
                MethodInfo method = GetType().GetMethod(nameof(OnDepthTextureUpdate), BindingFlags.Instance | BindingFlags.NonPublic);
                _depthTextureUpdateHandler = Delegate.CreateDelegate(_eiOnDepthTextureUpdate.EventHandlerType, this, method);
            }

            if (subscribe)
            {
                _miDepthTextureUpdateRemove.Invoke(depthManager, new object[] { _depthTextureUpdateHandler });
                _miDepthTextureUpdateAdd.Invoke(depthManager, new object[] { _depthTextureUpdateHandler });
            }
            else
            {
                _miDepthTextureUpdateRemove.Invoke(depthManager, new object[] { _depthTextureUpdateHandler });
            }
        }

        private void OnDepthTextureUpdate(RenderTexture updatedDepthTexture)
        {
            _updatedDepthTexture = updatedDepthTexture;
            CreateTextureCopyRequestIfNeeded();
        }

        private Vector3 ClosestDerivativeToAdjacentExtrapolations02(Vector2Int texCoord, Vector2Int axis, float centerDepth, Vector3 centerWorld)
        {
            float d0 = SampleDepthTexture02(texCoord - axis);
            float d1 = SampleDepthTexture02(texCoord + axis);
            float d2 = SampleDepthTexture02(texCoord - axis * 2);
            float d3 = SampleDepthTexture02(texCoord + axis * 2);

            var ext = new Vector2(
                Mathf.Abs(Extrapolate(d0, d2) - centerDepth),
                Mathf.Abs(Extrapolate(d1, d3) - centerDepth));

            return ext.x > ext.y
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

        private static Vector3 InvalidVector()
        {
            return new Vector3(float.NaN, float.NaN, float.NaN);
        }

        private static float InverseSafe(float value)
        {
            return Mathf.Abs(value) > 1e-6f ? 1f / value : 0f;
        }
    }
}
