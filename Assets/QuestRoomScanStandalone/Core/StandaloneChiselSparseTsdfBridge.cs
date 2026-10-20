using System;
using System.Reflection;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Adapts the real QuestRoomScanStandalone accepted depth packet to the
    /// isolated CHISEL sparse-TSDF proof.  It owns only four readback textures;
    /// production fusion, extraction and rendering never sample its state.
    /// </summary>
    [DefaultExecutionOrder(-15)]
    [DisallowMultipleComponent]
    internal sealed class StandaloneChiselSparseTsdfBridge : MonoBehaviour
    {
        [Header("CHSL observation source")]
        [Tooltip("Use exact-frame full-resolution depth after hand masking, before temporal/bilateral smoothing.")]
        [SerializeField] private bool usePreSpatialFilterDepth = true;

        // The reusable proof currently lives in Assembly-CSharp while this
        // production scene is in Genesis.RoomScan.asmdef. Unity forbids a named
        // assembly from directly referencing the predefined Assembly-CSharp,
        // so this tiny adapter resolves the public component once and caches
        // its public surface. No per-pixel or per-voxel reflection occurs.
        private Component _shadow;
        private MethodInfo _observeAcceptedStereoPair;
        private MethodInfo _clearShadow;
        private MethodInfo _setPreviewVisible;
        private PropertyInfo _acceptedStereoPairCount;
        private PropertyInfo _compactStatus;
        private PropertyInfo _hasPendingReadback;
        private PropertyInfo _previewVisible;
        private PropertyInfo _lastPreviewTriangleCount;
        private PropertyInfo _meshedChunkCount;
        private PropertyInfo _pendingMeshChunkCount;
        private PropertyInfo _lastMeshingMilliseconds;
        private PropertyInfo _minimumDepthMeters;
        private PropertyInfo _maximumDepthMeters;
        private ComputeShader _unproject;
        private int _kernel = -1;
        private RenderTexture _leftWorld;
        private RenderTexture _leftMeta;
        private RenderTexture _rightWorld;
        private RenderTexture _rightMeta;
        private int _width;
        private int _height;
        private string _failure = string.Empty;

        public bool RouteReady => _shadow != null &&
                                  _observeAcceptedStereoPair != null &&
                                  _unproject != null && _kernel >= 0 &&
                                  string.IsNullOrEmpty(_failure);
        public int AcceptedPairCount => ReadShadowProperty(
            _acceptedStereoPairCount, 0);
        public bool PreviewVisible => ReadShadowProperty(_previewVisible, false);
        public int PreviewTriangleCount => ReadShadowProperty(
            _lastPreviewTriangleCount, 0);
        public int MeshedChunkCount => ReadShadowProperty(
            _meshedChunkCount, 0);
        public int PendingMeshChunkCount => ReadShadowProperty(
            _pendingMeshChunkCount, 0);
        public float LastMeshingMilliseconds => ReadShadowProperty(
            _lastMeshingMilliseconds, 0f);
        public string CompactStatus => !string.IsNullOrEmpty(_failure)
            ? $"CHSL fault={_failure}"
            : _shadow != null
                ? ReadShadowProperty(_compactStatus, "CHSL waiting") +
                  $" src={(usePreSpatialFilterDepth ? "RAW" : "FILTERED")}"
                : "CHSL waiting";
        public string PreviewStatus =>
            $"CHSL-MC mesh={MeshedChunkCount} tri={PreviewTriangleCount} " +
            $"q={PendingMeshChunkCount} " +
            $"{LastMeshingMilliseconds:0.0}ms " +
            $"显示={(PreviewVisible ? "开" : "关")}";

        private void Awake()
        {
            EnsureRoute();
        }

        public void ObserveAcceptedFrame(DepthCapture depth)
        {
            Texture sourceDepth = depth != null && usePreSpatialFilterDepth
                ? depth.ChiselSourceDepthTex
                : depth != null ? depth.DepthTex : null;
            if (!EnsureRoute() || depth == null || !DepthCapture.DepthAvailable ||
                sourceDepth == null || depth.NormTex == null ||
                depth.Proj == null || depth.Proj.Length < 2 ||
                depth.ProjInv == null || depth.ProjInv.Length < 2 ||
                depth.View == null || depth.View.Length < 2 ||
                depth.ViewInv == null || depth.ViewInv.Length < 2)
                return;

            // Do not spend two unprojection dispatches for a pair that the
            // managed sparse route cannot accept yet.
            if (ReadShadowProperty(_hasPendingReadback, false))
                return;

            int width = sourceDepth.width;
            int height = sourceDepth.height;
            if (width <= 0 || height <= 0 || !EnsureTargets(width, height))
                return;

            DispatchEye(depth, sourceDepth, 0, _leftWorld, _leftMeta);
            DispatchEye(depth, sourceDepth, 1, _rightWorld, _rightMeta);

            Vector3 leftEye = depth.ViewInv[0].MultiplyPoint3x4(Vector3.zero);
            Vector3 rightEye = depth.ViewInv[1].MultiplyPoint3x4(Vector3.zero);
            // Keep the unprojected world points and the projective-depth test
            // on the exact same accepted DepthCapture packet.  Passing Proj
            // and View separately used to let the shadow rebuild this value;
            // the production contract now admits only the already paired
            // world-to-depth-clip matrix.
            Matrix4x4 leftWorldToDepthClip = depth.Proj[0] * depth.View[0];
            Matrix4x4 rightWorldToDepthClip = depth.Proj[1] * depth.View[1];
            try
            {
                _observeAcceptedStereoPair.Invoke(_shadow, new object[]
                {
                    depth.CurrentPlatformFrame,
                    _leftWorld, _leftMeta, leftEye,
                    leftWorldToDepthClip,
                    _rightWorld, _rightMeta, rightEye,
                    rightWorldToDepthClip
                });
            }
            catch (Exception e)
            {
                _failure = "observe-" + e.GetType().Name;
            }
        }

        public void ClearShadow()
        {
            if (_shadow == null || _clearShadow == null) return;
            try
            {
                _clearShadow.Invoke(_shadow, null);
            }
            catch (Exception e)
            {
                _failure = "clear-" + e.GetType().Name;
            }
        }

        public void SetPreviewVisible(bool visible)
        {
            if (!EnsureRoute() || _setPreviewVisible == null)
                return;
            try
            {
                _setPreviewVisible.Invoke(_shadow, new object[] { visible });
            }
            catch (Exception e)
            {
                _failure = "preview-" + e.GetType().Name;
            }
        }

        private bool EnsureRoute()
        {
            if (!string.IsNullOrEmpty(_failure))
                return false;
            if (_shadow == null)
            {
                Type shadowType = Type.GetType(
                    "ScanCoverChiselSparseTsdfShadow, Assembly-CSharp", false);
                if (shadowType == null || !typeof(Component).IsAssignableFrom(shadowType))
                {
                    _failure = "type-missing";
                    return false;
                }
                _shadow = GetComponent(shadowType) ?? gameObject.AddComponent(shadowType);
                _observeAcceptedStereoPair = shadowType.GetMethod(
                    "ObserveAcceptedStereoPair",
                    BindingFlags.Instance | BindingFlags.Public,
                    null,
                    new[]
                    {
                        typeof(int),
                        typeof(RenderTexture), typeof(RenderTexture),
                        typeof(Vector3), typeof(Matrix4x4),
                        typeof(RenderTexture), typeof(RenderTexture),
                        typeof(Vector3), typeof(Matrix4x4)
                    },
                    null);
                _clearShadow = shadowType.GetMethod(
                    "ClearShadow", BindingFlags.Instance | BindingFlags.Public);
                _setPreviewVisible = shadowType.GetMethod(
                    "SetPreviewVisible", BindingFlags.Instance | BindingFlags.Public);
                _acceptedStereoPairCount = shadowType.GetProperty(
                    "AcceptedStereoPairCount", BindingFlags.Instance | BindingFlags.Public);
                _compactStatus = shadowType.GetProperty(
                    "CompactStatus", BindingFlags.Instance | BindingFlags.Public);
                _hasPendingReadback = shadowType.GetProperty(
                    "HasPendingReadback", BindingFlags.Instance | BindingFlags.Public);
                _previewVisible = shadowType.GetProperty(
                    "PreviewVisible", BindingFlags.Instance | BindingFlags.Public);
                _lastPreviewTriangleCount = shadowType.GetProperty(
                    "LastPreviewTriangleCount", BindingFlags.Instance | BindingFlags.Public);
                _meshedChunkCount = shadowType.GetProperty(
                    "MeshedChunkCount", BindingFlags.Instance | BindingFlags.Public);
                _pendingMeshChunkCount = shadowType.GetProperty(
                    "PendingMeshChunkCount", BindingFlags.Instance | BindingFlags.Public);
                _lastMeshingMilliseconds = shadowType.GetProperty(
                    "LastMeshingMilliseconds", BindingFlags.Instance | BindingFlags.Public);
                _minimumDepthMeters = shadowType.GetProperty(
                    "MinimumDepthMeters", BindingFlags.Instance | BindingFlags.Public);
                _maximumDepthMeters = shadowType.GetProperty(
                    "MaximumDepthMeters", BindingFlags.Instance | BindingFlags.Public);
                if (_shadow == null || _observeAcceptedStereoPair == null ||
                    _clearShadow == null || _setPreviewVisible == null ||
                    _acceptedStereoPairCount == null || _compactStatus == null ||
                    _hasPendingReadback == null || _previewVisible == null ||
                    _lastPreviewTriangleCount == null ||
                    _meshedChunkCount == null ||
                    _pendingMeshChunkCount == null ||
                    _lastMeshingMilliseconds == null ||
                    _minimumDepthMeters == null ||
                    _maximumDepthMeters == null)
                {
                    _failure = "api-missing";
                    return false;
                }
            }
            if (_unproject == null)
            {
                _unproject = Resources.Load<ComputeShader>("ChiselDepthUnproject");
                if (_unproject == null)
                {
                    _failure = "shader-missing";
                    return false;
                }
                try
                {
                    _kernel = _unproject.FindKernel("UnprojectDepth");
                }
                catch
                {
                    _failure = "kernel-missing";
                    return false;
                }
            }
            return _kernel >= 0;
        }

        private T ReadShadowProperty<T>(PropertyInfo property, T fallback)
        {
            if (_shadow == null || property == null) return fallback;
            try
            {
                object value = property.GetValue(_shadow);
                return value is T typed ? typed : fallback;
            }
            catch
            {
                return fallback;
            }
        }

        private bool EnsureTargets(int width, int height)
        {
            if (_leftWorld != null && _leftWorld.IsCreated() &&
                _width == width && _height == height)
                return true;

            ReleaseTargets();
            _width = width;
            _height = height;
            _leftWorld = CreateTarget("CHSL Left World", width, height);
            _leftMeta = CreateTarget("CHSL Left Meta", width, height);
            _rightWorld = CreateTarget("CHSL Right World", width, height);
            _rightMeta = CreateTarget("CHSL Right Meta", width, height);
            if (_leftWorld == null || _leftMeta == null ||
                _rightWorld == null || _rightMeta == null)
            {
                _failure = "target-create";
                ReleaseTargets();
                return false;
            }
            return true;
        }

        private static RenderTexture CreateTarget(string name, int width, int height)
        {
            var target = new RenderTexture(width, height, 0,
                RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear)
            {
                name = name,
                dimension = UnityEngine.Rendering.TextureDimension.Tex2D,
                enableRandomWrite = true,
                useMipMap = false,
                autoGenerateMips = false,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            target.Create();
            if (target.IsCreated())
                return target;
            Destroy(target);
            return null;
        }

        private void DispatchEye(
            DepthCapture depth,
            Texture sourceDepth,
            int eye,
            RenderTexture world,
            RenderTexture meta)
        {
            _unproject.SetTexture(_kernel, "_DepthTex", sourceDepth);
            _unproject.SetTexture(_kernel, "_NormalTex", depth.NormTex);
            _unproject.SetTexture(_kernel, "_WorldPosition", world);
            _unproject.SetTexture(_kernel, "_ObservationMeta", meta);
            _unproject.SetInts("_ImageSize", _width, _height);
            _unproject.SetInt("_Eye", eye);
            _unproject.SetFloat("_MinimumDepthMeters",
                ReadShadowProperty(_minimumDepthMeters, 0.30f));
            _unproject.SetFloat("_MaximumDepthMeters",
                ReadShadowProperty(_maximumDepthMeters, 6.0f));
            _unproject.SetMatrix("_ProjectionInverse", depth.ProjInv[eye]);
            _unproject.SetMatrix("_ViewInverse", depth.ViewInv[eye]);
            _unproject.Dispatch(_kernel,
                Mathf.CeilToInt(_width / 8f),
                Mathf.CeilToInt(_height / 8f), 1);
        }

        private void OnDestroy()
        {
            ReleaseTargets();
        }

        private void ReleaseTargets()
        {
            Release(ref _leftWorld);
            Release(ref _leftMeta);
            Release(ref _rightWorld);
            Release(ref _rightMeta);
            _width = 0;
            _height = 0;
        }

        private static void Release(ref RenderTexture target)
        {
            if (target == null) return;
            target.Release();
            Destroy(target);
            target = null;
        }
    }
}
