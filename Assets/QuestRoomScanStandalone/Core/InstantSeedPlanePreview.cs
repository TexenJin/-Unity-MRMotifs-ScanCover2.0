using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Centre-disc seed-plane experiment. Green shows current fitted local
    /// patches, cyan the preceding fitted capture, and amber keeps the current
    /// source depth for other accepted triangles. The displayed meshes remain
    /// diagnostic; only the fitted world-plane parameters are staged for the
    /// single shared GunGel -> court -> TSDF production input after a clear.
    /// </summary>
    public sealed class InstantSeedPlanePreview : MonoBehaviour
    {
        private const int Hypotheses = 64;
        private const int MaxPlanePatches = 8;
        // Diameter grows from 50% to 60% of the shorter image side. The outer
        // image edge remains excluded; each local support region competes
        // independently instead of forcing one plane across every structure.
        private const float DiscRadiusFraction = 0.30f;
        private static readonly int[] LocalOffsets = { 4, -4, 7, -7 };

        private struct Sample
        {
            public Vector3 World;
            public Vector3 Normal;
            public bool NormalValid;
            public float Distance;
            public float LinearDepth;
            public bool Valid;
        }

        private struct PlanePatch
        {
            public Vector3 Normal;
            public Vector3 Point;
            public Vector3 AxisU;
            public Vector3 AxisV;
            public float MinU, MaxU, MinV, MaxV;
        }

        private RenderTexture _sourceCopy;
        private Mesh _planeMesh;
        private Mesh _referencePlaneMesh;
        private Mesh _otherMesh;
        private MeshRenderer _planeRenderer;
        private MeshRenderer _referencePlaneRenderer;
        private MeshRenderer _otherRenderer;
        private Material _planeMaterial;
        private Material _referencePlaneMaterial;
        private Material _otherMaterial;
        private bool _currentPlaneValid;
        private bool _referencePlaneValid;
        private int _currentPatchCount;
        private int _referencePatchCount;
        private Vector3 _currentPlaneNormal;
        private Vector3 _currentPlanePoint;
        private Vector3 _referencePlaneNormal;
        private Vector3 _referencePlanePoint;
        private bool _active;
        private bool _visible;
        private bool _pending;
        private int _generation;
        private int _width;
        private int _height;
        private int _pixelStep;
        private Matrix4x4 _projectionInverse;
        private Matrix4x4 _viewInverse;

        public bool IsPending => _pending;
        public string Status { get; private set; } = "未取样";

        public void SetVisible(bool visible)
        {
            SetCaptureState(visible, visible);
        }

        /// <summary>
        /// Keep fitting independent from rendering.  Automatic production
        /// bootstrap runs active but hidden, so taking the pre-fusion ruler
        /// does not require entering a diagnostic display mode.
        /// </summary>
        public void SetCaptureState(bool active, bool visible)
        {
            _active = active;
            _visible = active && visible;
            if (!active)
            {
                _generation++;
                _pending = false;
                ClearMeshes();
                Status = "未取样";
            }
            UpdateVisibility();
        }

        public bool Capture(RenderTexture packedDepth, Matrix4x4 projectionInverse,
            Matrix4x4 viewInverse, int pixelStep)
        {
            if (!_active || _pending || packedDepth == null ||
                !SystemInfo.supportsAsyncGPUReadback)
            {
                if (_active && !SystemInfo.supportsAsyncGPUReadback)
                    Status = "设备不支持异步深度回读";
                return false;
            }

            EnsureObjects(packedDepth.width, packedDepth.height);
            if (_planeRenderer == null || _referencePlaneRenderer == null ||
                _otherRenderer == null)
            {
                Status = "缺少种子观察材质";
                return false;
            }
            if (_sourceCopy == null || !_sourceCopy.IsCreated())
            {
                Status = "单帧深度副本不可用";
                return false;
            }

            _projectionInverse = projectionInverse;
            _viewInverse = viewInverse;
            _width = packedDepth.width;
            _height = packedDepth.height;
            _pixelStep = Mathf.Max(1, pixelStep);
            _pending = true;
            int generation = ++_generation;
            Status = "中央圆单帧拟合中";

            // Own this exact GPU frame until the readback completes. Live-shell
            // ring textures may be overwritten before its callback arrives.
            try
            {
                Graphics.CopyTexture(packedDepth, _sourceCopy);
                AsyncGPUReadback.Request(_sourceCopy, 0, TextureFormat.RGBAFloat,
                    request => OnReadback(request, generation));
                return true;
            }
            catch (Exception error)
            {
                _pending = false;
                Status = "单帧回读启动失败: " + error.GetType().Name;
                return false;
            }
        }

        private void OnReadback(AsyncGPUReadbackRequest request, int generation)
        {
            if (generation != _generation || !_active) return;
            _pending = false;
            if (request.hasError)
            {
                Status = "单帧回读失败";
                return;
            }

            try
            {
                var pixels = request.GetData<Color>();
                if (pixels.Length != _width * _height)
                {
                    Status = "深度尺寸不符";
                    return;
                }
                BuildPreview(pixels);
            }
            catch (Exception error)
            {
                ClearMeshes();
                Status = "拟合失败: " + error.GetType().Name;
            }
        }

        private void EnsureObjects(int width, int height)
        {
            if (_sourceCopy == null || _sourceCopy.width != width ||
                _sourceCopy.height != height)
            {
                if (_sourceCopy != null)
                {
                    _sourceCopy.Release();
                    Destroy(_sourceCopy);
                }
                _sourceCopy = new RenderTexture(width, height, 0,
                    UnityEngine.Experimental.Rendering.GraphicsFormat.R16G16B16A16_SFloat)
                {
                    name = "QRS Seed Plane Exact Source Frame",
                    dimension = TextureDimension.Tex2D,
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp,
                    useMipMap = false,
                    autoGenerateMips = false
                };
                _sourceCopy.Create();
            }

            if (_planeRenderer != null) return;
            Shader shader = Resources.Load<Shader>("InstantSeedPlanePreview");
            if (shader == null)
            {
                Status = "缺少种子观察材质";
                return;
            }
            _planeMaterial = new Material(shader) { name = "QRS Seed Plane Green" };
            _planeMaterial.SetColor("_LineColor", new Color(0.12f, 1f, 0.35f, 0.95f));
            _referencePlaneMaterial = new Material(shader) { name = "QRS Previous Seed Cyan" };
            _referencePlaneMaterial.SetColor("_LineColor", new Color(0.10f, 0.88f, 1f, 0.62f));
            _otherMaterial = new Material(shader) { name = "QRS Unfitted Depth Amber" };
            _otherMaterial.SetColor("_LineColor", new Color(1f, 0.58f, 0.08f, 0.72f));
            _planeRenderer = CreateRenderer("[QRS] Single Frame Fitted Seed", _planeMaterial);
            _referencePlaneRenderer = CreateRenderer("[QRS] Previous Fitted Seed", _referencePlaneMaterial);
            _otherRenderer = CreateRenderer("[QRS] Single Frame Unfitted Structure", _otherMaterial);
        }

        private MeshRenderer CreateRenderer(string name, Material material)
        {
            var child = new GameObject(name);
            child.transform.SetParent(transform, false);
            child.AddComponent<MeshFilter>();
            MeshRenderer renderer = child.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = material;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            renderer.enabled = false;
            return renderer;
        }

        private void BuildPreview(Unity.Collections.NativeArray<Color> pixels)
        {
            // Match the live shell's actual source lattice. Its recent 2x
            // larger visible squares are only a shader drawing choice.
            int step = _pixelStep;
            int nx = (_width - 1) / step + 1;
            int ny = (_height - 1) / step + 1;
            var samples = new Sample[nx * ny];
            var valid = new List<int>(samples.Length / 3);
            float radius = Mathf.Min(_width, _height) * DiscRadiusFraction;
            float radius2 = radius * radius;
            Vector3 camera = _viewInverse.MultiplyPoint3x4(Vector3.zero);

            for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
            {
                int px = x * step;
                int py = y * step;
                float dx = px + 0.5f - _width * 0.5f;
                float dy = py + 0.5f - _height * 0.5f;
                if (dx * dx + dy * dy > radius2) continue;
                Color packed = pixels[py * _width + px];
                float ndc = packed.r;
                if (ndc <= 0f || ndc >= 1f || !IsFinite(ndc)) continue;
                Vector4 clip = new Vector4(
                    2f * (px + 0.5f) / _width - 1f,
                    2f * (py + 0.5f) / _height - 1f,
                    ndc * 2f - 1f, 1f);
                Vector4 viewH = _projectionInverse * clip;
                if (!IsFinite(viewH.w) || Mathf.Abs(viewH.w) < 1e-6f) continue;
                float linearDepth = Mathf.Abs(viewH.z / viewH.w);
                if (!IsFinite(linearDepth) || linearDepth <= 0.12f ||
                    linearDepth >= 8f) continue;
                Vector4 worldH = _viewInverse * viewH;
                if (!IsFinite(worldH.w) || Mathf.Abs(worldH.w) < 1e-6f) continue;
                Vector3 world = new Vector3(worldH.x, worldH.y, worldH.z) / worldH.w;
                float distance = Vector3.Distance(camera, world);
                if (!IsFinite(distance)) continue;
                Vector3 surfaceNormal = new Vector3(packed.g, packed.b, packed.a);
                bool normalValid = IsFinite(surfaceNormal.x) &&
                                   IsFinite(surfaceNormal.y) &&
                                   IsFinite(surfaceNormal.z) &&
                                   surfaceNormal.sqrMagnitude > 0.0625f;
                int index = y * nx + x;
                samples[index] = new Sample
                {
                    World = world,
                    Normal = normalValid ? surfaceNormal.normalized : Vector3.zero,
                    NormalValid = normalValid,
                    Distance = distance,
                    LinearDepth = linearDepth,
                    Valid = true
                };
                valid.Add(index);
            }

            // Reproduce the shell's actual accepted triangles, including its
            // four-corner coplanar rescue. Rejected diagnostic edges are not
            // promoted to a fitted surface merely because the shell draws them.
            var acceptedFirst = new bool[(nx - 1) * (ny - 1)];
            var acceptedSecond = new bool[acceptedFirst.Length];
            for (int y = 0; y < ny - 1; y++)
            for (int x = 0; x < nx - 1; x++)
            {
                int a = y * nx + x, b = a + 1, c = a + nx, d = c + 1;
                int cell = y * (nx - 1) + x;
                int first = ClassifyShellTriangle(samples[a], samples[b], samples[c]);
                int second = ClassifyShellTriangle(samples[b], samples[d], samples[c]);
                bool rescue = (first == 2 || second == 2) &&
                              CanRescueCoplanarCell(samples[a], samples[b],
                                  samples[c], samples[d]);
                acceptedFirst[cell] = first == 0 || first == 2 && rescue;
                acceptedSecond[cell] = second == 0 || second == 2 && rescue;
            }

            var patchIds = new int[samples.Length];
            for (int i = 0; i < patchIds.Length; i++) patchIds[i] = -1;
            var patches = new List<PlanePatch>(MaxPlanePatches);
            var remaining = new List<int>(valid);
            int minimum = MinimumLocalSupport(valid.Count);
            int fittedPoints = 0;
            float firstP95 = 0f;
            string rejection = "无连续合格局部面";
            for (int patchIndex = 0; patchIndex < MaxPlanePatches &&
                 remaining.Count >= minimum; patchIndex++)
            {
                if (!TryFitLocalPlane(samples, remaining, nx, ny, step, minimum,
                        acceptedFirst, acceptedSecond,
                        out Vector3 planeNormal, out Vector3 planePoint,
                        out int planeAnchor, out _, out float p95,
                        out rejection))
                    break;

                var eligible = new bool[samples.Length];
                for (int i = 0; i < remaining.Count; i++)
                {
                    int index = remaining[i];
                    eligible[index] = samples[index].NormalValid &&
                        Mathf.Abs(Vector3.Dot(samples[index].Normal,
                            planeNormal)) >= 0.82f &&
                        Mathf.Abs(Vector3.Dot(samples[index].World - planePoint,
                            planeNormal)) <= FitTolerance(samples[index].Distance);
                }
                List<int> coherent = FloodConnectedFromSeed(eligible,
                    planeAnchor, nx, ny, acceptedFirst, acceptedSecond,
                    out bool[] mask);
                if (coherent.Count < minimum ||
                    !HasDrawableTriangle(mask, nx, ny,
                        acceptedFirst, acceptedSecond))
                {
                    rejection = "拟合点没有连续三角";
                    break;
                }

                for (int i = 0; i < coherent.Count; i++)
                    patchIds[coherent[i]] = patches.Count;
                patches.Add(BuildPlanePatch(samples, coherent,
                    planeNormal, planePoint));
                fittedPoints += coherent.Count;
                if (patches.Count == 1) firstP95 = p95;
                remaining.RemoveAll(index => patchIds[index] >= 0);
            }

            // Only interior vertices move. A fitted triangle touching raw
            // structure shares its original border vertices with that structure.
            bool[] interior = FindPatchInteriors(patchIds, nx, ny);

            var planeLines = new List<Vector3>(4096);
            var otherLines = new List<Vector3>(4096);
            int disconnectedTriangles = 0;
            for (int y = 0; y < ny - 1; y++)
            for (int x = 0; x < nx - 1; x++)
            {
                int a = y * nx + x;
                int b = a + 1;
                int c = a + nx;
                int d = c + 1;
                int cell = y * (nx - 1) + x;
                if (!AddTriangle(a, b, c, acceptedFirst[cell], samples,
                        patchIds, interior, patches,
                        otherLines, planeLines) &&
                    samples[a].Valid && samples[b].Valid && samples[c].Valid)
                    disconnectedTriangles++;
                if (!AddTriangle(b, d, c, acceptedSecond[cell], samples,
                        patchIds, interior, patches,
                        otherLines, planeLines) &&
                    samples[b].Valid && samples[d].Valid && samples[c].Valid)
                    disconnectedTriangles++;
            }

            // On each A capture, retain the previous successful fitted patch
            // set as a world-locked cyan reference. Failed new fits still
            // show their raw amber geometry and do not erase that reference.
            if (_currentPlaneValid && _planeMesh != null && _planeMesh.vertexCount > 0)
            {
                if (_referencePlaneMesh != null) Destroy(_referencePlaneMesh);
                _referencePlaneMesh = Instantiate(_planeMesh);
                _referencePlaneMesh.name = "QRS Previous Fitted Seed Lines";
                _referencePlaneRenderer.GetComponent<MeshFilter>().sharedMesh = _referencePlaneMesh;
                _referencePlaneNormal = _currentPlaneNormal;
                _referencePlanePoint = _currentPlanePoint;
                _referencePlaneValid = true;
                _referencePatchCount = _currentPatchCount;
            }

            ReplaceMesh(ref _planeMesh, _planeRenderer, planeLines, "QRS Fitted Seed Lines");
            ReplaceMesh(ref _otherMesh, _otherRenderer, otherLines, "QRS Unfitted Structure Lines");
            _currentPlaneValid = patches.Count > 0 && planeLines.Count > 0;
            _currentPatchCount = _currentPlaneValid ? patches.Count : 0;
            if (_currentPlaneValid)
            {
                _currentPlaneNormal = patches[0].Normal;
                _currentPlanePoint = patches[0].Point;
            }
            bool productionReady = false;
            if (_currentPlaneValid && DepthCapture.Instance != null)
            {
                var production = new DepthCapture.SeedPlanePatch[patches.Count];
                for (int i = 0; i < patches.Count; i++)
                {
                    PlanePatch patch = patches[i];
                    production[i] = new DepthCapture.SeedPlanePatch(
                        patch.Normal, patch.Point, patch.AxisU, patch.AxisV,
                        patch.MinU, patch.MaxU, patch.MinV, patch.MaxV);
                }
                productionReady = DepthCapture.Instance.StageSeedPlanePatches(production);
            }
            UpdateVisibility();
            int greenTriangles = planeLines.Count / 6;
            int amberTriangles = otherLines.Count / 6;
            string comparison = _currentPlaneValid && _referencePlaneValid &&
                                _currentPatchCount == 1 && _referencePatchCount == 1
                ? ComparisonStatus() : _referencePlaneValid ? " 青=上张面组" : "";
            DepthCapture productionDepth = DepthCapture.Instance;
            string productionStatus = !_currentPlaneValid ? ""
                : productionReady ? " 已送入口(离开预览生效)"
                : productionDepth != null && productionDepth.SeedPlaneAwaitClear
                    ? " 基底待清卷后生效"
                    : " 生产入口未就绪";
            Status = _currentPlaneValid
                ? $"局部拟合{patches.Count}块 绿{greenTriangles} 橙{amberTriangles} 断{disconnectedTriangles} 点{fittedPoints}/{valid.Count} 首面P95={firstP95 * 1000f:0}mm{productionStatus}{comparison}"
                : $"本帧不成板({rejection}) 橙{amberTriangles} 断{disconnectedTriangles} 有效{valid.Count}{comparison}";
        }

        private static PlanePatch BuildPlanePatch(Sample[] samples,
            List<int> members, Vector3 normal, Vector3 point)
        {
            Vector3 axisU = Vector3.Cross(normal, Vector3.up);
            if (axisU.sqrMagnitude < 0.01f)
                axisU = Vector3.Cross(normal, Vector3.right);
            axisU.Normalize();
            Vector3 axisV = Vector3.Cross(normal, axisU).normalized;
            var patch = new PlanePatch
            {
                Normal = normal, Point = point,
                AxisU = axisU, AxisV = axisV,
                MinU = float.MaxValue, MaxU = float.MinValue,
                MinV = float.MaxValue, MaxV = float.MinValue
            };
            for (int i = 0; i < members.Count; i++)
            {
                Vector3 world = samples[members[i]].World;
                float u = Vector3.Dot(world, axisU);
                float v = Vector3.Dot(world, axisV);
                patch.MinU = Mathf.Min(patch.MinU, u);
                patch.MaxU = Mathf.Max(patch.MaxU, u);
                patch.MinV = Mathf.Min(patch.MinV, v);
                patch.MaxV = Mathf.Max(patch.MaxV, v);
            }
            return patch;
        }

        private static bool TryFitLocalPlane(Sample[] samples, List<int> valid,
            int nx, int ny, int step, int minimum,
            bool[] acceptedFirst, bool[] acceptedSecond,
            out Vector3 normal, out Vector3 point,
            out int anchor, out int inliers, out float p95, out string rejection)
        {
            normal = Vector3.up;
            point = Vector3.zero;
            anchor = -1;
            inliers = 0;
            p95 = 0f;
            rejection = "有效点不足";
            if (valid.Count < minimum) return false;

            // Each hypothesis is born from three nearby pixels. A doorway,
            // ceiling and side wall may all be inside the disc; none needs to
            // own 45% of every valid pixel. Only a continuous, low-residual
            // local patch can become the one visible seed plane.
            var random = new System.Random(7319);
            int bestCount = 0;
            for (int h = 0; h < Hypotheses; h++)
            {
                int seed = valid[random.Next(valid.Count)];
                if (!TryLocalHypothesis(samples, seed, nx, ny,
                    out Vector3 hypothesisNormal, out Vector3 hypothesisPoint))
                    continue;

                var eligible = new bool[samples.Length];
                for (int i = 0; i < valid.Count; i++)
                {
                    int index = valid[i];
                    Sample s = samples[index];
                    eligible[index] = s.NormalValid &&
                        Mathf.Abs(Vector3.Dot(s.Normal, hypothesisNormal)) >= 0.82f &&
                        Mathf.Abs(Vector3.Dot(s.World - hypothesisPoint,
                            hypothesisNormal)) <= FitTolerance(s.Distance);
                }
                List<int> group = FloodConnectedFromSeed(eligible,
                    seed, nx, ny, acceptedFirst, acceptedSecond, out _);
                if (group.Count < minimum) continue;
                if (!HasAreaFootprint(group, nx, ny, step)) continue;

                Vector3 centre = MeanPoint(samples, group);
                Vector3 refinedNormal = SmallestCovarianceAxis(samples, group, centre);
                if (refinedNormal.sqrMagnitude < 0.5f) continue;
                int candidateAnchor = ClosestMember(samples, group, centre,
                    refinedNormal);
                if (candidateAnchor < 0) continue;

                var refinedEligible = new bool[samples.Length];
                for (int i = 0; i < valid.Count; i++)
                {
                    int index = valid[i];
                    Sample s = samples[index];
                    refinedEligible[index] = s.NormalValid &&
                        Mathf.Abs(Vector3.Dot(s.Normal, refinedNormal)) >= 0.82f &&
                        Mathf.Abs(Vector3.Dot(s.World - centre,
                            refinedNormal)) <= FitTolerance(s.Distance);
                }
                List<int> refinedGroup = FloodConnectedFromSeed(
                    refinedEligible, candidateAnchor, nx, ny,
                    acceptedFirst, acceptedSecond, out _);
                if (refinedGroup.Count < minimum ||
                    refinedGroup.Count <= bestCount ||
                    !HasAreaFootprint(refinedGroup, nx, ny, step))
                    continue;
                Vector3 finalCentre = MeanPoint(samples, refinedGroup);
                Vector3 finalNormal = SmallestCovarianceAxis(samples,
                    refinedGroup, finalCentre);
                if (finalNormal.sqrMagnitude < 0.5f) continue;
                int finalAnchor = ClosestMember(samples, refinedGroup,
                    finalCentre, finalNormal);
                if (finalAnchor < 0) continue;
                float residual = ResidualP95(samples, refinedGroup, finalCentre,
                    finalNormal);
                if (residual > 0.018f) continue;

                // The winning patch stays anchored to its own source surface.
                // A parallel but disconnected wall cannot take over merely
                // because the same infinite fitted plane passes through it.
                bestCount = refinedGroup.Count;
                normal = finalNormal;
                point = finalCentre;
                anchor = finalAnchor;
                inliers = refinedGroup.Count;
                p95 = residual;
            }

            if (anchor >= 0) return true;
            rejection = "无连续合格局部面";
            return false;
        }

        private static int MinimumLocalSupport(int validCount) =>
            Mathf.Max(64, Mathf.CeilToInt(validCount * 0.02f));

        private static bool TryLocalHypothesis(Sample[] samples, int seed,
            int nx, int ny, out Vector3 normal, out Vector3 point)
        {
            normal = Vector3.zero;
            point = samples[seed].World;
            Sample a = samples[seed];
            if (!a.NormalValid) return false;
            int x = seed % nx, y = seed / nx;
            for (int ix = 0; ix < LocalOffsets.Length; ix++)
            {
                int bx = x + LocalOffsets[ix];
                if (bx < 0 || bx >= nx) continue;
                Sample b = samples[y * nx + bx];
                if (!LocallyCompatible(a, b)) continue;
                for (int iy = 0; iy < LocalOffsets.Length; iy++)
                {
                    int cy = y + LocalOffsets[iy];
                    if (cy < 0 || cy >= ny) continue;
                    Sample c = samples[cy * nx + x];
                    if (!LocallyCompatible(a, c)) continue;
                    Vector3 cross = Vector3.Cross(b.World - a.World,
                        c.World - a.World);
                    if (cross.sqrMagnitude < 0.0004f) continue;
                    Vector3 candidate = cross.normalized;
                    if (Mathf.Abs(Vector3.Dot(candidate, a.Normal)) < 0.82f)
                        continue;
                    normal = candidate;
                    return true;
                }
            }
            return false;
        }

        private static bool LocallyCompatible(Sample a, Sample b) =>
            b.Valid && b.NormalValid &&
            Mathf.Abs(Vector3.Dot(a.Normal, b.Normal)) >= 0.82f &&
            Mathf.Abs(a.Distance - b.Distance) <=
                Mathf.Max(0.15f, a.Distance * 0.10f);

        private static bool HasAreaFootprint(List<int> group, int nx, int ny, int step)
        {
            int minX = nx, minY = ny, maxX = 0, maxY = 0;
            for (int i = 0; i < group.Count; i++)
            {
                int index = group[i];
                int x = index % nx, y = index / nx;
                minX = Mathf.Min(minX, x); maxX = Mathf.Max(maxX, x);
                minY = Mathf.Min(minY, y); maxY = Mathf.Max(maxY, y);
            }
            return (maxX - minX) * step >= 0.08f * nx * step &&
                   (maxY - minY) * step >= 0.08f * ny * step;
        }

        private static Vector3 MeanPoint(Sample[] samples, List<int> members)
        {
            Vector3 mean = Vector3.zero;
            for (int i = 0; i < members.Count; i++)
                mean += samples[members[i]].World;
            return mean / members.Count;
        }

        private static float ResidualP95(Sample[] samples, List<int> members,
            Vector3 point, Vector3 normal)
        {
            var residuals = new float[members.Count];
            for (int i = 0; i < members.Count; i++)
                residuals[i] = Mathf.Abs(Vector3.Dot(
                    samples[members[i]].World - point, normal));
            Array.Sort(residuals);
            return residuals[Mathf.Clamp(Mathf.CeilToInt(residuals.Length * 0.95f) - 1,
                0, residuals.Length - 1)];
        }

        private static int ClosestMember(Sample[] samples, List<int> members,
            Vector3 point, Vector3 normal)
        {
            int best = -1;
            float bestDistance = float.MaxValue;
            for (int i = 0; i < members.Count; i++)
            {
                int index = members[i];
                Sample sample = samples[index];
                if (!sample.NormalValid ||
                    Mathf.Abs(Vector3.Dot(sample.Normal, normal)) < 0.82f)
                    continue;
                float distance = Mathf.Abs(Vector3.Dot(
                    sample.World - point, normal));
                if (distance > FitTolerance(sample.Distance)) continue;
                if (distance >= bestDistance) continue;
                best = index;
                bestDistance = distance;
            }
            return best;
        }

        private static float FitTolerance(float distance) =>
            Mathf.Clamp(0.006f + distance * 0.003f, 0.008f, 0.022f);

        private string ComparisonStatus()
        {
            float absoluteDot = Mathf.Abs(Vector3.Dot(_currentPlaneNormal,
                _referencePlaneNormal));
            float angle = Mathf.Acos(Mathf.Clamp(absoluteDot, -1f, 1f)) *
                          Mathf.Rad2Deg;
            if (angle > 15f)
                return $" 青参照 方向差{angle:0}°(不同面勿比位差)";
            Vector3 tangent = Vector3.Cross(_referencePlaneNormal, Vector3.up);
            if (tangent.sqrMagnitude < 0.01f)
                tangent = Vector3.Cross(_referencePlaneNormal, Vector3.right);
            tangent.Normalize();
            Vector3 bitangent = Vector3.Cross(_referencePlaneNormal, tangent).normalized;
            ProjectPlaneFootprint(_referencePlaneMesh, _referencePlaneRenderer.transform,
                tangent, bitangent, out Vector2 oldMin, out Vector2 oldMax);
            ProjectPlaneFootprint(_planeMesh, _planeRenderer.transform,
                tangent, bitangent, out Vector2 newMin, out Vector2 newMax);
            float overlapU = Mathf.Min(oldMax.x, newMax.x) -
                             Mathf.Max(oldMin.x, newMin.x);
            float overlapV = Mathf.Min(oldMax.y, newMax.y) -
                             Mathf.Max(oldMin.y, newMin.y);
            if (overlapU < 0.10f || overlapV < 0.10f)
                return " 青参照 区域不重叠(不比位差)";
            float separation = Mathf.Abs(Vector3.Dot(
                _currentPlanePoint - _referencePlanePoint, _referencePlaneNormal));
            return $" 青→绿 方向{angle:0}° 位差{separation * 1000f:0}mm(平行候选)";
        }

        private static void ProjectPlaneFootprint(Mesh mesh, Transform owner,
            Vector3 tangent, Vector3 bitangent, out Vector2 min, out Vector2 max)
        {
            min = new Vector2(float.MaxValue, float.MaxValue);
            max = new Vector2(float.MinValue, float.MinValue);
            Vector3[] vertices = mesh.vertices;
            for (int i = 0; i < vertices.Length; i++)
            {
                Vector3 world = owner.TransformPoint(vertices[i]);
                float u = Vector3.Dot(world, tangent);
                float v = Vector3.Dot(world, bitangent);
                min.x = Mathf.Min(min.x, u); max.x = Mathf.Max(max.x, u);
                min.y = Mathf.Min(min.y, v); max.y = Mathf.Max(max.y, v);
            }
        }

        private static List<int> FloodConnectedFromSeed(bool[] eligible,
            int seed, int nx, int ny, bool[] acceptedFirst,
            bool[] acceptedSecond, out bool[] mask)
        {
            mask = new bool[eligible.Length];
            var group = new List<int>();
            if (seed < 0 || seed >= eligible.Length || !eligible[seed])
                return group;
            var seen = new bool[eligible.Length];
            var queue = new Queue<int>();
            queue.Enqueue(seed);
            seen[seed] = true;
            while (queue.Count > 0)
            {
                int current = queue.Dequeue();
                group.Add(current);
                mask[current] = true;
                int x = current % nx, y = current / nx;
                if (x > 0) TryEnqueue(current - 1);
                if (x + 1 < nx) TryEnqueue(current + 1);
                if (y > 0) TryEnqueue(current - nx);
                if (y + 1 < ny) TryEnqueue(current + nx);

                void TryEnqueue(int next)
                {
                    if (seen[next] || !eligible[next]) return;
                    if (!ShellEdgeConnected(current, next, nx, ny,
                            acceptedFirst, acceptedSecond)) return;
                    seen[next] = true;
                    queue.Enqueue(next);
                }
            }
            return group;
        }

        private static Vector3 SmallestCovarianceAxis(Sample[] samples,
            List<int> members, Vector3 centre)
        {
            var a = new float[3, 3];
            var v = new float[3, 3];
            for (int i = 0; i < 3; i++) v[i, i] = 1f;
            for (int i = 0; i < members.Count; i++)
            {
                Vector3 d = samples[members[i]].World - centre;
                a[0, 0] += d.x * d.x; a[0, 1] += d.x * d.y; a[0, 2] += d.x * d.z;
                a[1, 1] += d.y * d.y; a[1, 2] += d.y * d.z; a[2, 2] += d.z * d.z;
            }
            a[1, 0] = a[0, 1]; a[2, 0] = a[0, 2]; a[2, 1] = a[1, 2];
            for (int sweep = 0; sweep < 20; sweep++)
            {
                int p = 0, q = 1;
                float max = Mathf.Abs(a[0, 1]);
                if (Mathf.Abs(a[0, 2]) > max) { p = 0; q = 2; max = Mathf.Abs(a[0, 2]); }
                if (Mathf.Abs(a[1, 2]) > max) { p = 1; q = 2; max = Mathf.Abs(a[1, 2]); }
                if (max < 1e-7f) break;
                float angle = 0.5f * Mathf.Atan2(2f * a[p, q], a[q, q] - a[p, p]);
                float c = Mathf.Cos(angle), s = Mathf.Sin(angle);
                for (int k = 0; k < 3; k++)
                {
                    float akp = a[k, p], akq = a[k, q];
                    a[k, p] = c * akp - s * akq;
                    a[k, q] = s * akp + c * akq;
                    float vkp = v[k, p], vkq = v[k, q];
                    v[k, p] = c * vkp - s * vkq;
                    v[k, q] = s * vkp + c * vkq;
                }
                for (int k = 0; k < 3; k++)
                {
                    float apk = a[p, k], aqk = a[q, k];
                    a[p, k] = c * apk - s * aqk;
                    a[q, k] = s * apk + c * aqk;
                }
            }
            int smallest = a[1, 1] < a[0, 0] ? 1 : 0;
            if (a[2, 2] < a[smallest, smallest]) smallest = 2;
            return new Vector3(v[0, smallest], v[1, smallest], v[2, smallest]).normalized;
        }

        private static bool ShellEdgeConnected(int current, int next, int nx,
            int ny, bool[] first, bool[] second)
        {
            int x = current % nx, y = current / nx;
            if (next == current + 1)
                return (y < ny - 1 && first[y * (nx - 1) + x]) ||
                       (y > 0 && second[(y - 1) * (nx - 1) + x]);
            if (next == current - 1)
                return (y < ny - 1 && first[y * (nx - 1) + x - 1]) ||
                       (y > 0 && second[(y - 1) * (nx - 1) + x - 1]);
            if (next == current + nx)
                return (x < nx - 1 && first[y * (nx - 1) + x]) ||
                       (x > 0 && second[y * (nx - 1) + x - 1]);
            if (next == current - nx)
                return (x < nx - 1 && first[(y - 1) * (nx - 1) + x]) ||
                       (x > 0 && second[(y - 1) * (nx - 1) + x - 1]);
            return false;
        }

        // 0=accepted, 1=missing, 2=depth-only reject, 3=normal-only
        // reject, 4=both. These are the shell's source triangle rules.
        private static int ClassifyShellTriangle(Sample a, Sample b, Sample c)
        {
            if (!a.Valid || !b.Valid || !c.Valid || !a.NormalValid ||
                !b.NormalValid || !c.NormalValid)
                return 1;
            float near = Mathf.Min(a.LinearDepth,
                Mathf.Min(b.LinearDepth, c.LinearDepth));
            float far = Mathf.Max(a.LinearDepth,
                Mathf.Max(b.LinearDepth, c.LinearDepth));
            bool depthRejected = far - near > Mathf.Max(0.030f, near * 0.025f);
            float normalCoherence = Mathf.Min(Vector3.Dot(a.Normal, b.Normal),
                Mathf.Min(Vector3.Dot(b.Normal, c.Normal),
                    Vector3.Dot(c.Normal, a.Normal)));
            bool normalRejected = normalCoherence <= 0.35f;
            if (depthRejected && normalRejected) return 4;
            if (depthRejected) return 2;
            return normalRejected ? 3 : 0;
        }

        private static bool CanRescueCoplanarCell(Sample a, Sample b,
            Sample c, Sample d)
        {
            if (!a.Valid || !b.Valid || !c.Valid || !d.Valid ||
                !a.NormalValid || !b.NormalValid || !c.NormalValid ||
                !d.NormalValid)
                return false;
            float normalCoherence = Mathf.Min(
                Mathf.Min(Vector3.Dot(a.Normal, b.Normal),
                    Vector3.Dot(a.Normal, c.Normal)),
                Mathf.Min(Vector3.Dot(b.Normal, d.Normal),
                    Vector3.Dot(c.Normal, d.Normal)));
            Vector3 normalSum = a.Normal + b.Normal + c.Normal + d.Normal;
            if (normalCoherence < 0.70f || normalSum.magnitude < 0.5f)
                return false;
            Vector3 normal = normalSum.normalized;
            Vector3 geometric = Vector3.Cross(b.World - a.World,
                    c.World - a.World) +
                Vector3.Cross(d.World - b.World, c.World - b.World);
            if (geometric.sqrMagnitude < 1e-12f ||
                Mathf.Abs(Vector3.Dot(normal, geometric.normalized)) < 0.75f)
                return false;
            Vector3 centre = (a.World + b.World + c.World + d.World) * 0.25f;
            float residual = Mathf.Max(
                Mathf.Max(Mathf.Abs(Vector3.Dot(a.World - centre, normal)),
                    Mathf.Abs(Vector3.Dot(b.World - centre, normal))),
                Mathf.Max(Mathf.Abs(Vector3.Dot(c.World - centre, normal)),
                    Mathf.Abs(Vector3.Dot(d.World - centre, normal))));
            float centreDepth = (a.LinearDepth + b.LinearDepth +
                                 c.LinearDepth + d.LinearDepth) * 0.25f;
            return residual <= Mathf.Clamp(0.004f + centreDepth * 0.003f,
                0.006f, 0.015f);
        }

        private static bool HasDrawableTriangle(bool[] mask, int nx, int ny,
            bool[] first, bool[] second)
        {
            for (int y = 0; y < ny - 1; y++)
            for (int x = 0; x < nx - 1; x++)
            {
                int a = y * nx + x, b = a + 1, c = a + nx, d = c + 1;
                int cell = y * (nx - 1) + x;
                if (first[cell] && mask[a] && mask[b] && mask[c])
                    return true;
                if (second[cell] && mask[b] && mask[d] && mask[c])
                    return true;
            }
            return false;
        }

        private static bool[] FindPatchInteriors(int[] patchIds, int nx, int ny)
        {
            var interior = new bool[patchIds.Length];
            for (int y = 1; y < ny - 1; y++)
            for (int x = 1; x < nx - 1; x++)
            {
                int index = y * nx + x;
                int patch = patchIds[index];
                if (patch < 0) continue;
                bool surrounded = true;
                for (int dy = -1; dy <= 1 && surrounded; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    if (patchIds[index + dy * nx + dx] != patch)
                    {
                        surrounded = false;
                        break;
                    }
                interior[index] = surrounded;
            }
            return interior;
        }

        private static bool AddTriangle(int a, int b, int c, bool accepted,
            Sample[] samples,
            int[] patchIds, bool[] interior, List<PlanePatch> patches,
            List<Vector3> raw, List<Vector3> plane)
        {
            if (!accepted)
                return false;

            int patch = patchIds[a];
            bool onPlane = patch >= 0 && patchIds[b] == patch && patchIds[c] == patch;
            List<Vector3> target = onPlane ? plane : raw;
            Vector3 pa = samples[a].World, pb = samples[b].World, pc = samples[c].World;
            if (onPlane)
            {
                PlanePatch fitted = patches[patch];
                if (interior[a]) pa -= Vector3.Dot(pa - fitted.Point, fitted.Normal) * fitted.Normal;
                if (interior[b]) pb -= Vector3.Dot(pb - fitted.Point, fitted.Normal) * fitted.Normal;
                if (interior[c]) pc -= Vector3.Dot(pc - fitted.Point, fitted.Normal) * fitted.Normal;
            }
            target.Add(pa); target.Add(pb);
            target.Add(pb); target.Add(pc);
            target.Add(pc); target.Add(pa);
            return true;
        }

        private void ReplaceMesh(ref Mesh mesh, MeshRenderer renderer,
            List<Vector3> worldLines, string name)
        {
            if (renderer == null) return;
            MeshFilter filter = renderer.GetComponent<MeshFilter>();
            if (mesh != null) Destroy(mesh);
            mesh = new Mesh { name = name, indexFormat = IndexFormat.UInt32 };
            var vertices = new Vector3[worldLines.Count];
            var indices = new int[worldLines.Count];
            for (int i = 0; i < vertices.Length; i++)
            {
                vertices[i] = renderer.transform.InverseTransformPoint(worldLines[i]);
                indices[i] = i;
            }
            mesh.vertices = vertices;
            mesh.SetIndices(indices, MeshTopology.Lines, 0, true);
            filter.sharedMesh = mesh;
        }

        private void ClearMeshes()
        {
            if (_planeRenderer != null) _planeRenderer.GetComponent<MeshFilter>().sharedMesh = null;
            if (_referencePlaneRenderer != null)
                _referencePlaneRenderer.GetComponent<MeshFilter>().sharedMesh = null;
            if (_otherRenderer != null) _otherRenderer.GetComponent<MeshFilter>().sharedMesh = null;
            if (_planeMesh != null) Destroy(_planeMesh);
            if (_referencePlaneMesh != null) Destroy(_referencePlaneMesh);
            if (_otherMesh != null) Destroy(_otherMesh);
            _planeMesh = null;
            _referencePlaneMesh = null;
            _otherMesh = null;
            _currentPlaneValid = false;
            _referencePlaneValid = false;
            _currentPatchCount = 0;
            _referencePatchCount = 0;
            UpdateVisibility();
        }

        private void UpdateVisibility()
        {
            if (_planeRenderer != null)
                _planeRenderer.enabled = _visible && _planeMesh != null &&
                                         _planeMesh.vertexCount > 0;
            if (_referencePlaneRenderer != null)
                _referencePlaneRenderer.enabled = _visible &&
                    _referencePlaneMesh != null && _referencePlaneMesh.vertexCount > 0;
            if (_otherRenderer != null)
                _otherRenderer.enabled = _visible && _otherMesh != null &&
                                         _otherMesh.vertexCount > 0;
        }

        private static bool IsFinite(float value) =>
            !float.IsNaN(value) && !float.IsInfinity(value);

        private void OnDestroy()
        {
            _generation++;
            if (_sourceCopy != null) { _sourceCopy.Release(); Destroy(_sourceCopy); }
            if (_planeMesh != null) Destroy(_planeMesh);
            if (_referencePlaneMesh != null) Destroy(_referencePlaneMesh);
            if (_otherMesh != null) Destroy(_otherMesh);
            if (_planeMaterial != null) Destroy(_planeMaterial);
            if (_referencePlaneMaterial != null) Destroy(_referencePlaneMaterial);
            if (_otherMaterial != null) Destroy(_otherMaterial);
        }
    }
}
