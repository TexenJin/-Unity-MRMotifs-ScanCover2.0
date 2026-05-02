using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
public sealed class ScanCoverFusedPointCloudDisplayRenderer : MonoBehaviour
{
    private const int MaxInstancesPerDraw = 1023;

    private struct DisplayPointState
    {
        public Vector3 worldPos;
        public float confidence;
        public int batchIndex;
    }

    private struct DisplayGridCoord
    {
        public float u;
        public float v;
        public float w;
    }

    private struct DisplayBucketKey
    {
        public int patch;
        public int u;
        public int v;
        public int w;
    }

    private struct DisplayPatchKey
    {
        public int nx;
        public int ny;
        public int nz;
        public int w;
    }

    private struct DisplayPatchState
    {
        public Vector3 normal;
        public Vector3 centroid;
        public float planeDepth;
        public Vector3 tangent;
        public Vector3 bitangent;
        public float totalWeight;
        public int sampleCount;
    }

    private struct DisplayPatchFamilyState
    {
        public Vector3 normal;
        public Vector3 centroid;
        public float planeDepth;
        public Vector3 tangent;
        public Vector3 bitangent;
        public float totalWeight;
        public int sampleCount;
    }

    private sealed class RenderChunk
    {
        public GameObject gameObject;
        public MeshFilter meshFilter;
        public MeshRenderer meshRenderer;
        public Mesh mesh;
    }

    private const int MaxPointsPerChunk = 2048;

    [Header("Refs")]
    public ScanCoverFusedPointCloudManager fusedPointCloudManager;

    [Header("Display")]
    public bool showPointCloud = true;
    public PrimitiveType pointPrimitive = PrimitiveType.Quad;
    public bool billboardPointQuads = true;
    public bool scalePointByConfidence = true;
    public bool regularizeDisplayPoints = true;
    [Min(0.001f)] public float displayPointSpacingMeters = 0.0095f;
    [Min(0.001f)] public float displayPatchPlaneBandMeters = 0.06f;
    [Min(1f)] public float displayPatchNormalQuantizationSteps = 2f;
    [Range(-1f, 1f)] public float displayPatchMergeNormalDot = 0.965f;
    [Min(0.001f)] public float displayPatchMergeDepthMeters = 0.10f;
    [Min(0.001f)] public float pointScaleMeters = 0.0022f;
    [Min(0.001f)] public float minPointScaleMeters = 0.0016f;
    [Min(0.001f)] public float maxPointScaleMeters = 0.0030f;
    public bool colorizePointsByBatch = true;
    public Material pointMaterialOverride;
    public Color pointColor = new Color(0.95f, 0.95f, 0.97f, 1f);

    [Header("Debug")]
    public bool debugLog;

    public int LastDisplayPointCount { get; private set; }

    private readonly List<ScanCoverFusedPointCloudManager.FusedClusterState> _clusters = new List<ScanCoverFusedPointCloudManager.FusedClusterState>(4096);
    private readonly List<RenderChunk> _renderChunks = new List<RenderChunk>(16);
    private Material _runtimePointMaterial;
    private GameObject _root;
    private int _lastRenderedRevision = -1;
    private bool _loggedMaterialFailure;

    private void Awake()
    {
        ResolveRefs();
        EnsureObjects();
    }

    private void OnEnable()
    {
        ResolveRefs();
        EnsureObjects();
        RebuildNow(force: true);
    }

    private void LateUpdate()
    {
        if (fusedPointCloudManager == null)
            return;

        if (_lastRenderedRevision != fusedPointCloudManager.StateRevision)
            RebuildNow();
    }

    private void OnDestroy()
    {
        ClearChunks();
        if (_runtimePointMaterial != null)
            Destroy(_runtimePointMaterial);
        if (_root != null)
            Destroy(_root);
    }

    public void RebuildNow(bool force = false)
    {
        ResolveRefs();
        EnsureObjects();
        if (fusedPointCloudManager == null)
            return;
        if (!force && _lastRenderedRevision == fusedPointCloudManager.StateRevision)
            return;

        _lastRenderedRevision = fusedPointCloudManager.StateRevision;
        fusedPointCloudManager.GetClustersNonAlloc(_clusters);
        RebuildChunks();
    }

    private void RebuildChunks()
    {
        ClearChunks();
        LastDisplayPointCount = 0;
        if (!showPointCloud || _clusters.Count <= 0)
        {
            if (_root != null)
                _root.SetActive(false);
            return;
        }

        Material material = ResolvePointMaterial();
        if (material == null)
            return;

        List<DisplayPointState> displayPoints = BuildDisplayPoints();
        displayPoints.Sort((a, b) => a.batchIndex.CompareTo(b.batchIndex));
        LastDisplayPointCount = displayPoints.Count;
        if (debugLog)
            Debug.Log($"[ScanCoverFusedPointCloudDisplayRenderer] displayPoints={LastDisplayPointCount}, fusedClusters={_clusters.Count}");

        if (displayPoints.Count <= 0)
        {
            if (_root != null)
                _root.SetActive(false);
            return;
        }

        Quaternion billboardRotation = Quaternion.identity;
        if (billboardPointQuads && Camera.main != null)
            billboardRotation = Camera.main.transform.rotation;

        int start = 0;
        while (start < displayPoints.Count)
        {
            int batchIndex = displayPoints[start].batchIndex;
            int count = 0;
            while (start + count < displayPoints.Count &&
                   count < MaxPointsPerChunk &&
                   displayPoints[start + count].batchIndex == batchIndex)
            {
                count++;
            }

            Color batchColor = colorizePointsByBatch ? EvaluateBatchColor(batchIndex) : pointColor;
            RenderChunk chunk = CreateChunk(displayPoints, start, count, batchColor, billboardRotation);
            if (chunk != null)
                _renderChunks.Add(chunk);
            start += count;
        }

        if (_root != null)
            _root.SetActive(true);
    }

    private List<DisplayPointState> BuildDisplayPoints()
    {
        if (!regularizeDisplayPoints || _clusters.Count <= 1)
        {
            List<DisplayPointState> passthrough = new List<DisplayPointState>(_clusters.Count);
            for (int i = 0; i < _clusters.Count; i++)
            {
                ScanCoverFusedPointCloudManager.FusedClusterState cluster = _clusters[i];
                passthrough.Add(new DisplayPointState
                {
                    worldPos = cluster.worldPos,
                    confidence = cluster.confidence,
                    batchIndex = cluster.batchIndex
                });
            }

            return passthrough;
        }

        float spacing = Mathf.Max(0.001f, displayPointSpacingMeters);
        float normalBand = Mathf.Max(0.001f, displayPatchPlaneBandMeters);
        float normalQuantizationSteps = Mathf.Max(1f, displayPatchNormalQuantizationSteps);
        Dictionary<DisplayPatchKey, int> patchKeyToIndex = new Dictionary<DisplayPatchKey, int>(_clusters.Count);
        List<DisplayPatchState> patches = new List<DisplayPatchState>(_clusters.Count);
        int[] clusterToPatch = new int[_clusters.Count];

        for (int i = 0; i < _clusters.Count; i++)
        {
            ScanCoverFusedPointCloudManager.FusedClusterState cluster = _clusters[i];
            Vector3 patchNormal = ResolveDisplayPatchNormal(cluster.normal, normalQuantizationSteps);
            DisplayPatchKey patchKey = BuildDisplayPatchKey(cluster, patchNormal, normalBand);
            float clusterWeight = ScanCoverFusedPointCloudManager.ResolveClusterScore(cluster);
            float clusterPlaneDepth = Vector3.Dot(cluster.worldPos, patchNormal);

            if (!patchKeyToIndex.TryGetValue(patchKey, out int patchIndex))
            {
                patchIndex = patches.Count;
                patchKeyToIndex.Add(patchKey, patchIndex);
                BuildStablePatchBasis(patchNormal, out Vector3 tangent, out Vector3 bitangent);
                patches.Add(new DisplayPatchState
                {
                    normal = patchNormal,
                    centroid = cluster.worldPos,
                    planeDepth = clusterPlaneDepth,
                    tangent = tangent,
                    bitangent = bitangent,
                    totalWeight = clusterWeight,
                    sampleCount = 1
                });
            }
            else
            {
                DisplayPatchState patch = patches[patchIndex];
                float totalWeight = patch.totalWeight + clusterWeight;
                if (totalWeight <= 1e-6f)
                    totalWeight = 1f;

                Vector3 blendedNormal = patch.normal * patch.totalWeight + patchNormal * clusterWeight;
                patch.normal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : patch.normal;
                patch.centroid = (patch.centroid * patch.totalWeight + cluster.worldPos * clusterWeight) / totalWeight;
                patch.planeDepth = (patch.planeDepth * patch.totalWeight + clusterPlaneDepth * clusterWeight) / totalWeight;
                BuildStablePatchBasis(patch.normal, out patch.tangent, out patch.bitangent);
                patch.totalWeight = totalWeight;
                patch.sampleCount += 1;
                patches[patchIndex] = patch;
            }

            clusterToPatch[i] = patchIndex;
        }

        Dictionary<DisplayBucketKey, int> bucketToIndex = new Dictionary<DisplayBucketKey, int>(_clusters.Count);
        List<DisplayPointState> regularized = new List<DisplayPointState>(_clusters.Count);
        List<float> bucketWeights = new List<float>(_clusters.Count);
        List<int> bucketPatchIndices = new List<int>(_clusters.Count);
        List<DisplayBucketKey> bucketKeys = new List<DisplayBucketKey>(_clusters.Count);
        int[] patchToFamily = BuildDisplayPatchFamilies(patches);

        for (int i = 0; i < _clusters.Count; i++)
        {
            ScanCoverFusedPointCloudManager.FusedClusterState cluster = _clusters[i];
            int familyIndex = patchToFamily[clusterToPatch[i]];
            DisplayPatchFamilyState family = BuildDisplayPatchFamilyState(familyIndex, patches, patchToFamily);
            DisplayGridCoord coord = ResolveDisplayGridCoord(family, cluster.worldPos, spacing, normalBand);
            DisplayBucketKey bucket = new DisplayBucketKey
            {
                patch = familyIndex,
                u = Mathf.RoundToInt(coord.u),
                v = Mathf.RoundToInt(coord.v),
                w = Mathf.RoundToInt(coord.w)
            };

            AccumulateDisplayBucket(
                familyIndex,
                family,
                bucket,
                ScanCoverFusedPointCloudManager.ResolveClusterScore(cluster),
                cluster,
                spacing,
                normalBand,
                bucketToIndex,
                regularized,
                bucketWeights,
                bucketPatchIndices,
                bucketKeys,
                patches,
                patchToFamily);
        }

        return regularized;
    }

    private static DisplayPatchKey BuildDisplayPatchKey(ScanCoverFusedPointCloudManager.FusedClusterState cluster, Vector3 patchNormal, float normalBand)
    {
        float planeCoord = Vector3.Dot(cluster.worldPos, patchNormal) / normalBand;
        return new DisplayPatchKey
        {
            nx = Mathf.RoundToInt(patchNormal.x * 100f),
            ny = Mathf.RoundToInt(patchNormal.y * 100f),
            nz = Mathf.RoundToInt(patchNormal.z * 100f),
            w = Mathf.RoundToInt(planeCoord)
        };
    }

    private static Vector3 ResolveDisplayPatchNormal(Vector3 normal, float quantizationSteps)
    {
        Vector3 n = normal.sqrMagnitude > 1e-6f ? normal.normalized : Vector3.up;
        float step = 1f / Mathf.Max(1f, quantizationSteps);
        Vector3 snapped = new Vector3(
            Mathf.Round(n.x / step) * step,
            Mathf.Round(n.y / step) * step,
            Mathf.Round(n.z / step) * step);
        return snapped.sqrMagnitude <= 1e-6f ? n : snapped.normalized;
    }

    private static void BuildStablePatchBasis(Vector3 patchNormal, out Vector3 tangent, out Vector3 bitangent)
    {
        Vector3 n = patchNormal.sqrMagnitude > 1e-6f ? patchNormal.normalized : Vector3.up;
        float upDot = Mathf.Abs(Vector3.Dot(n, Vector3.up));
        if (upDot < 0.35f)
        {
            bitangent = Vector3.up;
            tangent = Vector3.Cross(bitangent, n).normalized;
            if (NeedsAxisFlip(tangent))
                tangent = -tangent;
            bitangent = Vector3.Cross(n, tangent).normalized;
            return;
        }

        if (upDot > 0.85f)
        {
            tangent = Vector3.right - n * Vector3.Dot(Vector3.right, n);
            if (tangent.sqrMagnitude <= 1e-6f)
                tangent = Vector3.forward - n * Vector3.Dot(Vector3.forward, n);
            tangent.Normalize();
            if (NeedsAxisFlip(tangent))
                tangent = -tangent;
            bitangent = Vector3.Cross(n, tangent).normalized;
            if (NeedsAxisFlip(bitangent))
                bitangent = -bitangent;
            return;
        }

        Vector3 tangentSeed = ResolveStableTangentSeed(n);
        tangent = tangentSeed - n * Vector3.Dot(tangentSeed, n);
        if (tangent.sqrMagnitude <= 1e-6f)
            tangent = Vector3.right - n * Vector3.Dot(Vector3.right, n);
        tangent.Normalize();
        if (NeedsAxisFlip(tangent))
            tangent = -tangent;
        bitangent = Vector3.Cross(n, tangent).normalized;
        if (NeedsAxisFlip(bitangent))
        {
            bitangent = -bitangent;
            tangent = Vector3.Cross(bitangent, n).normalized;
        }
    }

    private static DisplayGridCoord ResolveDisplayGridCoord(DisplayPatchFamilyState patch, Vector3 position, float spacing, float normalBand)
    {
        return new DisplayGridCoord
        {
            u = Vector3.Dot(position, patch.tangent) / spacing,
            v = Vector3.Dot(position, patch.bitangent) / spacing,
            w = (Vector3.Dot(position, patch.normal) - patch.planeDepth) / normalBand
        };
    }

    private static Vector3 ResolveDisplayPointPosition(DisplayPatchFamilyState patch, DisplayBucketKey bucket, float spacing, float normalBand)
    {
        return patch.tangent * (bucket.u * spacing)
            + patch.bitangent * (bucket.v * spacing)
            + patch.normal * (patch.planeDepth + bucket.w * normalBand);
    }

    private static void AccumulateDisplayBucket(
        int familyIndex,
        DisplayPatchFamilyState family,
        DisplayBucketKey bucket,
        float clusterWeight,
        ScanCoverFusedPointCloudManager.FusedClusterState cluster,
        float spacing,
        float normalBand,
        Dictionary<DisplayBucketKey, int> bucketToIndex,
        List<DisplayPointState> regularized,
        List<float> bucketWeights,
        List<int> bucketPatchIndices,
        List<DisplayBucketKey> bucketKeys,
        List<DisplayPatchState> patches,
        int[] patchToFamily)
    {
        if (clusterWeight <= 1e-5f)
            return;

        if (bucketToIndex.TryGetValue(bucket, out int displayIndex))
        {
            float accumulatedWeight = bucketWeights[displayIndex];
            float totalWeight = accumulatedWeight + clusterWeight;
            if (totalWeight <= 1e-6f)
                totalWeight = 1f;

            DisplayPointState merged = regularized[displayIndex];
            merged.worldPos = ResolveDisplayPointPosition(BuildDisplayPatchFamilyState(bucketPatchIndices[displayIndex], patches, patchToFamily), bucketKeys[displayIndex], spacing, normalBand);
            merged.confidence = Mathf.Clamp01((merged.confidence * accumulatedWeight + cluster.confidence * clusterWeight) / totalWeight);
            if (clusterWeight >= accumulatedWeight * 0.75f)
                merged.batchIndex = cluster.batchIndex;
            regularized[displayIndex] = merged;
            bucketWeights[displayIndex] = totalWeight;
            return;
        }

        bucketToIndex.Add(bucket, regularized.Count);
        bucketPatchIndices.Add(familyIndex);
        bucketKeys.Add(bucket);
        regularized.Add(new DisplayPointState
        {
            worldPos = ResolveDisplayPointPosition(family, bucket, spacing, normalBand),
            confidence = cluster.confidence,
            batchIndex = cluster.batchIndex
        });
        bucketWeights.Add(clusterWeight);
    }

    private int[] BuildDisplayPatchFamilies(List<DisplayPatchState> patches)
    {
        int[] patchToFamily = new int[patches.Count];
        List<DisplayPatchFamilyState> families = new List<DisplayPatchFamilyState>(patches.Count);
        float mergeNormalDot = Mathf.Clamp(displayPatchMergeNormalDot, -1f, 1f);
        float mergeDepthMeters = Mathf.Max(0.001f, displayPatchMergeDepthMeters);

        for (int i = 0; i < patches.Count; i++)
        {
            DisplayPatchState patch = patches[i];
            int bestFamily = -1;
            float bestScore = float.NegativeInfinity;

            for (int familyIndex = 0; familyIndex < families.Count; familyIndex++)
            {
                DisplayPatchFamilyState family = families[familyIndex];
                float normalDot = Vector3.Dot(family.normal, patch.normal);
                if (normalDot < mergeNormalDot)
                    continue;

                float depthDelta = Mathf.Abs(family.planeDepth - patch.planeDepth);
                if (depthDelta > mergeDepthMeters)
                    continue;

                float score = normalDot - depthDelta;
                if (score > bestScore)
                {
                    bestScore = score;
                    bestFamily = familyIndex;
                }
            }

            if (bestFamily < 0)
            {
                bestFamily = families.Count;
                families.Add(new DisplayPatchFamilyState
                {
                    normal = patch.normal,
                    centroid = patch.centroid,
                    planeDepth = patch.planeDepth,
                    tangent = patch.tangent,
                    bitangent = patch.bitangent,
                    totalWeight = patch.totalWeight,
                    sampleCount = patch.sampleCount
                });
            }
            else
            {
                DisplayPatchFamilyState family = families[bestFamily];
                float totalWeight = family.totalWeight + patch.totalWeight;
                if (totalWeight <= 1e-6f)
                    totalWeight = 1f;

                Vector3 blendedNormal = family.normal * family.totalWeight + patch.normal * patch.totalWeight;
                family.normal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : family.normal;
                family.centroid = (family.centroid * family.totalWeight + patch.centroid * patch.totalWeight) / totalWeight;
                family.planeDepth = (family.planeDepth * family.totalWeight + patch.planeDepth * patch.totalWeight) / totalWeight;
                BuildStablePatchBasis(family.normal, out family.tangent, out family.bitangent);
                family.totalWeight = totalWeight;
                family.sampleCount += patch.sampleCount;
                families[bestFamily] = family;
            }

            patchToFamily[i] = bestFamily;
        }

        return patchToFamily;
    }

    private static DisplayPatchFamilyState BuildDisplayPatchFamilyState(int familyIndex, List<DisplayPatchState> patches, int[] patchToFamily)
    {
        DisplayPatchFamilyState family = default;
        bool initialized = false;
        for (int i = 0; i < patches.Count; i++)
        {
            if (patchToFamily[i] != familyIndex)
                continue;

            DisplayPatchState patch = patches[i];
            if (!initialized)
            {
                family = new DisplayPatchFamilyState
                {
                    normal = patch.normal,
                    centroid = patch.centroid,
                    planeDepth = patch.planeDepth,
                    tangent = patch.tangent,
                    bitangent = patch.bitangent,
                    totalWeight = patch.totalWeight,
                    sampleCount = patch.sampleCount
                };
                initialized = true;
                continue;
            }

            float totalWeight = family.totalWeight + patch.totalWeight;
            if (totalWeight <= 1e-6f)
                totalWeight = 1f;

            Vector3 blendedNormal = family.normal * family.totalWeight + patch.normal * patch.totalWeight;
            family.normal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : family.normal;
            family.centroid = (family.centroid * family.totalWeight + patch.centroid * patch.totalWeight) / totalWeight;
            family.planeDepth = (family.planeDepth * family.totalWeight + patch.planeDepth * patch.totalWeight) / totalWeight;
            BuildStablePatchBasis(family.normal, out family.tangent, out family.bitangent);
            family.totalWeight = totalWeight;
            family.sampleCount += patch.sampleCount;
        }

        return family;
    }

    private static Vector3 ResolveStableTangentSeed(Vector3 normal)
    {
        Vector3[] candidates = { Vector3.right, Vector3.up, Vector3.forward };
        Vector3 best = Vector3.right;
        float bestScore = float.PositiveInfinity;
        for (int i = 0; i < candidates.Length; i++)
        {
            float score = Mathf.Abs(Vector3.Dot(normal, candidates[i]));
            if (score < bestScore)
            {
                bestScore = score;
                best = candidates[i];
            }
        }

        return best;
    }

    private static bool NeedsAxisFlip(Vector3 axis)
    {
        Vector3 abs = new Vector3(Mathf.Abs(axis.x), Mathf.Abs(axis.y), Mathf.Abs(axis.z));
        if (abs.x >= abs.y && abs.x >= abs.z)
            return axis.x < 0f;
        if (abs.y >= abs.x && abs.y >= abs.z)
            return axis.y < 0f;
        return axis.z < 0f;
    }

    private void ResolveRefs()
    {
        if (fusedPointCloudManager == null)
            fusedPointCloudManager = GetComponent<ScanCoverFusedPointCloudManager>();
    }

    private void EnsureObjects()
    {
        if (_root == null)
        {
            _root = new GameObject("[ScanCover] Fused Point Cloud");
            _root.transform.SetParent(null, false);
        }

        if (pointMaterialOverride != null)
        {
            if (_runtimePointMaterial != null)
            {
                Destroy(_runtimePointMaterial);
                _runtimePointMaterial = null;
            }
        }
        else if (_runtimePointMaterial == null)
        {
            _runtimePointMaterial = CreateRuntimePointMaterial();
        }

        Material material = ResolvePointMaterial();
        if (material != null)
        {
            material.enableInstancing = true;
            if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", pointColor);
            if (material.HasProperty("_Color")) material.SetColor("_Color", pointColor);
            if (material.HasProperty("_Cull")) material.SetFloat("_Cull", (float)CullMode.Off);
            if (material.HasProperty("_Glossiness")) material.SetFloat("_Glossiness", 0f);
            if (material.HasProperty("_Smoothness")) material.SetFloat("_Smoothness", 0f);
            if (material.HasProperty("_ZWrite")) material.SetFloat("_ZWrite", 0f);
            if (material.HasProperty("_Surface")) material.SetFloat("_Surface", 1f);
            material.renderQueue = (int)RenderQueue.Transparent;
        }
        else if (!_loggedMaterialFailure)
        {
            Debug.LogError("[ScanCoverFusedPointCloudDisplayRenderer] Failed to create point material. Assign pointMaterialOverride explicitly.");
            _loggedMaterialFailure = true;
        }

    }

    private Material ResolvePointMaterial()
    {
        if (pointMaterialOverride == null && _runtimePointMaterial == null)
            _runtimePointMaterial = CreateRuntimePointMaterial();
        return pointMaterialOverride != null ? pointMaterialOverride : _runtimePointMaterial;
    }

    private Mesh BuildPrimitiveMesh(PrimitiveType primitiveType)
    {
        if (primitiveType == PrimitiveType.Quad)
            return BuildQuadMesh();

        GameObject temp = GameObject.CreatePrimitive(primitiveType);
        Mesh sourceMesh = temp.GetComponent<MeshFilter>() != null ? temp.GetComponent<MeshFilter>().sharedMesh : null;
        Mesh instanceMesh = sourceMesh != null ? Instantiate(sourceMesh) : null;
        Destroy(temp);
        if (instanceMesh != null)
            instanceMesh.name = $"ScanCover_Fpc_{primitiveType}_Mesh";
        return instanceMesh;
    }

    private Material CreateRuntimePointMaterial()
    {
        string[] shaderNames =
        {
            "Universal Render Pipeline/Unlit",
            "Unlit/Color",
            "Sprites/Default",
            "Standard"
        };

        Shader pointShader = null;
        for (int i = 0; i < shaderNames.Length; i++)
        {
            pointShader = Shader.Find(shaderNames[i]);
            if (pointShader != null)
                break;
        }

        if (pointShader == null)
            return null;

        Material material = new Material(pointShader) { name = "ScanCover_FusedPointCloud_PointMat" };
        _loggedMaterialFailure = false;
        return material;
    }

    private static Mesh BuildQuadMesh()
    {
        Mesh mesh = new Mesh
        {
            name = "ScanCover_Fpc_Quad_Mesh"
        };

        mesh.vertices = new[]
        {
            new Vector3(-0.5f, -0.5f, 0f),
            new Vector3(0.5f, -0.5f, 0f),
            new Vector3(0.5f, 0.5f, 0f),
            new Vector3(-0.5f, 0.5f, 0f)
        };
        mesh.uv = new[]
        {
            new Vector2(0f, 0f),
            new Vector2(1f, 0f),
            new Vector2(1f, 1f),
            new Vector2(0f, 1f)
        };
        mesh.triangles = new[] { 0, 1, 2, 0, 2, 3 };
        mesh.normals = new[] { Vector3.forward, Vector3.forward, Vector3.forward, Vector3.forward };
        mesh.RecalculateBounds();
        return mesh;
    }

    private RenderChunk CreateChunk(List<DisplayPointState> displayPoints, int start, int count, Color color, Quaternion billboardRotation)
    {
        if (count <= 0)
            return null;

        Vector3[] vertices = new Vector3[count * 4];
        Vector2[] uvs = new Vector2[count * 4];
        int[] triangles = new int[count * 6];
        Vector3[] normals = new Vector3[count * 4];
        Vector3 right = billboardRotation * Vector3.right;
        Vector3 up = billboardRotation * Vector3.up;

        for (int i = 0; i < count; i++)
        {
            DisplayPointState pointState = displayPoints[start + i];
            float displayScale = Mathf.Max(0.001f, pointScaleMeters);
            if (scalePointByConfidence)
            {
                float confidence01 = Mathf.Clamp01(pointState.confidence);
                displayScale = Mathf.Lerp(Mathf.Max(0.001f, minPointScaleMeters), Mathf.Max(minPointScaleMeters, maxPointScaleMeters), confidence01);
            }

            float half = displayScale * 0.5f;
            Vector3 center = pointState.worldPos;
            Vector3 vx = right * half;
            Vector3 vy = up * half;
            int vi = i * 4;
            int ti = i * 6;

            vertices[vi + 0] = center - vx - vy;
            vertices[vi + 1] = center + vx - vy;
            vertices[vi + 2] = center + vx + vy;
            vertices[vi + 3] = center - vx + vy;

            uvs[vi + 0] = new Vector2(0f, 0f);
            uvs[vi + 1] = new Vector2(1f, 0f);
            uvs[vi + 2] = new Vector2(1f, 1f);
            uvs[vi + 3] = new Vector2(0f, 1f);

            normals[vi + 0] = -Vector3.forward;
            normals[vi + 1] = -Vector3.forward;
            normals[vi + 2] = -Vector3.forward;
            normals[vi + 3] = -Vector3.forward;

            triangles[ti + 0] = vi + 0;
            triangles[ti + 1] = vi + 1;
            triangles[ti + 2] = vi + 2;
            triangles[ti + 3] = vi + 0;
            triangles[ti + 4] = vi + 2;
            triangles[ti + 5] = vi + 3;
        }

        GameObject chunkObject = new GameObject($"FpcChunk_{_renderChunks.Count:000}");
        chunkObject.transform.SetParent(_root.transform, false);
        chunkObject.layer = gameObject.layer;
        MeshFilter meshFilter = chunkObject.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = chunkObject.AddComponent<MeshRenderer>();
        meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
        meshRenderer.receiveShadows = false;
        meshRenderer.lightProbeUsage = LightProbeUsage.Off;
        meshRenderer.reflectionProbeUsage = ReflectionProbeUsage.Off;
        meshRenderer.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;
        meshRenderer.sharedMaterial = ResolvePointMaterial();
        MaterialPropertyBlock propertyBlock = new MaterialPropertyBlock();
        propertyBlock.SetColor("_BaseColor", color);
        propertyBlock.SetColor("_Color", color);
        meshRenderer.SetPropertyBlock(propertyBlock);

        Mesh mesh = new Mesh
        {
            name = $"ScanCover_Fpc_Chunk_{_renderChunks.Count:000}"
        };
        if (vertices.Length > 60000)
            mesh.indexFormat = IndexFormat.UInt32;
        mesh.vertices = vertices;
        mesh.uv = uvs;
        mesh.normals = normals;
        mesh.triangles = triangles;
        mesh.RecalculateBounds();
        meshFilter.sharedMesh = mesh;

        return new RenderChunk
        {
            gameObject = chunkObject,
            meshFilter = meshFilter,
            meshRenderer = meshRenderer,
            mesh = mesh
        };
    }

    private void ClearChunks()
    {
        for (int i = 0; i < _renderChunks.Count; i++)
        {
            RenderChunk chunk = _renderChunks[i];
            if (chunk == null)
                continue;
            if (chunk.mesh != null)
                Destroy(chunk.mesh);
            if (chunk.gameObject != null)
                Destroy(chunk.gameObject);
        }
        _renderChunks.Clear();
        if (_root != null)
            _root.SetActive(false);
    }

    private Color EvaluateBatchColor(int batchIndex)
    {
        if (batchIndex <= 0)
            return pointColor;

        float hue = Mathf.Repeat(0.13f + batchIndex * 0.19f, 1f);
        Color color = Color.HSVToRGB(hue, 0.65f, 1f);
        color.a = pointColor.a;
        return color;
    }
}
