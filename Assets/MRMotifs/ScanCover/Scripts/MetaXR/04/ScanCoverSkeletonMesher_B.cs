using System;
using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class ScanCoverSkeletonMesher_B : MonoBehaviour
{
    [Serializable]
    public struct ChunkKey : IEquatable<ChunkKey>
    {
        public int x, y, z;
        public ChunkKey(int x, int y, int z) { this.x = x; this.y = y; this.z = z; }
        public bool Equals(ChunkKey other) => x == other.x && y == other.y && z == other.z;
        public override bool Equals(object obj) => obj is ChunkKey other && Equals(other);
        public override int GetHashCode()
        {
            unchecked
            {
                int h = 17;
                h = h * 31 + x;
                h = h * 31 + y;
                h = h * 31 + z;
                return h;
            }
        }
        public override string ToString() => $"({x},{y},{z})";
    }

    private sealed class Chunk
    {
        public GameObject go;
        public Mesh mesh;
        public MeshFilter mf;
        public MeshRenderer mr;
        public MeshCollider mc;
    }

    [Header("Refs")]
    public ScanCoverSkeletonBuilder_A builder;
    public Transform referenceFrame;
    public Transform chunksRoot;

    [Header("Meshing")]
    [Min(2)] public int chunkSizeVoxels = 8;
    [Min(1)] public int minHitsToConfirm = 2;
    public bool buildCollider = true;
    public bool renderChunks = true;
    public Material chunkMaterial;

    [Header("Hole Fill (Voxel)")]
    [Tooltip("对确认体素做小洞补全（按体素邻域规则）")]
    public bool enableHoleFill = true;
    [Tooltip("仅在冻结构建时补洞，避免扫描阶段频繁重算")]
    public bool holeFillOnlyWhenFrozen = true;
    [Min(1)] public int holeFillIterations = 1;
    [Range(0, 6)] public int holeFillMinSolidNeighbors = 5;
    [Tooltip("要求至少存在一组轴向对夹（x+/x- 或 y+/y- 或 z+/z-）")]
    public bool holeFillRequireAxisPair = true;
    [Min(0)] public int holeFillMaxAddedVoxelsPerChunk = 512;

    [Header("Live Build (optional)")]
    [Tooltip("默认关闭。仅当你明确希望冻结前也用 chunk 做后台更新时才开启。")]
    public bool liveBuild = false;
    [Min(0.1f)] public float liveRebuildIntervalSeconds = 1.0f;

    [Header("Render")]
    public bool rendererVisibleAfterBuild = true;
    public bool hideRendererNearCamera = false;
    [Min(0f)] public float hideRenderNearCameraMeters = 0.60f;
    public Camera renderCamera;

    [Header("Debug")]
    public bool debugLog = false;

    private readonly Dictionary<ChunkKey, Chunk> _chunks = new Dictionary<ChunkKey, Chunk>(256);
    private readonly List<ScanCoverSkeletonBuilder_A.CellInfo> _snapshot = new List<ScanCoverSkeletonBuilder_A.CellInfo>(16384);
    private readonly Dictionary<ChunkKey, List<ScanCoverSkeletonBuilder_A.VoxelKey>> _voxelsByChunk = new Dictionary<ChunkKey, List<ScanCoverSkeletonBuilder_A.VoxelKey>>(256);
    private readonly HashSet<ScanCoverSkeletonBuilder_A.VoxelKey> _confirmed = new HashSet<ScanCoverSkeletonBuilder_A.VoxelKey>();
    private readonly List<ChunkKey> _chunkOrder = new List<ChunkKey>(256);
    private readonly List<ScanCoverSkeletonBuilder_A.VoxelKey> _holeFillScratch = new List<ScanCoverSkeletonBuilder_A.VoxelKey>(1024);
    private float _nextLiveRebuild;

    public int ChunkCount => _chunks.Count;

    private void Awake()
    {
        if (!builder) builder = GetComponent<ScanCoverSkeletonBuilder_A>();
        if (!referenceFrame && builder) referenceFrame = builder.referenceFrame;
        if (!referenceFrame) referenceFrame = transform;
        if (IsLegacyVisualChainAllowed())
            EnsureChunksRoot();
        if (!renderCamera && builder && builder.sampleCamera) renderCamera = builder.sampleCamera;
        if (!renderCamera && Camera.main) renderCamera = Camera.main;

        if (!chunkMaterial)
        {
            Shader sh = Shader.Find("Universal Render Pipeline/Lit");
            if (!sh) sh = Shader.Find("Universal Render Pipeline/Unlit");
            if (sh) chunkMaterial = new Material(sh);
        }
    }

    private void Update()
    {
        if (!IsLegacyVisualChainAllowed())
        {
            DisableLegacyRuntimeOutput();
            return;
        }

        if (renderChunks && _chunks.Count > 0)
            RefreshVisibility();

        if (!liveBuild || !builder || !builder.scanEnabled)
            return;
        if (Time.time < _nextLiveRebuild)
            return;
        _nextLiveRebuild = Time.time + Mathf.Max(0.1f, liveRebuildIntervalSeconds);
        BuildAllNow();
    }

    private void EnsureChunksRoot()
    {
        if (!IsLegacyVisualChainAllowed())
            return;
        if (chunksRoot) return;
        Transform existing = referenceFrame ? referenceFrame.Find("[ScanCover] FrozenSkeletonChunks") : null;
        if (existing) chunksRoot = existing;
        else
        {
            GameObject go = new GameObject("[ScanCover] FrozenSkeletonChunks");
            go.transform.SetParent(referenceFrame ? referenceFrame : transform, false);
            chunksRoot = go.transform;
        }
    }

    public void ClearAllChunks()
    {
        foreach (var kv in _chunks)
        {
            Chunk c = kv.Value;
            if (c == null) continue;
            if (c.mesh != null)
            {
                if (Application.isPlaying) Destroy(c.mesh);
                else DestroyImmediate(c.mesh);
            }
            if (c.go == null) continue;
            if (Application.isPlaying) Destroy(c.go);
            else DestroyImmediate(c.go);
        }
        _chunks.Clear();
    }

    public void BuildAllNow()
    {
        if (!IsLegacyVisualChainAllowed())
        {
            DisableLegacyRuntimeOutput();
            return;
        }
        if (!builder) return;
        _snapshot.Clear();
        builder.GetCellsSnapshot(_snapshot);
        BuildCommittedChunksFromSnapshot(_snapshot, null, false);
    }

    public void BuildOrUpdateCommittedChunksFromSnapshot(List<ScanCoverSkeletonBuilder_A.CellInfo> snapshot, ICollection<ChunkKey> keys)
    {
        if (!IsLegacyVisualChainAllowed())
        {
            DisableLegacyRuntimeOutput();
            return;
        }
        BuildCommittedChunksFromSnapshot(snapshot, keys, false);
    }

    public void SetRendererVisible(bool visible)
    {
        if (!IsLegacyVisualChainAllowed())
            visible = false;
        rendererVisibleAfterBuild = visible;
        RefreshVisibility();
    }

    public void GetChunkStats(out int total, out int rendered, out int colliders)
    {
        total = rendered = colliders = 0;
        foreach (var kv in _chunks)
        {
            Chunk c = kv.Value;
            if (c == null || c.go == null) continue;
            total++;
            if (c.mr && c.mr.enabled) rendered++;
            if (c.mc && c.mc.enabled) colliders++;
        }
    }

    public static ChunkKey VoxelToChunkKey(ScanCoverSkeletonBuilder_A.VoxelKey vk, int chunkSize)
    {
        return new ChunkKey(FloorDiv(vk.x, chunkSize), FloorDiv(vk.y, chunkSize), FloorDiv(vk.z, chunkSize));
    }

    public ChunkKey WorldToChunkKey(Vector3 worldPos)
    {
        if (!referenceFrame) referenceFrame = builder && builder.referenceFrame ? builder.referenceFrame : transform;
        float s = builder ? Mathf.Max(1e-4f, builder.cellSizeMeters) : 0.08f;
        int cs = Mathf.Max(2, chunkSizeVoxels);
        Vector3 pRef = referenceFrame.InverseTransformPoint(worldPos);
        return new ChunkKey(
            FloorDiv(Mathf.FloorToInt(pRef.x / s), cs),
            FloorDiv(Mathf.FloorToInt(pRef.y / s), cs),
            FloorDiv(Mathf.FloorToInt(pRef.z / s), cs));
    }

    private void BuildCommittedChunksFromSnapshot(List<ScanCoverSkeletonBuilder_A.CellInfo> snapshot, ICollection<ChunkKey> keys, bool clearPrevious)
    {
        if (!IsLegacyVisualChainAllowed())
        {
            DisableLegacyRuntimeOutput();
            return;
        }
        if (!builder) return;
        if (!referenceFrame) referenceFrame = builder.referenceFrame ? builder.referenceFrame : transform;
        EnsureChunksRoot();

        if (clearPrevious)
            ClearAllChunks();

        _confirmed.Clear();
        _voxelsByChunk.Clear();
        _chunkOrder.Clear();

        float s = Mathf.Max(1e-4f, builder.cellSizeMeters);
        int cs = Mathf.Max(2, chunkSizeVoxels);
        bool filterKeys = keys != null;
        HashSet<ChunkKey> keySet = null;
        if (filterKeys)
            keySet = keys as HashSet<ChunkKey> ?? new HashSet<ChunkKey>(keys);

        for (int i = 0; i < snapshot.Count; i++)
        {
            var c = snapshot[i];
            if (c.count < minHitsToConfirm) continue;
            _confirmed.Add(c.key);
            ChunkKey ck = VoxelToChunkKey(c.key, cs);
            if (filterKeys && !keySet.Contains(ck))
                continue;
            if (!_voxelsByChunk.TryGetValue(ck, out var list))
            {
                list = new List<ScanCoverSkeletonBuilder_A.VoxelKey>(256);
                _voxelsByChunk.Add(ck, list);
                _chunkOrder.Add(ck);
            }
            list.Add(c.key);
        }

        for (int i = 0; i < _chunkOrder.Count; i++)
        {
            ChunkKey ck = _chunkOrder[i];
            if (_voxelsByChunk.TryGetValue(ck, out var voxels) && voxels != null && voxels.Count > 0)
            {
                if (ShouldApplyHoleFill())
                    ApplyHoleFillToChunk(ck, voxels, cs);
                BuildOrUpdateChunk(ck, voxels, cs, s);
            }
        }

        if (debugLog)
        {
            int keyCount = (keys != null) ? keys.Count : _chunkOrder.Count;
            Debug.Log($"[ScanCoverSkeletonMesher_B] BuildCommittedChunks keys={keyCount} built={_chunkOrder.Count} confirmed={_confirmed.Count}");
        }
    }

    private void BuildOrUpdateChunk(ChunkKey ck, List<ScanCoverSkeletonBuilder_A.VoxelKey> voxels, int chunkSize, float voxelSize)
    {
        Chunk chunk = GetOrCreateChunk(ck);

        Vector3 originRef = ChunkOriginRef(ck, chunkSize, voxelSize);
        chunk.go.transform.SetParent(chunksRoot, false);
        chunk.go.transform.localPosition = originRef;
        chunk.go.transform.localRotation = Quaternion.identity;
        chunk.go.transform.localScale = Vector3.one;

        var verts = new List<Vector3>(Mathf.Max(24, voxels.Count * 24));
        var norms = new List<Vector3>(Mathf.Max(24, voxels.Count * 24));
        var tris = new List<int>(Mathf.Max(36, voxels.Count * 36));

        for (int i = 0; i < voxels.Count; i++)
            AddVoxelFaces(voxels[i], voxelSize, originRef, verts, norms, tris);

        Mesh mesh = chunk.mesh;
        if (mesh == null)
        {
            mesh = new Mesh();
            mesh.name = $"Chunk_{ck.x}_{ck.y}_{ck.z}";
            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            chunk.mesh = mesh;
        }
        else
        {
            mesh.Clear();
        }

        mesh.SetVertices(verts);
        mesh.SetNormals(norms);
        mesh.SetTriangles(tris, 0);
        mesh.RecalculateBounds();
        chunk.mf.sharedMesh = mesh;

        if (chunk.mc)
        {
            chunk.mc.sharedMesh = null;
            chunk.mc.sharedMesh = mesh;
            chunk.mc.enabled = buildCollider;
        }

        if (chunk.mr)
        {
            if (chunkMaterial && chunk.mr.sharedMaterial != chunkMaterial)
                chunk.mr.sharedMaterial = chunkMaterial;
            chunk.mr.enabled = renderChunks && rendererVisibleAfterBuild && !ShouldHideNear(chunk.go.transform.position);
        }
    }

    private Chunk GetOrCreateChunk(ChunkKey ck)
    {
        if (_chunks.TryGetValue(ck, out var existing) && existing != null && existing.go != null)
            return existing;

        GameObject go = new GameObject($"Chunk_{ck.x}_{ck.y}_{ck.z}");
        go.transform.SetParent(chunksRoot, false);
        MeshFilter mf = go.AddComponent<MeshFilter>();
        MeshRenderer mr = go.AddComponent<MeshRenderer>();
        if (chunkMaterial)
            mr.sharedMaterial = chunkMaterial;
        mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        mr.receiveShadows = false;
        MeshCollider mc = null;
        if (buildCollider)
        {
            mc = go.AddComponent<MeshCollider>();
            mc.convex = false;
        }

        var chunk = new Chunk { go = go, mf = mf, mr = mr, mc = mc };
        _chunks[ck] = chunk;
        return chunk;
    }

    private void RefreshVisibility()
    {
        foreach (var kv in _chunks)
        {
            Chunk c = kv.Value;
            if (c?.mr == null || c.go == null) continue;
            c.mr.enabled = renderChunks && rendererVisibleAfterBuild && !ShouldHideNear(c.go.transform.position);
            if (c.mc) c.mc.enabled = buildCollider;
        }
    }

    private bool ShouldHideNear(Vector3 chunkWorldOrigin)
    {
        if (!hideRendererNearCamera || hideRenderNearCameraMeters <= 0f || !renderCamera)
            return false;
        return (chunkWorldOrigin - renderCamera.transform.position).sqrMagnitude <= hideRenderNearCameraMeters * hideRenderNearCameraMeters;
    }

    private void AddVoxelFaces(ScanCoverSkeletonBuilder_A.VoxelKey vk, float s, Vector3 chunkOriginRef,
        List<Vector3> verts, List<Vector3> norms, List<int> tris)
    {
        float x0 = vk.x * s - chunkOriginRef.x;
        float y0 = vk.y * s - chunkOriginRef.y;
        float z0 = vk.z * s - chunkOriginRef.z;
        float x1 = x0 + s;
        float y1 = y0 + s;
        float z1 = z0 + s;

        if (!_confirmed.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(vk.x + 1, vk.y, vk.z)))
            AddQuad(new Vector3(x1, y0, z0), new Vector3(x1, y0, z1), new Vector3(x1, y1, z1), new Vector3(x1, y1, z0), Vector3.right, verts, norms, tris);
        if (!_confirmed.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(vk.x - 1, vk.y, vk.z)))
            AddQuad(new Vector3(x0, y0, z1), new Vector3(x0, y0, z0), new Vector3(x0, y1, z0), new Vector3(x0, y1, z1), Vector3.left, verts, norms, tris);
        if (!_confirmed.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(vk.x, vk.y + 1, vk.z)))
            AddQuad(new Vector3(x0, y1, z0), new Vector3(x1, y1, z0), new Vector3(x1, y1, z1), new Vector3(x0, y1, z1), Vector3.up, verts, norms, tris);
        if (!_confirmed.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(vk.x, vk.y - 1, vk.z)))
            AddQuad(new Vector3(x0, y0, z1), new Vector3(x1, y0, z1), new Vector3(x1, y0, z0), new Vector3(x0, y0, z0), Vector3.down, verts, norms, tris);
        if (!_confirmed.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(vk.x, vk.y, vk.z + 1)))
            AddQuad(new Vector3(x1, y0, z1), new Vector3(x0, y0, z1), new Vector3(x0, y1, z1), new Vector3(x1, y1, z1), Vector3.forward, verts, norms, tris);
        if (!_confirmed.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(vk.x, vk.y, vk.z - 1)))
            AddQuad(new Vector3(x0, y0, z0), new Vector3(x1, y0, z0), new Vector3(x1, y1, z0), new Vector3(x0, y1, z0), Vector3.back, verts, norms, tris);
    }

    private static void AddQuad(Vector3 a, Vector3 b, Vector3 c, Vector3 d, Vector3 n,
        List<Vector3> verts, List<Vector3> norms, List<int> tris)
    {
        int baseIdx = verts.Count;
        verts.Add(a); verts.Add(b); verts.Add(c); verts.Add(d);
        norms.Add(n); norms.Add(n); norms.Add(n); norms.Add(n);
        tris.Add(baseIdx + 0); tris.Add(baseIdx + 1); tris.Add(baseIdx + 2);
        tris.Add(baseIdx + 0); tris.Add(baseIdx + 2); tris.Add(baseIdx + 3);
    }

    private static int FloorDiv(int a, int b)
    {
        if (b <= 0) return 0;
        if (a >= 0) return a / b;
        return -(((-a) + b - 1) / b);
    }

    private static Vector3 ChunkOriginRef(ChunkKey ck, int chunkSize, float voxelSize)
    {
        return new Vector3(ck.x * chunkSize * voxelSize, ck.y * chunkSize * voxelSize, ck.z * chunkSize * voxelSize);
    }

    private bool ShouldApplyHoleFill()
    {
        if (!enableHoleFill)
            return false;
        if (!holeFillOnlyWhenFrozen)
            return true;
        return builder != null && builder.IsFrozen;
    }

    private void ApplyHoleFillToChunk(ChunkKey ck, List<ScanCoverSkeletonBuilder_A.VoxelKey> voxels, int chunkSize)
    {
        if (voxels == null || voxels.Count <= 0)
            return;

        int maxAdd = Mathf.Max(0, holeFillMaxAddedVoxelsPerChunk);
        if (maxAdd == 0)
            return;

        var localSolid = new HashSet<ScanCoverSkeletonBuilder_A.VoxelKey>(voxels);
        int startX = ck.x * chunkSize;
        int startY = ck.y * chunkSize;
        int startZ = ck.z * chunkSize;
        int endX = startX + chunkSize - 1;
        int endY = startY + chunkSize - 1;
        int endZ = startZ + chunkSize - 1;

        int addedTotal = 0;
        int iterations = Mathf.Max(1, holeFillIterations);
        int minNeighbors = Mathf.Clamp(holeFillMinSolidNeighbors, 0, 6);

        for (int iter = 0; iter < iterations && addedTotal < maxAdd; iter++)
        {
            _holeFillScratch.Clear();

            for (int x = startX; x <= endX; x++)
            {
                for (int y = startY; y <= endY; y++)
                {
                    for (int z = startZ; z <= endZ; z++)
                    {
                        var vk = new ScanCoverSkeletonBuilder_A.VoxelKey(x, y, z);
                        if (localSolid.Contains(vk) || _confirmed.Contains(vk))
                            continue;

                        bool xp = IsSolidVoxel(x + 1, y, z, localSolid);
                        bool xn = IsSolidVoxel(x - 1, y, z, localSolid);
                        bool yp = IsSolidVoxel(x, y + 1, z, localSolid);
                        bool yn = IsSolidVoxel(x, y - 1, z, localSolid);
                        bool zp = IsSolidVoxel(x, y, z + 1, localSolid);
                        bool zn = IsSolidVoxel(x, y, z - 1, localSolid);

                        int neighbors = 0;
                        if (xp) neighbors++;
                        if (xn) neighbors++;
                        if (yp) neighbors++;
                        if (yn) neighbors++;
                        if (zp) neighbors++;
                        if (zn) neighbors++;
                        if (neighbors < minNeighbors)
                            continue;

                        if (holeFillRequireAxisPair)
                        {
                            bool hasAxisPair = (xp && xn) || (yp && yn) || (zp && zn);
                            if (!hasAxisPair)
                                continue;
                        }

                        _holeFillScratch.Add(vk);
                        if (addedTotal + _holeFillScratch.Count >= maxAdd)
                            break;
                    }
                    if (addedTotal + _holeFillScratch.Count >= maxAdd)
                        break;
                }
                if (addedTotal + _holeFillScratch.Count >= maxAdd)
                    break;
            }

            if (_holeFillScratch.Count == 0)
                break;

            for (int i = 0; i < _holeFillScratch.Count; i++)
            {
                var addVk = _holeFillScratch[i];
                if (!localSolid.Add(addVk))
                    continue;
                if (_confirmed.Add(addVk))
                    voxels.Add(addVk);
                addedTotal++;
                if (addedTotal >= maxAdd)
                    break;
            }
        }

        if (debugLog && addedTotal > 0)
            Debug.Log($"[ScanCoverSkeletonMesher_B] HoleFill chunk={ck} +{addedTotal} voxels");
    }

    private bool IsSolidVoxel(int x, int y, int z, HashSet<ScanCoverSkeletonBuilder_A.VoxelKey> localSolid)
    {
        var vk = new ScanCoverSkeletonBuilder_A.VoxelKey(x, y, z);
        return localSolid.Contains(vk) || _confirmed.Contains(vk);
    }

    private bool IsLegacyVisualChainAllowed()
    {
        ScanCoverSkeletonSessionController sessionController = GetComponent<ScanCoverSkeletonSessionController>();
        return sessionController == null || sessionController.enableLegacyVisualChain;
    }

    private void DisableLegacyRuntimeOutput()
    {
        renderChunks = false;
        ClearAllChunks();
        if (chunksRoot != null)
        {
            if (Application.isPlaying) Destroy(chunksRoot.gameObject);
            else DestroyImmediate(chunksRoot.gameObject);
            chunksRoot = null;
        }
    }
}
