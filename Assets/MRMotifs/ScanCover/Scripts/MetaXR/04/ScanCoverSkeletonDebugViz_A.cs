using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class ScanCoverSkeletonDebugViz_A : MonoBehaviour
{
    [Header("Refs")]
    public ScanCoverSkeletonBuilder_A builder;
    public Transform vizRoot;

    [Header("Viz")]
    public bool visible = false;
    [Min(0.05f)] public float updateIntervalSeconds = 0.25f;
    [Min(1)] public int maxVisibleCells = 4000;
    [Min(16)] public int poolSize = 5000;
    [Range(0.1f, 1.0f)] public float cubeScale = 0.9f;
    public bool modulateByCount = true;

    private readonly List<ScanCoverSkeletonBuilder_A.CellInfo> _snapshot = new List<ScanCoverSkeletonBuilder_A.CellInfo>(8192);

    private struct CubeSlot
    {
        public bool used;
        public ScanCoverSkeletonBuilder_A.VoxelKey key;
        public Transform tr;
        public Renderer renderer;
    }

    private readonly Dictionary<ScanCoverSkeletonBuilder_A.VoxelKey, int> _map = new Dictionary<ScanCoverSkeletonBuilder_A.VoxelKey, int>(8192);
    private CubeSlot[] _pool;
    private int _poolCursor;
    private float _nextUpdate;
    private MaterialPropertyBlock _mpb;
    private static readonly int BaseColorId = Shader.PropertyToID("_BaseColor");
    private static readonly int ColorId = Shader.PropertyToID("_Color");

    private void Awake()
    {
        if (!builder) builder = GetComponent<ScanCoverSkeletonBuilder_A>();
        if (!vizRoot) vizRoot = transform;
        _mpb = new MaterialPropertyBlock();
        BuildPool();
    }

    private void Update()
    {
        if (!builder) return;
        if (!visible)
        {
            SetAllActive(false);
            return;
        }
        if (Time.time < _nextUpdate) return;
        _nextUpdate = Time.time + Mathf.Max(0.05f, updateIntervalSeconds);

        builder.GetCellsSnapshot(_snapshot);
        int n = Mathf.Min(_snapshot.Count, maxVisibleCells);
        for (int i = 0; i < n; i++)
        {
            var c = _snapshot[i];
            int slot = GetOrAllocSlot(c.key);
            if (slot < 0) break;
            Transform tr = _pool[slot].tr;
            tr.position = c.worldPos;
            float s = builder.cellSizeMeters * cubeScale;
            tr.localScale = new Vector3(s, s, s);

            Renderer r = _pool[slot].renderer;
            if (r != null)
            {
                float k = modulateByCount ? Mathf.Clamp01((c.count - 1) / 8.0f) : 0.6f;
                Color col = Color.Lerp(new Color(1f, 0.25f, 0.2f, 1f), new Color(0.2f, 1f, 1f, 1f), k);
                _mpb.SetColor(BaseColorId, col);
                _mpb.SetColor(ColorId, col);
                r.SetPropertyBlock(_mpb);
            }

            _pool[slot].used = true;
            if (!_pool[slot].tr.gameObject.activeSelf)
                _pool[slot].tr.gameObject.SetActive(true);
        }

        for (int i = 0; i < _pool.Length; i++)
        {
            if (_pool[i].used) _pool[i].used = false;
            else if (_pool[i].tr != null && _pool[i].tr.gameObject.activeSelf)
                _pool[i].tr.gameObject.SetActive(false);
        }
    }

    private void BuildPool()
    {
        int n = Mathf.Max(16, poolSize);
        _pool = new CubeSlot[n];
        for (int i = 0; i < n; i++)
        {
            GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = $"Vox_{i:0000}";
            go.transform.SetParent(vizRoot, true);
            Collider c = go.GetComponent<Collider>();
            if (c) Destroy(c);
            Renderer r = go.GetComponent<Renderer>();
            if (r != null && r.sharedMaterial == null)
                r.sharedMaterial = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
            go.SetActive(false);
            _pool[i] = new CubeSlot { tr = go.transform, renderer = r };
        }
    }

    private int GetOrAllocSlot(ScanCoverSkeletonBuilder_A.VoxelKey key)
    {
        if (_map.TryGetValue(key, out int idx)) return idx;
        for (int guard = 0; guard < _pool.Length; guard++)
        {
            int i = (_poolCursor + guard) % _pool.Length;
            if (_map.ContainsKey(_pool[i].key)) _map.Remove(_pool[i].key);
            _pool[i].key = key;
            _map[key] = i;
            _poolCursor = (i + 1) % _pool.Length;
            return i;
        }
        return -1;
    }

    private void SetAllActive(bool on)
    {
        if (_pool == null) return;
        for (int i = 0; i < _pool.Length; i++)
            if (_pool[i].tr != null && _pool[i].tr.gameObject.activeSelf != on)
                _pool[i].tr.gameObject.SetActive(on);
    }
}
