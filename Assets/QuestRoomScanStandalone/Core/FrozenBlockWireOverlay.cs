using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Read-only runtime view of the 32^3 fusion/freeze management grid.
    /// Geometry is rebuilt only on the supervisor census cadence; it never
    /// feeds TSDF integration, maturity, HERA routing, or lifecycle decisions.
    /// </summary>
    public sealed class FrozenBlockWireOverlay : MonoBehaviour
    {
        private static readonly int[] EdgePairs =
        {
            0, 1, 1, 2, 2, 3, 3, 0,
            4, 5, 5, 6, 6, 7, 7, 4,
            0, 4, 1, 5, 2, 6, 3, 7
        };

        private static readonly Color StableFrozen = new Color(0.10f, 0.78f, 1.00f, 0.68f);
        private static readonly Color HotOnce = new Color(1.00f, 0.78f, 0.08f, 0.82f);
        private static readonly Color HotConfirmed = new Color(1.00f, 0.12f, 0.04f, 1.00f);
        private static readonly Color Thawed = new Color(1.00f, 0.06f, 0.92f, 0.92f);

        private readonly List<Vector3> _vertices = new List<Vector3>(4096);
        private readonly List<Vector4> _otherAndSide = new List<Vector4>(4096);
        private readonly List<Color> _colors = new List<Color>(4096);
        private readonly List<int> _indices = new List<int>(6144);
        private readonly HashSet<int> _thawedScratch = new HashSet<int>();

        private Mesh _mesh;
        private Material _material;
        private MeshRenderer _renderer;
        private bool _visible = true;
        private bool _hasGeometry;

        public bool Visible => _visible;

        private void Awake()
        {
            BuildRenderer();
        }

        public void SetVisible(bool visible)
        {
            _visible = visible;
            if (_renderer != null) _renderer.enabled = visible && _hasGeometry;
        }

        public void Clear()
        {
            _vertices.Clear();
            _otherAndSide.Clear();
            _colors.Clear();
            _indices.Clear();
            _thawedScratch.Clear();
            _mesh?.Clear();
            _hasGeometry = false;
            if (_renderer != null) _renderer.enabled = false;
        }

        public void Rebuild(
            int3 voxelCount,
            float voxelSize,
            int3 grid,
            HashSet<int> frozen,
            HashSet<int> hot,
            HashSet<int> confirmedHot,
            IEnumerable<int> thawed)
        {
            if (_mesh == null || voxelSize <= 0f || math.any(grid <= 0)) return;

            _vertices.Clear();
            _otherAndSide.Clear();
            _colors.Clear();
            _indices.Clear();
            _thawedScratch.Clear();
            if (thawed != null) _thawedScratch.UnionWith(thawed);

            int total = grid.x * grid.y * grid.z;
            int blockX = Mathf.CeilToInt(voxelCount.x / (float)grid.x);
            int blockY = Mathf.CeilToInt(voxelCount.y / (float)grid.y);
            int blockZ = Mathf.CeilToInt(voxelCount.z / (float)grid.z);
            Vector3 volumeMin = -0.5f * voxelSize * new Vector3(voxelCount.x, voxelCount.y, voxelCount.z);

            for (int b = 0; b < total; b++)
            {
                Color color;
                if (_thawedScratch.Contains(b)) color = Thawed;
                else if (confirmedHot != null && confirmedHot.Contains(b)) color = HotConfirmed;
                else if (hot != null && hot.Contains(b)) color = HotOnce;
                else if (frozen != null && frozen.Contains(b)) color = StableFrozen;
                else continue;

                int bx = b % grid.x;
                int by = (b / grid.x) % grid.y;
                int bz = b / (grid.x * grid.y);
                int minX = bx * blockX;
                int minY = by * blockY;
                int minZ = bz * blockZ;
                int maxX = Mathf.Min(minX + blockX, voxelCount.x);
                int maxY = Mathf.Min(minY + blockY, voxelCount.y);
                int maxZ = Mathf.Min(minZ + blockZ, voxelCount.z);
                Vector3 min = volumeMin + voxelSize * new Vector3(minX, minY, minZ);
                Vector3 max = volumeMin + voxelSize * new Vector3(maxX, maxY, maxZ);
                AddCube(min, max, color);
            }

            _mesh.Clear();
            _mesh.SetVertices(_vertices);
            _mesh.SetUVs(0, _otherAndSide);
            _mesh.SetColors(_colors);
            _mesh.SetIndices(_indices, MeshTopology.Triangles, 0, true);
            _hasGeometry = _indices.Count > 0;
            if (_renderer != null) _renderer.enabled = _visible && _hasGeometry;
        }

        private void AddCube(Vector3 min, Vector3 max, Color color)
        {
            var corners = new Vector3[8];
            corners[0] = new Vector3(min.x, min.y, min.z);
            corners[1] = new Vector3(max.x, min.y, min.z);
            corners[2] = new Vector3(max.x, min.y, max.z);
            corners[3] = new Vector3(min.x, min.y, max.z);
            corners[4] = new Vector3(min.x, max.y, min.z);
            corners[5] = new Vector3(max.x, max.y, min.z);
            corners[6] = new Vector3(max.x, max.y, max.z);
            corners[7] = new Vector3(min.x, max.y, max.z);
            for (int i = 0; i < EdgePairs.Length; i += 2)
                AddScreenSpaceLine(corners[EdgePairs[i]], corners[EdgePairs[i + 1]], color);
        }

        /// <summary>
        /// Four vertices form a camera-facing strip.  The shader expands it in
        /// screen pixels using the same _RSWireThickness global as the scan mesh;
        /// unlike MeshTopology.Lines, Quest/Vulkan can therefore match its width.
        /// </summary>
        private void AddScreenSpaceLine(Vector3 start, Vector3 end, Color color)
        {
            int v = _vertices.Count;
            _vertices.Add(start);
            _vertices.Add(start);
            _vertices.Add(end);
            _vertices.Add(end);
            _otherAndSide.Add(new Vector4(end.x, end.y, end.z, -1f));
            _otherAndSide.Add(new Vector4(end.x, end.y, end.z, 1f));
            // End-point direction is reversed, so its side sign is reversed too.
            _otherAndSide.Add(new Vector4(start.x, start.y, start.z, 1f));
            _otherAndSide.Add(new Vector4(start.x, start.y, start.z, -1f));
            for (int i = 0; i < 4; i++) _colors.Add(color);
            _indices.Add(v);
            _indices.Add(v + 2);
            _indices.Add(v + 1);
            _indices.Add(v + 1);
            _indices.Add(v + 2);
            _indices.Add(v + 3);
        }

        private void BuildRenderer()
        {
            _mesh = new Mesh
            {
                name = "QRS Frozen Management Block Wires",
                indexFormat = IndexFormat.UInt32
            };
            _mesh.MarkDynamic();

            var child = new GameObject("[QRS] 32 Management Block Wires");
            child.layer = gameObject.layer;
            child.transform.SetParent(transform, false);
            child.AddComponent<MeshFilter>().sharedMesh = _mesh;
            _renderer = child.AddComponent<MeshRenderer>();
            _renderer.shadowCastingMode = ShadowCastingMode.Off;
            _renderer.receiveShadows = false;
            _renderer.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;

            Shader shader = Resources.Load<Shader>("ManagementBlockWire");
            if (shader == null)
            {
                Logger.Error("缺少 ManagementBlockWire shader；32管理块线框不可见");
                child.SetActive(false);
                return;
            }
            _material = new Material(shader) { name = "QRS 32 Management Block Wires" };
            _renderer.sharedMaterial = _material;
            _renderer.enabled = false;
        }

        private void OnDestroy()
        {
            if (_mesh != null) Destroy(_mesh);
            if (_material != null) Destroy(_material);
        }
    }
}
