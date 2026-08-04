using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Indirect renderer for the A-chain GPU surface nets mesh — faithful port
/// of QuestRoomScan's GPUMeshRenderer pattern: the mesh never leaves the
/// GPU, LateUpdate draws it via RenderPrimitivesIndirect straight from the
/// extraction buffers, so there is no per-frame CPU readback or Mesh upload.
/// Snapshot/export consumers keep using ScanCoverTsdfBranch.BuildSurfaceNow
/// (on-demand readback path) unchanged.
/// </summary>
[DisallowMultipleComponent]
public sealed class ScanCoverGpuMeshRenderer : MonoBehaviour
{
    [Tooltip("留空则自动查找同物体或场景中的 ScanCoverTsdfBranch")]
    public ScanCoverTsdfBranch branch;
    public bool renderVisible = true;
    public Color surfaceColor = new Color(0.18f, 0.58f, 1.0f, 0.85f);

    public string LastIssue { get; private set; }

    private Material _material;
    private MaterialPropertyBlock _propertyBlock;

    private static readonly int VerticesId = Shader.PropertyToID("_Vertices");
    private static readonly int IndicesId = Shader.PropertyToID("_Indices");
    private static readonly int VolumeLocalToWorldId = Shader.PropertyToID("_VolumeLocalToWorld");
    private static readonly int SurfaceColorId = Shader.PropertyToID("_SurfaceColor");

    private void Awake()
    {
        if (branch == null)
            branch = GetComponent<ScanCoverTsdfBranch>();
        if (branch == null)
            branch = FindFirstObjectByType<ScanCoverTsdfBranch>();
    }

    private void LateUpdate()
    {
        if (!renderVisible || branch == null)
            return;

        if (!branch.TryGetGpuRenderData(
                out GraphicsBuffer vertices,
                out GraphicsBuffer indices,
                out GraphicsBuffer drawArgs,
                out Matrix4x4 localToWorld))
            return;

        if (!EnsureMaterial())
            return;

        _propertyBlock ??= new MaterialPropertyBlock();
        _propertyBlock.SetBuffer(VerticesId, vertices);
        _propertyBlock.SetBuffer(IndicesId, indices);
        _propertyBlock.SetMatrix(VolumeLocalToWorldId, localToWorld);
        _propertyBlock.SetColor(SurfaceColorId, surfaceColor);

        RenderParams renderParams = new RenderParams(_material)
        {
            worldBounds = branch.GetGpuRenderWorldBounds(),
            matProps = _propertyBlock,
            shadowCastingMode = ShadowCastingMode.Off,
            receiveShadows = false,
            layer = gameObject.layer
        };
        Graphics.RenderPrimitivesIndirect(renderParams, MeshTopology.Triangles, drawArgs, 1);
    }

    private bool EnsureMaterial()
    {
        if (_material != null)
            return true;

        Shader shader = Resources.Load<Shader>("ScanCoverGpuSurfaceNetsTri");
        if (shader == null)
            shader = Shader.Find("ScanCover/GpuSurfaceNetsTri");
        if (shader == null)
        {
            LastIssue = "ScanCoverGpuSurfaceNetsTri shader 资源缺失";
            return false;
        }

        _material = new Material(shader);
        LastIssue = null;
        return true;
    }

    private void OnDestroy()
    {
        if (_material != null)
        {
            Destroy(_material);
            _material = null;
        }
    }
}
