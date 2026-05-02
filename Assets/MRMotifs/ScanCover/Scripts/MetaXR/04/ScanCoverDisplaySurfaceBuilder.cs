using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class ScanCoverDisplaySurfaceBuilder : MonoBehaviour
{
    public enum DisplayStyle
    {
        FacesOnly = 0,
        FacesAndWire = 1,
        WireOnly = 2
    }

    public enum CompareViewMode
    {
        Raw = 0,
        Smoothed = 1,
        Both = 2
    }

    [Header("Refs")]
    public ScanCoverSkeletonMesher_B mesher;
    public ScanCoverSkeletonSessionController sessionController;
    public Transform referenceFrame;
    public Transform sourceChunksRoot;
    public Transform displayRoot;
    public Transform viewerTransform;

    [Header("Display Mesh")]
    public Material displayMaterial;
    [Tooltip("Prefer the ScanCover observation-surface shader to draw a regular grid overlay on top of the smoothed mesh.")]
    public bool useObservationSurfaceShader = true;
    [Tooltip("Use a neutral URP/Unlit material by default to validate geometry without Lit shading bias.")]
    public bool useNeutralUnlitPreview = true;
    public Color neutralUnlitColor = new Color(0.78f, 0.82f, 0.86f, 1f);
    [Min(1000)] public int combineMaxVertices = 60000;
    public bool autoRebuildWhenFrozen = true;
    [Tooltip("Before frozen rebuild, force Mesher to refresh all chunks once.")]
    public bool refreshMesherBeforeFrozenRebuild = true;
    [Min(0.05f)] public float rebuildIntervalSeconds = 0.50f;
    public bool rebuildWhenChunkCountChanges = true;
    public bool buildOnEnable = false;

    [Header("Post Process")]
    public bool weldVertices = true;
    [Min(0f)] public float weldToleranceMeters = 0.001f;
    public bool recalculateNormals = false;
    public bool recalculateBounds = true;

    [Header("Visibility")]
    public bool hideSourceRenderersWhenBuilt = true;
    public bool visibleAfterBuild = true;

    [Header("Compare View")]
    [Tooltip("Build an additional smoothed preview mesh set for A/B validation.")]
    public bool buildSmoothedPreview = true;
    public CompareViewMode compareViewMode = CompareViewMode.Smoothed;
    [Tooltip("Build only the active compare mode during freeze to reduce stall time.")]
    public bool buildOnlyCurrentCompareMode = true;
    [Tooltip("When toggling compare mode, auto rebuild missing mesh set.")]
    public bool autoRebuildMissingModeOnSwitch = true;

    [Header("Visualization")]
    public DisplayStyle displayStyle = DisplayStyle.WireOnly;
    public Material wireMaterial;
    public Color wireColor = new Color(0.24f, 0.92f, 0.98f, 0f);

    [Header("Surface Grid Overlay")]
    public Color surfaceBaseColor = new Color(0.18f, 0.95f, 0.98f, 0f);
    public Color surfaceFresnelColor = new Color(0.95f, 1.0f, 1.0f, 0f);
    [Range(0f, 1f)] public float surfaceBaseAlpha = 0f;
    [Min(0.1f)] public float surfaceGridScale = 4.0f;
    [Range(0.001f, 0.2f)] public float surfaceGridThickness = 0.005f;
    [Range(0f, 3f)] public float surfaceGridIntensity = 1.45f;
    [Range(0.1f, 8f)] public float surfaceFresnelPower = 2.2f;
    [Range(0f, 3f)] public float surfaceFresnelStrength = 1.15f;

    [Header("Geometry Filters")]
    [Tooltip("Cull triangles too close to the viewer to avoid face-hugging overlays.")]
    public bool enableNearViewerCull = true;
    [Min(0.05f)] public float nearViewerCullMeters = 0.25f;
    [Tooltip("Cull small disconnected islands for cleaner display validation.")]
    public bool enableSmallIslandCull = true;
    [Min(1)] public int minIslandTriangles = 24;
    [Min(0f)] public float minIslandAreaM2 = 0.02f;

    [Header("Freeze Lock")]
    [Tooltip("Frozen閲嶅缓鍚庨攣瀹氬綋鍓岲isplay Mesh蹇収锛屽悗缁壂鎻忎笉瑕嗙洊")]
    public bool lockDisplayAfterFrozenBuild = true;
    [Tooltip("鎵弿闃舵缁х画鏄剧ず宸查攣瀹欴isplay")]
    public bool keepLockedDisplayWhileScanning = true;
    [Tooltip("Allow auto rebuild and refresh of locked snapshot while frozen.")]
    public bool allowLockedRefreshWhenFrozen = false;

    [Header("Locked Smoothing")]
    [Tooltip("Only smooth locked display meshes; do not write back to Skeleton/Mesher")]
    public bool smoothLockedDisplay = true;
    [Range(0, 8)] public int lockedSmoothIterations = 5;
    [Range(0.01f, 0.95f)] public float lockedSmoothLambda = 0.60f;
    [Range(-0.95f, -0.01f)] public float lockedSmoothMu = -0.62f;
    [Tooltip("Max displacement per vertex per pass (meters)")]
    [Min(0f)] public float lockedSmoothMaxStepMeters = 0.04f;
    [Tooltip("Preserve border vertices to avoid silhouette collapse.")]
    public bool preserveBoundaryVertices = false;
    [Range(0f, 1f)] public float boundarySmoothFactor = 0.45f;
    [Tooltip("Only use neighbors with similar normals (degrees).")]
    [Range(5f, 90f)] public float smoothNormalThresholdDeg = 80f;
    [Tooltip("Project displacement to tangent plane to reduce shrink and fake bulging.")]
    public bool tangentPlaneSmoothing = false;

    [Header("Debug")]
    public bool debugLog = false;

    private readonly List<GameObject> _builtObjects = new List<GameObject>(16);
    private readonly List<Mesh> _builtMeshes = new List<Mesh>(16);
    private readonly List<GameObject> _smoothedObjects = new List<GameObject>(16);
    private readonly List<Mesh> _smoothedMeshes = new List<Mesh>(16);
    private readonly List<GameObject> _rawWireObjects = new List<GameObject>(16);
    private readonly List<Mesh> _rawWireMeshes = new List<Mesh>(16);
    private readonly List<GameObject> _smoothedWireObjects = new List<GameObject>(16);
    private readonly List<Mesh> _smoothedWireMeshes = new List<Mesh>(16);
    private readonly List<Mesh> _lockedMeshes = new List<Mesh>(16);
    private float _nextRebuildTime;
    private int _lastChunkCount = -1;
    private bool _lastFrozen;

    public bool HasDisplay => _builtObjects.Count > 0 || _smoothedObjects.Count > 0;
    public bool HasLockedDisplay => _lockedMeshes.Count > 0;
    public int DisplayObjectCount => _builtObjects.Count + _smoothedObjects.Count;
    public int LastSourceChunkCount => _lastChunkCount;
    public CompareViewMode CurrentCompareViewMode => compareViewMode;

    private void Awake()
    {
        ResolveRefs();
        if (!IsLegacyVisualChainAllowed())
        {
            DisableLegacyOutput();
            return;
        }
        EnsureDisplayRoot();
        EnsureDisplayMaterial();
        EnsureWireMaterial();
    }

    private void OnEnable()
    {
        ResolveRefs();
        if (!IsLegacyVisualChainAllowed())
        {
            DisableLegacyOutput();
            return;
        }
        EnsureDisplayRoot();
        EnsureDisplayMaterial();
        EnsureWireMaterial();
        if (buildOnEnable)
            RebuildFromSource();
    }

    private void Update()
    {
        ResolveRefs();
        if (!IsLegacyVisualChainAllowed())
        {
            DisableLegacyOutput();
            return;
        }

        bool frozen = sessionController && sessionController.State == ScanCoverSkeletonSessionController.SessionState.Frozen;
        if (autoRebuildWhenFrozen && frozen)
        {
            if (lockDisplayAfterFrozenBuild && HasLockedDisplay && !allowLockedRefreshWhenFrozen)
            {
                if (!HasDisplay)
                    RebuildFromLocked();
                _lastFrozen = frozen;
                return;
            }

            bool needs = false;
            if (!_lastFrozen)
                needs = true;
            if (rebuildWhenChunkCountChanges && mesher && mesher.ChunkCount != _lastChunkCount)
                needs = true;
            if (needs && Time.time >= _nextRebuildTime)
            {
                RebuildFromSource();
                _nextRebuildTime = Time.time + Mathf.Max(0.05f, rebuildIntervalSeconds);
            }
        }
        else if (!frozen && lockDisplayAfterFrozenBuild && HasLockedDisplay)
        {
            if (keepLockedDisplayWhileScanning)
            {
                if (!HasDisplay)
                    RebuildFromLocked();
            }
            else
            {
                SetVisible(false);
            }
        }
        _lastFrozen = frozen;
    }

    private void ResolveRefs()
    {
        if (!mesher) mesher = GetComponent<ScanCoverSkeletonMesher_B>();
        if (!sessionController) sessionController = GetComponent<ScanCoverSkeletonSessionController>();
        if (!referenceFrame)
        {
            if (mesher && mesher.referenceFrame) referenceFrame = mesher.referenceFrame;
            else referenceFrame = transform;
        }
        if (!sourceChunksRoot && mesher) sourceChunksRoot = mesher.chunksRoot;
        if (!viewerTransform && Camera.main) viewerTransform = Camera.main.transform;
    }

    private void EnsureDisplayRoot()
    {
        if (!IsLegacyVisualChainAllowed())
            return;
        if (displayRoot) return;
        Transform parent = referenceFrame ? referenceFrame : transform;
        Transform existing = parent.Find("[ScanCover] DisplaySurface");
        if (existing) displayRoot = existing;
        else
        {
            GameObject go = new GameObject("[ScanCover] DisplaySurface");
            go.transform.SetParent(parent, false);
            displayRoot = go.transform;
        }
    }

    private void EnsureDisplayMaterial()
    {
        if (!IsLegacyVisualChainAllowed())
            return;
        if (!displayMaterial)
        {
            Shader sh = null;
            if (useObservationSurfaceShader)
                sh = Shader.Find("MRMotifs/ScanCover/ObservationSurface");
            if (!sh && useNeutralUnlitPreview)
                sh = Shader.Find("Universal Render Pipeline/Unlit");
            if (!sh) sh = Shader.Find("Universal Render Pipeline/Lit");
            if (!sh) sh = Shader.Find("Universal Render Pipeline/Unlit");
            if (!sh) return;

            displayMaterial = new Material(sh);
        }

        ApplyDisplayMaterialProperties();
    }

    private void ApplyDisplayMaterialProperties()
    {
        if (!displayMaterial) return;

        if (displayMaterial.HasProperty("_BaseColor"))
            displayMaterial.SetColor("_BaseColor", useObservationSurfaceShader ? surfaceBaseColor : neutralUnlitColor);
        else if (displayMaterial.HasProperty("_Color"))
            displayMaterial.SetColor("_Color", useObservationSurfaceShader ? surfaceBaseColor : neutralUnlitColor);

        if (!useObservationSurfaceShader)
            return;

        if (displayMaterial.HasProperty("_FresnelColor"))
            displayMaterial.SetColor("_FresnelColor", surfaceFresnelColor);
        if (displayMaterial.HasProperty("_GridColor"))
            displayMaterial.SetColor("_GridColor", wireColor);
        if (displayMaterial.HasProperty("_BaseAlpha"))
            displayMaterial.SetFloat("_BaseAlpha", surfaceBaseAlpha);
        if (displayMaterial.HasProperty("_FresnelPower"))
            displayMaterial.SetFloat("_FresnelPower", surfaceFresnelPower);
        if (displayMaterial.HasProperty("_FresnelStrength"))
            displayMaterial.SetFloat("_FresnelStrength", surfaceFresnelStrength);
        if (displayMaterial.HasProperty("_GridScale"))
            displayMaterial.SetFloat("_GridScale", surfaceGridScale);
        if (displayMaterial.HasProperty("_GridThickness"))
            displayMaterial.SetFloat("_GridThickness", surfaceGridThickness);
        if (displayMaterial.HasProperty("_GridIntensity"))
            displayMaterial.SetFloat("_GridIntensity", surfaceGridIntensity);
        if (displayMaterial.HasProperty("_Cull"))
            displayMaterial.SetFloat("_Cull", 0f);
    }

    private void EnsureWireMaterial()
    {
        if (!IsLegacyVisualChainAllowed())
            return;
        if (wireMaterial) return;
        Shader sh = Shader.Find("Universal Render Pipeline/Unlit");
        if (!sh) sh = Shader.Find("Unlit/Color");
        if (!sh) return;
        wireMaterial = new Material(sh);
        if (wireMaterial.HasProperty("_BaseColor"))
            wireMaterial.SetColor("_BaseColor", wireColor);
        else if (wireMaterial.HasProperty("_Color"))
            wireMaterial.SetColor("_Color", wireColor);
    }

    public void ClearDisplay()
    {
        for (int i = 0; i < _builtMeshes.Count; i++)
        {
            if (_builtMeshes[i] != null)
            {
                if (Application.isPlaying) Destroy(_builtMeshes[i]);
                else DestroyImmediate(_builtMeshes[i]);
            }
        }
        _builtMeshes.Clear();
        for (int i = 0; i < _smoothedMeshes.Count; i++)
        {
            if (_smoothedMeshes[i] != null)
            {
                if (Application.isPlaying) Destroy(_smoothedMeshes[i]);
                else DestroyImmediate(_smoothedMeshes[i]);
            }
        }
        _smoothedMeshes.Clear();
        for (int i = 0; i < _rawWireMeshes.Count; i++)
        {
            if (_rawWireMeshes[i] != null)
            {
                if (Application.isPlaying) Destroy(_rawWireMeshes[i]);
                else DestroyImmediate(_rawWireMeshes[i]);
            }
        }
        _rawWireMeshes.Clear();
        for (int i = 0; i < _smoothedWireMeshes.Count; i++)
        {
            if (_smoothedWireMeshes[i] != null)
            {
                if (Application.isPlaying) Destroy(_smoothedWireMeshes[i]);
                else DestroyImmediate(_smoothedWireMeshes[i]);
            }
        }
        _smoothedWireMeshes.Clear();

        for (int i = 0; i < _builtObjects.Count; i++)
        {
            if (_builtObjects[i] != null)
            {
                if (Application.isPlaying) Destroy(_builtObjects[i]);
                else DestroyImmediate(_builtObjects[i]);
            }
        }
        _builtObjects.Clear();
        for (int i = 0; i < _smoothedObjects.Count; i++)
        {
            if (_smoothedObjects[i] != null)
            {
                if (Application.isPlaying) Destroy(_smoothedObjects[i]);
                else DestroyImmediate(_smoothedObjects[i]);
            }
        }
        _smoothedObjects.Clear();
        for (int i = 0; i < _rawWireObjects.Count; i++)
        {
            if (_rawWireObjects[i] != null)
            {
                if (Application.isPlaying) Destroy(_rawWireObjects[i]);
                else DestroyImmediate(_rawWireObjects[i]);
            }
        }
        _rawWireObjects.Clear();
        for (int i = 0; i < _smoothedWireObjects.Count; i++)
        {
            if (_smoothedWireObjects[i] != null)
            {
                if (Application.isPlaying) Destroy(_smoothedWireObjects[i]);
                else DestroyImmediate(_smoothedWireObjects[i]);
            }
        }
        _smoothedWireObjects.Clear();
    }

    public void ClearLockedDisplay()
    {
        for (int i = 0; i < _lockedMeshes.Count; i++)
        {
            if (_lockedMeshes[i] != null)
            {
                if (Application.isPlaying) Destroy(_lockedMeshes[i]);
                else DestroyImmediate(_lockedMeshes[i]);
            }
        }
        _lockedMeshes.Clear();
    }

    public void SetVisible(bool visible)
    {
        if (!IsLegacyVisualChainAllowed())
            visible = false;

        if (!visible)
        {
            for (int i = 0; i < _builtObjects.Count; i++)
            {
                if (_builtObjects[i]) _builtObjects[i].SetActive(false);
            }
            for (int i = 0; i < _smoothedObjects.Count; i++)
            {
                if (_smoothedObjects[i]) _smoothedObjects[i].SetActive(false);
            }
            for (int i = 0; i < _rawWireObjects.Count; i++)
            {
                if (_rawWireObjects[i]) _rawWireObjects[i].SetActive(false);
            }
            for (int i = 0; i < _smoothedWireObjects.Count; i++)
            {
                if (_smoothedWireObjects[i]) _smoothedWireObjects[i].SetActive(false);
            }
            return;
        }

        bool showRaw = compareViewMode == CompareViewMode.Raw || compareViewMode == CompareViewMode.Both;
        bool showSmoothed = compareViewMode == CompareViewMode.Smoothed || compareViewMode == CompareViewMode.Both;

        for (int i = 0; i < _builtObjects.Count; i++)
        {
            if (_builtObjects[i]) _builtObjects[i].SetActive(showRaw);
        }
        for (int i = 0; i < _smoothedObjects.Count; i++)
        {
            if (_smoothedObjects[i]) _smoothedObjects[i].SetActive(showSmoothed);
        }
        for (int i = 0; i < _rawWireObjects.Count; i++)
        {
            if (_rawWireObjects[i]) _rawWireObjects[i].SetActive(showRaw);
        }
        for (int i = 0; i < _smoothedWireObjects.Count; i++)
        {
            if (_smoothedWireObjects[i]) _smoothedWireObjects[i].SetActive(showSmoothed);
        }
        ApplyDisplayStyle();
    }

    public void SetCompareViewMode(CompareViewMode mode, bool applyVisibility = true)
    {
        bool changed = compareViewMode != mode;
        compareViewMode = mode;
        if (changed && autoRebuildMissingModeOnSwitch)
            EnsureModeBuilt(mode);
        if (applyVisibility)
            SetVisible(true);
    }

    public CompareViewMode CycleCompareViewMode(bool applyVisibility = true)
    {
        int next = ((int)compareViewMode + 1) % 3;
        compareViewMode = (CompareViewMode)next;
        if (autoRebuildMissingModeOnSwitch)
            EnsureModeBuilt(compareViewMode);
        if (applyVisibility)
            SetVisible(true);
        return compareViewMode;
    }

    private void EnsureModeBuilt(CompareViewMode mode)
    {
        if (!buildOnlyCurrentCompareMode) return;

        bool needRaw = mode == CompareViewMode.Raw || mode == CompareViewMode.Both;
        bool needSmoothed = mode == CompareViewMode.Smoothed || mode == CompareViewMode.Both;
        bool hasRaw = _builtMeshes.Count > 0 || _rawWireMeshes.Count > 0;
        bool hasSmoothed = _smoothedMeshes.Count > 0 || _smoothedWireMeshes.Count > 0;

        if ((needRaw && !hasRaw) || (needSmoothed && !hasSmoothed))
        {
            if (HasLockedDisplay) RebuildFromLocked();
            else RebuildFromSource();
        }
    }

    public void SetDisplayStyle(DisplayStyle style, bool applyVisibility = true)
    {
        displayStyle = style;
        if (applyVisibility)
            SetVisible(true);
    }

    private void ApplyDisplayStyle()
    {
        bool showFaces = displayStyle != DisplayStyle.WireOnly;
        bool showWire = displayStyle != DisplayStyle.FacesOnly;

        for (int i = 0; i < _builtObjects.Count; i++)
        {
            if (!_builtObjects[i]) continue;
            var mr = _builtObjects[i].GetComponent<MeshRenderer>();
            if (mr) mr.enabled = showFaces;
        }
        for (int i = 0; i < _smoothedObjects.Count; i++)
        {
            if (!_smoothedObjects[i]) continue;
            var mr = _smoothedObjects[i].GetComponent<MeshRenderer>();
            if (mr) mr.enabled = showFaces;
        }
        for (int i = 0; i < _rawWireObjects.Count; i++)
        {
            if (!_rawWireObjects[i]) continue;
            var mr = _rawWireObjects[i].GetComponent<MeshRenderer>();
            if (mr) mr.enabled = showWire;
        }
        for (int i = 0; i < _smoothedWireObjects.Count; i++)
        {
            if (!_smoothedWireObjects[i]) continue;
            var mr = _smoothedWireObjects[i].GetComponent<MeshRenderer>();
            if (mr) mr.enabled = showWire;
        }
    }

    private bool ShouldBuildRawSet()
    {
        if (!buildOnlyCurrentCompareMode) return true;
        return compareViewMode == CompareViewMode.Raw || compareViewMode == CompareViewMode.Both;
    }

    private bool ShouldBuildSmoothedSet()
    {
        if (!buildSmoothedPreview) return false;
        if (!buildOnlyCurrentCompareMode) return true;
        return compareViewMode == CompareViewMode.Smoothed || compareViewMode == CompareViewMode.Both;
    }

    public void RebuildFromSource()
    {
        ResolveRefs();
        if (!IsLegacyVisualChainAllowed())
        {
            DisableLegacyOutput();
            return;
        }
        bool buildRawSet = ShouldBuildRawSet();
        bool buildSmoothedSet = ShouldBuildSmoothedSet();
        bool buildWireSet = displayStyle != DisplayStyle.FacesOnly;
        bool frozen = sessionController && sessionController.State == ScanCoverSkeletonSessionController.SessionState.Frozen;
        if (frozen && refreshMesherBeforeFrozenRebuild && mesher != null)
        {
            mesher.BuildAllNow();
            sourceChunksRoot = mesher.chunksRoot;
        }
        EnsureDisplayRoot();
        EnsureDisplayMaterial();
        ClearDisplay();

        if (!sourceChunksRoot)
        {
            if (debugLog) Debug.LogWarning("[ScanCoverDisplaySurfaceBuilder] sourceChunksRoot is null.");
            _lastChunkCount = 0;
            return;
        }

        MeshFilter[] filters = sourceChunksRoot.GetComponentsInChildren<MeshFilter>(true);
        if (filters == null || filters.Length == 0)
        {
            _lastChunkCount = 0;
            return;
        }

        Matrix4x4 worldToDisplay = displayRoot.worldToLocalMatrix;
        var batch = new List<CombineInstance>(256);
        int batchVerts = 0;
        int batchIndex = 0;
        int usedFilters = 0;

        for (int i = 0; i < filters.Length; i++)
        {
            MeshFilter mf = filters[i];
            if (!mf || mf.transform == sourceChunksRoot) continue;
            Mesh src = mf.sharedMesh;
            if (!src || src.vertexCount <= 0) continue;

            var ci = new CombineInstance
            {
                mesh = src,
                transform = worldToDisplay * mf.transform.localToWorldMatrix,
                lightmapScaleOffset = Vector4.zero,
                realtimeLightmapScaleOffset = Vector4.zero
            };

            if (batchVerts > 0 && batchVerts + src.vertexCount > combineMaxVertices)
            {
                FlushBatch(batch, batchIndex++, buildRawSet, buildSmoothedSet, buildWireSet);
                batchVerts = 0;
            }

            batch.Add(ci);
            batchVerts += src.vertexCount;
            usedFilters++;
        }

        if (batch.Count > 0)
            FlushBatch(batch, batchIndex++, buildRawSet, buildSmoothedSet, buildWireSet);

        _lastChunkCount = mesher ? mesher.ChunkCount : usedFilters;

        if (hideSourceRenderersWhenBuilt)
            SetSourceChunkRenderersVisible(false);
        SetVisible(visibleAfterBuild);

        if (frozen && lockDisplayAfterFrozenBuild)
        {
            CaptureLockFromCurrentDisplay();
            if (smoothLockedDisplay)
                RebuildFromLocked();
        }

        if (debugLog)
            Debug.Log($"[ScanCoverDisplaySurfaceBuilder] built raw={_builtObjects.Count}, smoothed={_smoothedObjects.Count}, sourceFilters={usedFilters}, chunks={_lastChunkCount}, mode={compareViewMode}");
    }

    public void RebuildFromLocked()
    {
        if (!IsLegacyVisualChainAllowed())
        {
            DisableLegacyOutput();
            return;
        }
        EnsureDisplayRoot();
        EnsureDisplayMaterial();
        bool buildRawSet = ShouldBuildRawSet();
        bool buildSmoothedSet = ShouldBuildSmoothedSet();
        bool buildWireSet = displayStyle != DisplayStyle.FacesOnly;
        if (!HasLockedDisplay)
        {
            ClearDisplay();
            return;
        }

        ClearDisplay();
        for (int i = 0; i < _lockedMeshes.Count; i++)
        {
            Mesh src = _lockedMeshes[i];
            if (!src) continue;

            if (buildRawSet)
            {
                Mesh rawClone = Instantiate(src);
                rawClone.name = src.name + "_Raw";
                ApplyGeometryFiltersInPlace(rawClone);
                GameObject rawGo = new GameObject(rawClone.name, typeof(MeshFilter), typeof(MeshRenderer));
                rawGo.transform.SetParent(displayRoot, false);
                var rawMf = rawGo.GetComponent<MeshFilter>();
                var rawMr = rawGo.GetComponent<MeshRenderer>();
                rawMf.sharedMesh = rawClone;
                rawMr.sharedMaterial = displayMaterial;
                _builtMeshes.Add(rawClone);
                _builtObjects.Add(rawGo);
                if (buildWireSet)
                    CreateWireObjectFor(rawClone, rawClone.name + "_Wire", _rawWireMeshes, _rawWireObjects);
            }

            if (buildSmoothedSet)
            {
                Mesh smoothClone = Instantiate(src);
                smoothClone.name = src.name + "_Smoothed";
                if (smoothLockedDisplay)
                    SmoothMeshTaubinInPlace(
                        smoothClone,
                        lockedSmoothIterations,
                        lockedSmoothLambda,
                        lockedSmoothMu,
                        lockedSmoothMaxStepMeters,
                        preserveBoundaryVertices,
                        boundarySmoothFactor,
                        smoothNormalThresholdDeg,
                        tangentPlaneSmoothing);
                ApplyGeometryFiltersInPlace(smoothClone);
                GameObject smoothGo = new GameObject(smoothClone.name, typeof(MeshFilter), typeof(MeshRenderer));
                smoothGo.transform.SetParent(displayRoot, false);
                var smoothMf = smoothGo.GetComponent<MeshFilter>();
                var smoothMr = smoothGo.GetComponent<MeshRenderer>();
                smoothMf.sharedMesh = smoothClone;
                smoothMr.sharedMaterial = displayMaterial;
                _smoothedMeshes.Add(smoothClone);
                _smoothedObjects.Add(smoothGo);
                if (buildWireSet)
                    CreateWireObjectFor(smoothClone, smoothClone.name + "_Wire", _smoothedWireMeshes, _smoothedWireObjects);
            }
        }
        SetVisible(visibleAfterBuild);
    }

    public void SetSourceChunkRenderersVisible(bool visible)
    {
        if (!IsLegacyVisualChainAllowed())
            visible = false;
        if (!sourceChunksRoot) return;
        var renderers = sourceChunksRoot.GetComponentsInChildren<MeshRenderer>(true);
        for (int i = 0; i < renderers.Length; i++)
        {
            if (renderers[i]) renderers[i].enabled = visible;
        }
        if (mesher) mesher.rendererVisibleAfterBuild = visible;
    }

    private bool IsLegacyVisualChainAllowed()
    {
        if (!sessionController)
            sessionController = GetComponent<ScanCoverSkeletonSessionController>();
        return sessionController == null || sessionController.enableLegacyVisualChain;
    }

    private void DisableLegacyOutput()
    {
        ClearDisplay();
        ClearLockedDisplay();
        if (displayRoot != null)
        {
            if (Application.isPlaying) Destroy(displayRoot.gameObject);
            else DestroyImmediate(displayRoot.gameObject);
            displayRoot = null;
        }
    }

    private void FlushBatch(List<CombineInstance> batch, int batchIndex, bool buildRawSet, bool buildSmoothedSet, bool buildWireSet)
    {
        if (batch == null || batch.Count == 0) return;

        Mesh combined = new Mesh();
        combined.name = $"DisplaySurface_{batchIndex}";
        combined.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        combined.CombineMeshes(batch.ToArray(), true, true, false);

        if (weldVertices && weldToleranceMeters > 0f)
            WeldMeshInPlace(combined, weldToleranceMeters, recalculateNormals);
        else if (recalculateNormals)
            combined.RecalculateNormals();

        ApplyGeometryFiltersInPlace(combined);

        if (recalculateBounds)
            combined.RecalculateBounds();

        if (buildRawSet)
        {
            GameObject go = new GameObject($"DisplaySurface_{batchIndex}", typeof(MeshFilter), typeof(MeshRenderer));
            go.transform.SetParent(displayRoot, false);
            var mf = go.GetComponent<MeshFilter>();
            var mr = go.GetComponent<MeshRenderer>();
            mf.sharedMesh = combined;
            mr.sharedMaterial = displayMaterial;
            _builtMeshes.Add(combined);
            _builtObjects.Add(go);
            if (buildWireSet)
                CreateWireObjectFor(combined, $"DisplaySurface_{batchIndex}_WireRaw", _rawWireMeshes, _rawWireObjects);
        }

        if (buildSmoothedSet)
        {
            Mesh smoothed = Instantiate(combined);
            smoothed.name = $"DisplaySurface_{batchIndex}_Smoothed";
            if (smoothLockedDisplay)
                SmoothMeshTaubinInPlace(
                    smoothed,
                    lockedSmoothIterations,
                    lockedSmoothLambda,
                    lockedSmoothMu,
                    lockedSmoothMaxStepMeters,
                    preserveBoundaryVertices,
                    boundarySmoothFactor,
                    smoothNormalThresholdDeg,
                    tangentPlaneSmoothing);
            ApplyGeometryFiltersInPlace(smoothed);

            GameObject smoothGo = new GameObject(smoothed.name, typeof(MeshFilter), typeof(MeshRenderer));
            smoothGo.transform.SetParent(displayRoot, false);
            var smf = smoothGo.GetComponent<MeshFilter>();
            var smr = smoothGo.GetComponent<MeshRenderer>();
            smf.sharedMesh = smoothed;
            smr.sharedMaterial = displayMaterial;
            _smoothedMeshes.Add(smoothed);
            _smoothedObjects.Add(smoothGo);
            if (buildWireSet)
                CreateWireObjectFor(smoothed, $"{smoothed.name}_Wire", _smoothedWireMeshes, _smoothedWireObjects);
        }

        if (!buildRawSet)
        {
            if (Application.isPlaying) Destroy(combined);
            else DestroyImmediate(combined);
        }

        batch.Clear();
    }

    private void CreateWireObjectFor(Mesh source, string name, List<Mesh> meshStore, List<GameObject> objStore)
    {
        EnsureWireMaterial();
        if (!source || !wireMaterial) return;
        Mesh wire = BuildWireMeshFromTriangles(source);
        if (!wire) return;

        GameObject go = new GameObject(name, typeof(MeshFilter), typeof(MeshRenderer));
        go.transform.SetParent(displayRoot, false);
        var mf = go.GetComponent<MeshFilter>();
        var mr = go.GetComponent<MeshRenderer>();
        mf.sharedMesh = wire;
        mr.sharedMaterial = wireMaterial;
        meshStore.Add(wire);
        objStore.Add(go);
    }

    private void CaptureLockFromCurrentDisplay()
    {
        ClearLockedDisplay();
        List<Mesh> source = _builtMeshes.Count > 0 ? _builtMeshes : _smoothedMeshes;
        for (int i = 0; i < source.Count; i++)
        {
            Mesh src = source[i];
            if (!src) continue;
            Mesh clone = Instantiate(src);
            clone.name = src.name + "_Locked";
            _lockedMeshes.Add(clone);
        }
        if (debugLog)
            Debug.Log($"[ScanCoverDisplaySurfaceBuilder] Lock captured meshes={_lockedMeshes.Count}");
    }

    private static void SmoothMeshTaubinInPlace(
        Mesh mesh,
        int iterations,
        float lambda,
        float mu,
        float maxStepMeters,
        bool preserveBoundary,
        float boundaryFactor,
        float normalThresholdDeg,
        bool tangentPlaneConstraint)
    {
        if (!mesh || iterations <= 0)
            return;

        Vector3[] verts = mesh.vertices;
        int[] tris = mesh.triangles;
        if (verts == null || verts.Length == 0 || tris == null || tris.Length < 3)
            return;

        var neighbors = BuildVertexNeighbors(verts.Length, tris, out bool[] isBoundary);
        if (neighbors == null || neighbors.Length != verts.Length)
            return;

        Vector3[] work = new Vector3[verts.Length];
        Vector3[] next = new Vector3[verts.Length];
        Vector3[] baseNormals = BuildVertexNormals(verts, tris);
        float cosThreshold = Mathf.Cos(Mathf.Clamp(normalThresholdDeg, 1f, 89f) * Mathf.Deg2Rad);

        System.Array.Copy(verts, work, verts.Length);

        float maxStep = Mathf.Max(0f, maxStepMeters);
        float boundaryMul = Mathf.Clamp01(boundaryFactor);
        for (int iter = 0; iter < iterations; iter++)
        {
            TaubinPass(
                work, next, neighbors, isBoundary, baseNormals, lambda, maxStep,
                preserveBoundary, boundaryMul, cosThreshold, tangentPlaneConstraint);
            TaubinPass(
                next, work, neighbors, isBoundary, baseNormals, mu, maxStep,
                preserveBoundary, boundaryMul, cosThreshold, tangentPlaneConstraint);
        }

        mesh.SetVertices(work);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
    }

    private static void TaubinPass(
        Vector3[] src,
        Vector3[] dst,
        List<int>[] neighbors,
        bool[] isBoundary,
        Vector3[] baseNormals,
        float factor,
        float maxStep,
        bool preserveBoundary,
        float boundaryMul,
        float cosThreshold,
        bool tangentPlaneConstraint)
    {
        for (int i = 0; i < src.Length; i++)
        {
            List<int> n = neighbors[i];
            if (n == null || n.Count == 0)
            {
                dst[i] = src[i];
                continue;
            }

            Vector3 avg = Vector3.zero;
            int used = 0;
            Vector3 ni = baseNormals[i].sqrMagnitude > 1e-8f ? baseNormals[i].normalized : Vector3.up;
            for (int k = 0; k < n.Count; k++)
            {
                int j = n[k];
                Vector3 nj = baseNormals[j].sqrMagnitude > 1e-8f ? baseNormals[j].normalized : ni;
                if (Vector3.Dot(ni, nj) < cosThreshold)
                    continue;
                avg += src[j];
                used++;
            }
            if (used <= 0)
            {
                dst[i] = src[i];
                continue;
            }
            avg /= used;

            Vector3 step = (avg - src[i]) * factor;
            if (tangentPlaneConstraint)
            {
                Vector3 nrm = ni;
                step -= nrm * Vector3.Dot(step, nrm);
            }
            if (preserveBoundary && isBoundary != null && i < isBoundary.Length && isBoundary[i])
                step *= boundaryMul;
            if (maxStep > 0f)
                step = Vector3.ClampMagnitude(step, maxStep);
            dst[i] = src[i] + step;
        }
    }

    private static List<int>[] BuildVertexNeighbors(int vertexCount, int[] tris, out bool[] isBoundary)
    {
        var sets = new HashSet<int>[vertexCount];
        var edgeUse = new Dictionary<ulong, int>(tris.Length);
        for (int i = 0; i < tris.Length; i += 3)
        {
            if (i + 2 >= tris.Length) break;
            int a = tris[i];
            int b = tris[i + 1];
            int c = tris[i + 2];
            if (a < 0 || b < 0 || c < 0 || a >= vertexCount || b >= vertexCount || c >= vertexCount)
                continue;

            if (sets[a] == null) sets[a] = new HashSet<int>();
            if (sets[b] == null) sets[b] = new HashSet<int>();
            if (sets[c] == null) sets[c] = new HashSet<int>();

            sets[a].Add(b); sets[a].Add(c);
            sets[b].Add(a); sets[b].Add(c);
            sets[c].Add(a); sets[c].Add(b);

            CountEdge(edgeUse, a, b);
            CountEdge(edgeUse, b, c);
            CountEdge(edgeUse, c, a);
        }

        var result = new List<int>[vertexCount];
        isBoundary = new bool[vertexCount];
        for (int i = 0; i < vertexCount; i++)
            result[i] = sets[i] != null ? new List<int>(sets[i]) : new List<int>(0);

        foreach (var kv in edgeUse)
        {
            if (kv.Value != 1) continue;
            DecodeEdge(kv.Key, out int a, out int b);
            if (a >= 0 && a < isBoundary.Length) isBoundary[a] = true;
            if (b >= 0 && b < isBoundary.Length) isBoundary[b] = true;
        }
        return result;
    }

    private static Vector3[] BuildVertexNormals(Vector3[] verts, int[] tris)
    {
        Vector3[] normals = new Vector3[verts.Length];
        for (int i = 0; i + 2 < tris.Length; i += 3)
        {
            int a = tris[i];
            int b = tris[i + 1];
            int c = tris[i + 2];
            if (a < 0 || b < 0 || c < 0 || a >= verts.Length || b >= verts.Length || c >= verts.Length)
                continue;
            Vector3 n = Vector3.Cross(verts[b] - verts[a], verts[c] - verts[a]);
            normals[a] += n;
            normals[b] += n;
            normals[c] += n;
        }
        for (int i = 0; i < normals.Length; i++)
            normals[i] = normals[i].sqrMagnitude > 1e-8f ? normals[i].normalized : Vector3.up;
        return normals;
    }

    private static void CountEdge(Dictionary<ulong, int> edgeUse, int a, int b)
    {
        ulong key = EncodeEdge(a, b);
        edgeUse.TryGetValue(key, out int v);
        edgeUse[key] = v + 1;
    }

    private static ulong EncodeEdge(int a, int b)
    {
        uint x = (uint)Mathf.Min(a, b);
        uint y = (uint)Mathf.Max(a, b);
        return ((ulong)x << 32) | y;
    }

    private static void DecodeEdge(ulong key, out int a, out int b)
    {
        a = (int)(key >> 32);
        b = (int)(key & 0xFFFFFFFFu);
    }

    private static void WeldMeshInPlace(Mesh mesh, float tolerance, bool recalcNormals)
    {
        if (!mesh) return;
        Vector3[] verts = mesh.vertices;
        int[] tris = mesh.triangles;
        Vector3[] norms = mesh.normals;
        bool hasNormals = norms != null && norms.Length == verts.Length;

        var map = new Dictionary<QuantizedVec3, int>(verts.Length);
        var newVerts = new List<Vector3>(verts.Length);
        var newNorms = new List<Vector3>(verts.Length);
        int[] remap = new int[verts.Length];

        float inv = 1f / Mathf.Max(1e-6f, tolerance);
        for (int i = 0; i < verts.Length; i++)
        {
            QuantizedVec3 q = new QuantizedVec3(verts[i], inv);
            if (!map.TryGetValue(q, out int idx))
            {
                idx = newVerts.Count;
                map.Add(q, idx);
                newVerts.Add(verts[i]);
                newNorms.Add(hasNormals ? norms[i] : Vector3.zero);
            }
            else if (hasNormals)
            {
                newNorms[idx] += norms[i];
            }
            remap[i] = idx;
        }

        for (int i = 0; i < tris.Length; i++)
            tris[i] = remap[tris[i]];

        mesh.Clear();
        mesh.SetVertices(newVerts);
        mesh.SetTriangles(tris, 0, true);

        if (hasNormals && !recalcNormals)
        {
            for (int i = 0; i < newNorms.Count; i++)
                newNorms[i] = newNorms[i].sqrMagnitude > 1e-8f ? newNorms[i].normalized : Vector3.up;
            mesh.SetNormals(newNorms);
        }
        else
        {
            mesh.RecalculateNormals();
        }
    }

    private void ApplyGeometryFiltersInPlace(Mesh mesh)
    {
        if (!mesh) return;

        Vector3[] verts = mesh.vertices;
        int[] tris = mesh.triangles;
        if (verts == null || tris == null || tris.Length < 3) return;

        int[] working = tris;

        if (enableNearViewerCull)
        {
            Vector3 viewerLocal = GetViewerLocalPosition();
            working = CullNearViewerTriangles(verts, working, viewerLocal, Mathf.Max(0.05f, nearViewerCullMeters));
        }

        if (enableSmallIslandCull)
        {
            working = CullSmallIslands(
                verts,
                working,
                Mathf.Max(1, minIslandTriangles),
                Mathf.Max(0f, minIslandAreaM2));
        }

        if (working.Length == tris.Length)
            return;

        mesh.SetTriangles(working, 0, true);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
    }

    private Vector3 GetViewerLocalPosition()
    {
        if (!viewerTransform && Camera.main) viewerTransform = Camera.main.transform;
        if (!displayRoot) return viewerTransform ? viewerTransform.position : Vector3.zero;
        if (!viewerTransform) return Vector3.zero;
        return displayRoot.InverseTransformPoint(viewerTransform.position);
    }

    private static int[] CullNearViewerTriangles(Vector3[] verts, int[] tris, Vector3 viewerLocal, float minDistance)
    {
        float minDistSq = minDistance * minDistance;
        var kept = new List<int>(tris.Length);
        for (int i = 0; i + 2 < tris.Length; i += 3)
        {
            int a = tris[i];
            int b = tris[i + 1];
            int c = tris[i + 2];
            if (a < 0 || b < 0 || c < 0 || a >= verts.Length || b >= verts.Length || c >= verts.Length)
                continue;

            Vector3 center = (verts[a] + verts[b] + verts[c]) / 3f;
            if ((center - viewerLocal).sqrMagnitude < minDistSq)
                continue;

            kept.Add(a);
            kept.Add(b);
            kept.Add(c);
        }
        return kept.ToArray();
    }

    private static int[] CullSmallIslands(Vector3[] verts, int[] tris, int minTriCount, float minArea)
    {
        int triCount = tris.Length / 3;
        if (triCount <= 0) return tris;

        var triByVertex = new List<int>[verts.Length];
        for (int t = 0; t < triCount; t++)
        {
            int i = t * 3;
            int a = tris[i];
            int b = tris[i + 1];
            int c = tris[i + 2];
            if (a < 0 || b < 0 || c < 0 || a >= verts.Length || b >= verts.Length || c >= verts.Length)
                continue;

            if (triByVertex[a] == null) triByVertex[a] = new List<int>(6);
            if (triByVertex[b] == null) triByVertex[b] = new List<int>(6);
            if (triByVertex[c] == null) triByVertex[c] = new List<int>(6);
            triByVertex[a].Add(t);
            triByVertex[b].Add(t);
            triByVertex[c].Add(t);
        }

        var visited = new bool[triCount];
        var keepTri = new bool[triCount];
        var queue = new Queue<int>(128);
        var component = new List<int>(128);

        for (int seed = 0; seed < triCount; seed++)
        {
            if (visited[seed]) continue;
            visited[seed] = true;
            queue.Enqueue(seed);
            component.Clear();
            float area = 0f;

            while (queue.Count > 0)
            {
                int t = queue.Dequeue();
                component.Add(t);

                int idx = t * 3;
                int a = tris[idx];
                int b = tris[idx + 1];
                int c = tris[idx + 2];
                if (a < 0 || b < 0 || c < 0 || a >= verts.Length || b >= verts.Length || c >= verts.Length)
                    continue;

                area += Vector3.Cross(verts[b] - verts[a], verts[c] - verts[a]).magnitude * 0.5f;

                EnqueueNeighborTriangles(a, triByVertex, visited, queue);
                EnqueueNeighborTriangles(b, triByVertex, visited, queue);
                EnqueueNeighborTriangles(c, triByVertex, visited, queue);
            }

            bool keep = component.Count >= minTriCount && area >= minArea;
            if (keep)
            {
                for (int i = 0; i < component.Count; i++)
                    keepTri[component[i]] = true;
            }
        }

        var kept = new List<int>(tris.Length);
        for (int t = 0; t < triCount; t++)
        {
            if (!keepTri[t]) continue;
            int i = t * 3;
            kept.Add(tris[i]);
            kept.Add(tris[i + 1]);
            kept.Add(tris[i + 2]);
        }

        return kept.ToArray();
    }

    private static void EnqueueNeighborTriangles(int vertex, List<int>[] triByVertex, bool[] visited, Queue<int> queue)
    {
        if (vertex < 0 || vertex >= triByVertex.Length) return;
        var list = triByVertex[vertex];
        if (list == null) return;

        for (int i = 0; i < list.Count; i++)
        {
            int tri = list[i];
            if (tri < 0 || tri >= visited.Length || visited[tri]) continue;
            visited[tri] = true;
            queue.Enqueue(tri);
        }
    }

    private static Mesh BuildWireMeshFromTriangles(Mesh source)
    {
        if (!source) return null;
        var verts = source.vertices;
        var tris = source.triangles;
        if (verts == null || tris == null || verts.Length == 0 || tris.Length < 3) return null;

        var edgeSet = new HashSet<ulong>(tris.Length);
        var lineVerts = new List<Vector3>(tris.Length * 2);
        var lineIdx = new List<int>(tris.Length * 2);

        for (int i = 0; i + 2 < tris.Length; i += 3)
        {
            int a = tris[i];
            int b = tris[i + 1];
            int c = tris[i + 2];
            if (a < 0 || b < 0 || c < 0 || a >= verts.Length || b >= verts.Length || c >= verts.Length) continue;
            AppendEdgeIfNew(a, b, verts, edgeSet, lineVerts, lineIdx);
            AppendEdgeIfNew(b, c, verts, edgeSet, lineVerts, lineIdx);
            AppendEdgeIfNew(c, a, verts, edgeSet, lineVerts, lineIdx);
        }

        if (lineIdx.Count <= 0) return null;

        Mesh m = new Mesh();
        m.name = source.name + "_WireMesh";
        m.indexFormat = lineVerts.Count > 65535 ? UnityEngine.Rendering.IndexFormat.UInt32 : UnityEngine.Rendering.IndexFormat.UInt16;
        m.SetVertices(lineVerts);
        m.SetIndices(lineIdx, MeshTopology.Lines, 0, true);
        m.RecalculateBounds();
        return m;
    }

    private static void AppendEdgeIfNew(
        int a,
        int b,
        Vector3[] verts,
        HashSet<ulong> edgeSet,
        List<Vector3> lineVerts,
        List<int> lineIdx)
    {
        ulong key = EncodeEdge(a, b);
        if (!edgeSet.Add(key)) return;

        int baseIdx = lineVerts.Count;
        lineVerts.Add(verts[a]);
        lineVerts.Add(verts[b]);
        lineIdx.Add(baseIdx);
        lineIdx.Add(baseIdx + 1);
    }

    private readonly struct QuantizedVec3 : System.IEquatable<QuantizedVec3>
    {
        private readonly int x;
        private readonly int y;
        private readonly int z;

        public QuantizedVec3(Vector3 v, float invTol)
        {
            x = Mathf.RoundToInt(v.x * invTol);
            y = Mathf.RoundToInt(v.y * invTol);
            z = Mathf.RoundToInt(v.z * invTol);
        }

        public bool Equals(QuantizedVec3 other) => x == other.x && y == other.y && z == other.z;
        public override bool Equals(object obj) => obj is QuantizedVec3 other && Equals(other);
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
    }
}
