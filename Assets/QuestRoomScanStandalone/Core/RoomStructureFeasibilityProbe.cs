using System;
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Read-only feasibility probe for a future room-understanding layer.
    /// It reads the production TSDF, weight and confidence volumes, but never
    /// writes them and never changes extraction/publication decisions.
    /// </summary>
    internal sealed class RoomStructureFeasibilityProbe : IDisposable
    {
        private const int RecordStride = 17;
        // The integration lattice visits different voxel phases over several
        // accepted frames.  Twelve epochs covers one complete eight-phase pass
        // plus readback latency, without allowing an old TSDF fit to pose as a
        // new observation indefinitely.
        private const int FreshObservationEpochWindow = 12;
        private const int PaletteSize = 64;
        private const float StructuralAngleDegrees = 20f;
        // A tile may be noisy from one probe epoch to the next, but a local
        // family is already a geometric claim.  Keep the temporal latch a
        // little wider than the family gate, then require neighbouring tiles
        // to agree tightly before they can become one observation.  The old
        // shared 12 degree / 20 cm gate and 18 degree expansion could average
        // a real turn into one blended plane before the global layer saw it.
        private const float TemporalTrackAngleDegrees = 9f;
        private const float TemporalTrackOffsetMeters = 0.12f;
        private const float FamilyAngleDegrees = 6f;
        private const float FamilyOffsetMeters = 0.08f;
        private const float ExpansionAngleDegrees = 8f;
        private const float ExpansionOffsetMeters = 0.08f;
        private const float ExpansionResidualScale = 1.6f;
        private const float ExpansionLowConfidenceExtra = 0.30f;
        private const int ExpansionMaxDepth = 2;
        private const float StableAngleJitterDegrees = 5f;
        private const float StableOffsetJitterMeters = 0.06f;
        private const int StableHitCount = 3;
        private const float PlaneCellSizeMeters = 0.20f;
        private const float StrongPlaneTrackAngleDegrees = 7f;
        private const float StrongPlaneTrackOffsetMeters = 0.12f;
        private const float PlaneTrackAngleDegrees = 10f;
        private const float PlaneTrackOffsetMeters = 0.15f;
        private const float PlaneTrackGapMeters = 1.20f;
        private const float PlaneTrackMergeGapMeters = 0.65f;
        private const float PendingPlaneLifetimeSeconds = 6f;
        private const float PendingCellLifetimeSeconds = 5f;
        private const float ActiveTrackLifetimeSeconds = 15f;
        private const float ActiveCellLifetimeSeconds = 15f;
        private const int VisiblePlaneEpochs = 4;
        private const int RoomPlaneEpochs = 16;
        private const int VisibleCellVotes = 4;
        private const int DurableCellVotes = 12;
        private const int RoomPlaneCellCount = 20;
        private const int CompetitionLossLimit = 4;
        private const int InitialTrackEvidence = 12;
        private const int MaximumTrackEvidence = 30;
        private const int WeakEvidenceLossLimit = 16;
        private const float StrongBirthAreaMeters = 0.25f;
        private const float MediumBirthAreaMeters = 0.12f;
        private const float FineBirthAreaMeters = 0.025f;

        private static readonly Color[] Palette = BuildPalette();

        private sealed class TileTrack
        {
            internal int Index;
            internal Vector3Int Coord;
            internal Vector3 CenterLocal;
            internal Vector3 Normal;
            internal float Offset;
            internal float ResidualMeters;
            internal float AngleJitterDegrees;
            internal float OffsetJitterMeters;
            internal float FreeSpaceScore;
            internal int SurfaceSamples;
            internal int GoodStreak;
            internal float LastGoodTime;
            internal bool Current;
            internal int FamilyId;
            internal int CellSizeVoxels;
            internal int WitnessSubcell;
            internal int SupportMask;

            internal bool Stable(float now)
            {
                return GoodStreak >= StableHitCount &&
                       now - LastGoodTime <= 1.25f &&
                       AngleJitterDegrees <= StableAngleJitterDegrees &&
                       OffsetJitterMeters <= StableOffsetJitterMeters;
            }
        }

        private sealed class Family
        {
            internal int Id;
            internal readonly List<TileTrack> Members = new List<TileTrack>(16);
            internal readonly List<TileEvidence> Extent = new List<TileEvidence>(16);
            internal Vector3 Normal;
            internal float Offset;
            internal Vector3 Center;
            internal Bounds Bounds;
            internal float SupportArea;
            internal float AngleJitter;
            internal float OffsetJitter;
            internal float FreeSpaceScore;
            internal bool Horizontal;
            internal int TrackId;
        }

        private sealed class TileEvidence
        {
            internal int Index;
            internal Vector3Int Coord;
            internal Vector3 CenterLocal;
            internal Vector3 Normal;
            internal float Offset;
            internal float ResidualMeters;
            internal float LowConfidenceFraction;
            internal int SurfaceSamples;
            internal int FreshSurfaceSamples;
            internal int Verdict;
            internal int ExpansionDepth;
            internal int CellSizeVoxels;
            internal int WitnessSubcell;
            internal int SupportMask;
        }

        private sealed class IntersectionTrack
        {
            internal Vector3 Anchor;
            internal Vector3 Direction;
            internal float DriftMeters;
            internal float AngleJitterDegrees;
            internal int Hits;
            internal float LastSeen;
        }

        private sealed class PlaneCell
        {
            internal int Votes;
            internal int LastEpoch;
            internal float LastSeen;
        }

        private sealed class PlaneTrack
        {
            internal int Id;
            internal Vector3 Normal;
            internal float Offset;
            internal Vector3 U;
            internal Vector3 V;
            internal bool Horizontal;
            internal int SeenEpochs;
            internal int LastSeenEpoch;
            internal float LastSeen;
            internal int EvidenceScore;
            internal int WeakEvidenceStreak;
            internal int CompetitionLosses;
            internal int LastCompetitionEpoch;
            internal int MinU;
            internal int MaxU;
            internal int MinV;
            internal int MaxV;
            internal bool HasBounds;
            internal readonly Dictionary<long, PlaneCell> Cells =
                new Dictionary<long, PlaneCell>(128);
        }

        private sealed class PlaneSeed
        {
            internal int FamilyId;
            internal Vector3 Normal;
            internal float Offset;
            internal Vector3 Center;
            internal bool Horizontal;
            internal int Hits;
            internal int LastSeenEpoch;
            internal float LastSeen;
        }

        private readonly VolumeIntegrator _volume;
        private readonly Transform _volumeTransform;
        private readonly ComputeShader _compute;
        private readonly int _tileSize;
        private readonly float _interval;
        private readonly float _minWeight;
        private readonly float _lowConfidenceThreshold;
        private readonly float _maxLowConfidenceFraction;
        private readonly float _maxResidualMeters;
        private readonly int _minSurfaceSamples;

        private ComputeKernelHelper _kernel;
        private ComputeBuffer _records;
        private Vector3Int _tileCount;
        private int _blockCount;
        private int _generation;
        private int _pendingGeneration;
        private bool _pending;
        private float _pendingSince;
        private float _lastTick = -10f;
        private int _lastObservedIntegrationCount = -1;
        private bool _disposed;
        private bool _showOverlay;
        private string _fault = string.Empty;

        private readonly Dictionary<int, TileTrack> _tracks = new Dictionary<int, TileTrack>();
        private readonly Dictionary<int, TileEvidence> _currentEvidence =
            new Dictionary<int, TileEvidence>();
        private readonly Dictionary<long, IntersectionTrack> _intersections =
            new Dictionary<long, IntersectionTrack>();
        private readonly List<Family> _families = new List<Family>(24);
        // Independent canonical landmark layer. Local Family instances are
        // observations only; persistent identity, competition and retirement
        // live entirely behind this boundary.
        private readonly GlobalPlaneLandmarkLayer _globalPlaneLandmarks =
            new GlobalPlaneLandmarkLayer();
        private readonly Dictionary<int, PlaneTrack> _planeTracks =
            new Dictionary<int, PlaneTrack>();
        private readonly Dictionary<int, PlaneSeed> _planeSeeds =
            new Dictionary<int, PlaneSeed>();
        private readonly Dictionary<int, int> _paletteSlotByFamily =
            new Dictionary<int, int>();
        private int _nextFamilyId = 1;
        private int _nextPlaneTrackId = 1;
        private int _planeEpoch;

        private GameObject _visualRoot;
        private readonly Mesh[] _visualMeshes = new Mesh[PaletteSize];
        private readonly Material[] _visualMaterials = new Material[PaletteSize];
        private readonly List<Vector3>[] _visualVertices = new List<Vector3>[PaletteSize];
        private readonly List<int>[] _visualIndices = new List<int>[PaletteSize];

        private static readonly int RecordsId = Shader.PropertyToID("_RoomProbeRecords");
        private static readonly int ConfidenceId = Shader.PropertyToID("_RoomProbeConfidence");
        private static readonly int BlockCountId = Shader.PropertyToID("_RoomProbeBlockCount");
        private static readonly int ChunkCountId = Shader.PropertyToID("_RoomProbeChunkCount");
        private static readonly int ChunkSizeId = Shader.PropertyToID("_RoomProbeChunkSize");
        private static readonly int MinWeightId = Shader.PropertyToID("_RoomProbeMinWeight");
        private static readonly int LowConfMinId = Shader.PropertyToID("_RoomProbeLowConfMin");
        private static readonly int LowConfMaxFracId = Shader.PropertyToID("_RoomProbeLowConfMaxFrac");
        private static readonly int MaxResidualId = Shader.PropertyToID("_RoomProbeMaxResidualMeters");
        private static readonly int MinSurfaceId = Shader.PropertyToID("_RoomProbeMinSurfaceVoxels");
        private static readonly int VoxCountXId = Shader.PropertyToID("_FitVoxCountX");
        private static readonly int VoxCountYId = Shader.PropertyToID("_FitVoxCountY");
        private static readonly int VoxCountZId = Shader.PropertyToID("_FitVoxCountZ");
        private static readonly int VoxelSizeId = Shader.PropertyToID("_FitVoxelSize");
        private static readonly int TsdfId = Shader.PropertyToID("_FitTsdfVolume");
        private static readonly int WriteEpochsId =
            Shader.PropertyToID("_RoomProbeWriteEpochs");
        private static readonly int CurrentWriteEpochId =
            Shader.PropertyToID("_RoomProbeCurrentWriteEpoch");
        private static readonly int FreshEpochWindowId =
            Shader.PropertyToID("_RoomProbeFreshEpochWindow");

        internal string StatusCompact { get; private set; } = "骨架只读: 等待TSDF";

        internal void SetOverlayVisible(bool visible)
        {
            _showOverlay = visible;
            if (_visualRoot != null) _visualRoot.SetActive(visible);
        }

        internal RoomStructureFeasibilityProbe(VolumeIntegrator volume,
            Transform volumeTransform, ComputeShader compute, int tileSize = 8,
            float intervalSeconds = 0.5f, float minWeight = 0.08f,
            float lowConfidenceThreshold = 0.20f,
            float maxLowConfidenceFraction = 0.35f,
            float maxResidualMeters = 0.05f, int minSurfaceSamples = 24)
        {
            _volume = volume;
            _volumeTransform = volumeTransform;
            _compute = compute;
            _tileSize = Mathf.Max(4, tileSize);
            _interval = Mathf.Max(0.25f, intervalSeconds);
            _minWeight = Mathf.Max(0.01f, minWeight);
            _lowConfidenceThreshold = Mathf.Clamp01(lowConfidenceThreshold);
            _maxLowConfidenceFraction = Mathf.Clamp01(maxLowConfidenceFraction);
            _maxResidualMeters = Mathf.Max(0.005f, maxResidualMeters);
            _minSurfaceSamples = Mathf.Max(8, minSurfaceSamples);
        }

        internal void Tick(float now, bool showOverlay)
        {
            if (_disposed) return;
            SetOverlayVisible(showOverlay);
            if (_pending)
            {
                if (now - _pendingSince > 3f)
                {
                    _pending = false;
                    _fault = "STUCK";
                    StatusCompact = "骨架只读: 回读超时";
                }
                return;
            }
            if (now - _lastTick < _interval) return;
            if (_volume == null || _volume.Volume == null ||
                _volume.ConfidenceVolume == null ||
                _volume.InfiniTamWriteEpochs == null ||
                _volume.InfiniTamWriteEpoch == 0u || _compute == null)
            {
                StatusCompact = "骨架只读: 等待TSDF/置信/观测帧号";
                return;
            }
            // Time alone is not evidence.  Re-reading one unchanged TSDF while
            // the headset is paused must never promote a pending plane.
            int integrationCount = _volume.IntegrationCount;
            if (integrationCount <= _lastObservedIntegrationCount) return;
            _lastObservedIntegrationCount = integrationCount;
            _lastTick = now;

            try
            {
                EnsureResources();
                if (_records == null) return;
                Vector3Int vox = new Vector3Int(_volume.VoxelCount.x,
                    _volume.VoxelCount.y, _volume.VoxelCount.z);
                _compute.SetInt(BlockCountId, _blockCount);
                _compute.SetInts(ChunkCountId, _tileCount.x, _tileCount.y, _tileCount.z);
                _compute.SetInt(ChunkSizeId, _tileSize);
                _compute.SetFloat(MinWeightId, _minWeight);
                _compute.SetFloat(LowConfMinId, _lowConfidenceThreshold);
                _compute.SetFloat(LowConfMaxFracId, _maxLowConfidenceFraction);
                _compute.SetFloat(MaxResidualId, _maxResidualMeters);
                _compute.SetInt(MinSurfaceId, _minSurfaceSamples);
                _compute.SetInt(CurrentWriteEpochId,
                    unchecked((int)_volume.InfiniTamWriteEpoch));
                _compute.SetInt(FreshEpochWindowId,
                    FreshObservationEpochWindow);
                _compute.SetInt(VoxCountXId, vox.x);
                _compute.SetInt(VoxCountYId, vox.y);
                _compute.SetInt(VoxCountZId, vox.z);
                _compute.SetFloat(VoxelSizeId, _volume.VoxelSize);
                _kernel.Set(RecordsId, _records);
                _kernel.Set(ConfidenceId, _volume.ConfidenceVolume);
                _kernel.Set(TsdfId, _volume.Volume);
                _kernel.Set(WriteEpochsId, _volume.InfiniTamWriteEpochs);
                _kernel.DispatchFit(_blockCount, 1, 1);
                _pending = true;
                _pendingSince = now;
                int requestGeneration = _generation;
                _pendingGeneration = requestGeneration;
                AsyncGPUReadback.Request(_records,
                    request => OnReadback(request, requestGeneration));
            }
            catch (Exception ex)
            {
                _fault = ex.GetType().Name;
                StatusCompact = "骨架只读: " + _fault;
            }
        }

        private void EnsureResources()
        {
            Vector3Int vox = new Vector3Int(_volume.VoxelCount.x,
                _volume.VoxelCount.y, _volume.VoxelCount.z);
            Vector3Int wanted = new Vector3Int(
                Mathf.CeilToInt(vox.x / (float)_tileSize),
                Mathf.CeilToInt(vox.y / (float)_tileSize),
                Mathf.CeilToInt(vox.z / (float)_tileSize));
            int wantedBlocks = wanted.x * wanted.y * wanted.z;
            if (_records != null && wanted == _tileCount && wantedBlocks == _blockCount)
                return;

            ReleaseGpuResources();
            if (!_compute.HasKernel("RoomStructureProbe"))
            {
                _fault = "NOKRN";
                StatusCompact = "骨架只读: 内核缺失";
                return;
            }
            _tileCount = wanted;
            _blockCount = wantedBlocks;
            _records = new ComputeBuffer(_blockCount * RecordStride, sizeof(int));
            _kernel = new ComputeKernelHelper(_compute, "RoomStructureProbe");
            _generation++;
            _tracks.Clear();
            _families.Clear();
            _planeTracks.Clear();
            _planeSeeds.Clear();
            _paletteSlotByFamily.Clear();
            _planeEpoch = 0;
            _nextPlaneTrackId = 1;
            _intersections.Clear();
            _fault = string.Empty;
        }

        private void OnReadback(AsyncGPUReadbackRequest request, int requestGeneration)
        {
            if (_disposed) return;
            if (requestGeneration != _generation)
            {
                if (_pendingGeneration == requestGeneration) _pending = false;
                return;
            }
            if (_pendingGeneration != requestGeneration) return;
            _pending = false;
            if (request.hasError || _records == null)
            {
                _fault = "RBERR";
                StatusCompact = "骨架只读: 回读失败";
                return;
            }
            NativeArray<int> data = request.GetData<int>();
            if (data.Length < _blockCount * RecordStride)
            {
                _fault = "LEN";
                StatusCompact = "骨架只读: 记录不完整";
                return;
            }
            float now = Time.time;
            int candidates = UpdateTracks(data, now);
            BuildFamilies(now);
            UpdatePlaneTracks(now);
            UpdateIntersections(now);
            UpdateVisuals(now);
            UpdateStatus(candidates, now);
        }

        private int UpdateTracks(NativeArray<int> data, float now)
        {
            foreach (TileTrack oldTrack in _tracks.Values) oldTrack.Current = false;
            _currentEvidence.Clear();
            int candidates = 0;
            Vector3Int vox = new Vector3Int(_volume.VoxelCount.x,
                _volume.VoxelCount.y, _volume.VoxelCount.z);
            float voxelSize = _volume.VoxelSize;
            float structuralCos = Mathf.Cos(StructuralAngleDegrees * Mathf.Deg2Rad);
            float verticalYMax = Mathf.Sin(StructuralAngleDegrees * Mathf.Deg2Rad);

            for (int slot = 0; slot < _blockCount; slot++)
            {
                int b = slot * RecordStride;
                int verdict = data[b];
                if (verdict != 1 && verdict != 3 && verdict != 5) continue;
                Vector3 normal = new Vector3(data[b + 4], data[b + 5], data[b + 6]) / 100000f;
                if (normal.sqrMagnitude < 0.5f) continue;
                normal.Normalize();
                bool horizontal = Mathf.Abs(normal.y) >= structuralCos;
                // A wall is vertical regardless of its yaw in the room. The old
                // dominant-X/Z test silently rejected diagonal walls.
                bool vertical = Mathf.Abs(normal.y) <= verticalYMax;
                if (!horizontal && !vertical) continue;
                CanonicalizeNormal(ref normal);

                Vector3 centerVoxel = new Vector3(data[b + 7], data[b + 8], data[b + 9]) / 1000f;
                Vector3 centerLocal = new Vector3(
                    (centerVoxel.x - vox.x * 0.5f) * voxelSize,
                    (centerVoxel.y - vox.y * 0.5f) * voxelSize,
                    (centerVoxel.z - vox.z * 0.5f) * voxelSize);
                float offset = Vector3.Dot(normal, centerLocal);
                int sideCount = data[b + 12];
                float freeScore = sideCount > 0
                    ? Mathf.Clamp01((data[b + 10] + data[b + 11]) / (float)sideCount)
                    : 0f;
                int index = data[b + 13];
                int packedWindow = data[b + 14];
                int cellSizeVoxels = (packedWindow & 255) > 0
                    ? packedWindow & 255
                    : _tileSize;
                int witnessSubcell = packedWindow >> 8;
                int supportMask = data[b + 15] & 255;
                int surfaceSamples = data[b + 1];
                int freshSurfaceSamples = data[b + 16];
                // A fit over the persistent TSDF is only geometry.  It becomes
                // temporal evidence when a meaningful fraction of those exact
                // surface voxels was touched by recent valid depth-frame claims.
                // Otherwise scanning anywhere in the room would re-vote every
                // old tile in the volume on every probe epoch.
                if (!HasFreshObservation(surfaceSamples, freshSurfaceSamples))
                    continue;
                float lowConfidenceFraction = surfaceSamples > 0
                    ? data[b + 2] / (float)surfaceSamples
                    : 1f;
                _currentEvidence[index] = new TileEvidence
                {
                    Index = index,
                    Coord = DecodeCoord(index),
                    CenterLocal = centerLocal,
                    Normal = normal,
                    Offset = offset,
                    ResidualMeters = data[b + 3] / 10000f,
                    LowConfidenceFraction = lowConfidenceFraction,
                    SurfaceSamples = surfaceSamples,
                    FreshSurfaceSamples = freshSurfaceSamples,
                    Verdict = verdict,
                    CellSizeVoxels = cellSizeVoxels,
                    WitnessSubcell = witnessSubcell,
                    SupportMask = supportMask,
                };

                // Only the original strict verdict can create or mature a
                // temporal core. Rejected fits are expansion evidence only.
                if (verdict != 1) continue;
                if (!_tracks.TryGetValue(index, out TileTrack track))
                {
                    track = new TileTrack
                    {
                        Index = index,
                        Coord = DecodeCoord(index),
                        CenterLocal = centerLocal,
                        Normal = normal,
                        Offset = offset,
                        ResidualMeters = data[b + 3] / 10000f,
                        FreeSpaceScore = freeScore,
                        SurfaceSamples = data[b + 1],
                        CellSizeVoxels = cellSizeVoxels,
                        WitnessSubcell = witnessSubcell,
                        SupportMask = supportMask,
                        GoodStreak = 1,
                        LastGoodTime = now,
                        Current = true,
                    };
                    _tracks.Add(index, track);
                }
                else
                {
                    float angle = Vector3.Angle(track.Normal, normal);
                    float offsetDelta = Mathf.Abs(track.Offset - offset);
                    bool continuous = angle <= TemporalTrackAngleDegrees &&
                                      offsetDelta <= TemporalTrackOffsetMeters &&
                                      track.CellSizeVoxels == cellSizeVoxels &&
                                      track.WitnessSubcell == witnessSubcell;
                    track.GoodStreak = continuous ? Mathf.Min(255, track.GoodStreak + 1) : 1;
                    track.AngleJitterDegrees = track.GoodStreak <= 1
                        ? angle
                        : Mathf.Lerp(track.AngleJitterDegrees, angle, 0.35f);
                    track.OffsetJitterMeters = track.GoodStreak <= 1
                        ? offsetDelta
                        : Mathf.Lerp(track.OffsetJitterMeters, offsetDelta, 0.35f);
                    if (Vector3.Dot(track.Normal, normal) < 0f)
                    {
                        normal = -normal;
                        offset = -offset;
                    }
                    track.Normal = Vector3.Lerp(track.Normal, normal, 0.35f).normalized;
                    track.CenterLocal = Vector3.Lerp(track.CenterLocal, centerLocal, 0.35f);
                    track.Offset = Mathf.Lerp(track.Offset, offset, 0.35f);
                    track.ResidualMeters = Mathf.Lerp(track.ResidualMeters,
                        data[b + 3] / 10000f, 0.35f);
                    track.FreeSpaceScore = Mathf.Lerp(track.FreeSpaceScore, freeScore, 0.35f);
                    track.SurfaceSamples = data[b + 1];
                    track.CellSizeVoxels = cellSizeVoxels;
                    track.WitnessSubcell = witnessSubcell;
                    track.SupportMask = supportMask;
                    track.LastGoodTime = now;
                    track.Current = true;
                }
                candidates++;
            }

            var stale = new List<int>();
            foreach (KeyValuePair<int, TileTrack> pair in _tracks)
                if (now - pair.Value.LastGoodTime > 5f) stale.Add(pair.Key);
            for (int i = 0; i < stale.Count; i++) _tracks.Remove(stale[i]);
            return candidates;
        }

        private static bool HasFreshObservation(int surfaceSamples,
            int freshSurfaceSamples)
        {
            if (surfaceSamples <= 0) return false;
            int required = Mathf.Max(2, Mathf.CeilToInt(surfaceSamples * 0.10f));
            return freshSurfaceSamples >= required;
        }

        private void BuildFamilies(float now)
        {
            _families.Clear();
            var stable = new List<TileTrack>();
            foreach (TileTrack track in _tracks.Values)
                if (track.Stable(now)) stable.Add(track);
            int count = stable.Count;
            if (count == 0) return;
            stable.Sort(CompareTrackQuality);
            var assigned = new bool[count];
            float cosMatch = Mathf.Cos(FamilyAngleDegrees * Mathf.Deg2Rad);
            for (int i = 0; i < count; i++)
            {
                if (assigned[i]) continue;
                TileTrack anchor = stable[i];
                var family = new Family();
                family.Members.Add(anchor);
                assigned[i] = true;

                bool grew;
                do
                {
                    grew = false;
                    for (int j = 0; j < count; j++)
                    {
                        if (assigned[j] ||
                            !MatchesFixedAnchor(anchor, stable[j], cosMatch)) continue;
                        bool touches = false;
                        for (int m = 0; m < family.Members.Count; m++)
                        {
                            if (!AreNeighbourTiles(family.Members[m].Coord,
                                    stable[j].Coord)) continue;
                            touches = true;
                            break;
                        }
                        if (!touches) continue;
                        family.Members.Add(stable[j]);
                        assigned[j] = true;
                        grew = true;
                    }
                } while (grew);
                _families.Add(family);
            }

            var usedIds = new HashSet<int>();
            foreach (Family family in _families)
            {
                int inheritedId = PickInheritedFamilyId(family.Members, usedIds);
                family.Id = inheritedId > 0 ? inheritedId : _nextFamilyId++;
                usedIds.Add(family.Id);
                AccumulateFamily(family);
                for (int i = 0; i < family.Members.Count; i++)
                    family.Members[i].FamilyId = family.Id;
            }
            ExpandFamilies(now);
        }

        private static int CompareTrackQuality(TileTrack a, TileTrack b)
        {
            int residual = a.ResidualMeters.CompareTo(b.ResidualMeters);
            if (residual != 0) return residual;
            int jitter = a.AngleJitterDegrees.CompareTo(b.AngleJitterDegrees);
            if (jitter != 0) return jitter;
            return b.GoodStreak.CompareTo(a.GoodStreak);
        }

        private static bool MatchesFixedAnchor(TileTrack anchor, TileTrack candidate,
            float cosMatch)
        {
            float alignment = Vector3.Dot(anchor.Normal, candidate.Normal);
            float sign = alignment < 0f ? -1f : 1f;
            if (Mathf.Abs(alignment) < cosMatch) return false;
            float alignedOffset = candidate.Offset * sign;
            if (Mathf.Abs(anchor.Offset - alignedOffset) > FamilyOffsetMeters) return false;
            return Mathf.Abs(Vector3.Dot(anchor.Normal, candidate.CenterLocal) -
                             anchor.Offset) <= FamilyOffsetMeters;
        }

        private void ExpandFamilies(float now)
        {
            if (_families.Count == 0 || _currentEvidence.Count == 0) return;
            var claimed = new HashSet<int>();
            foreach (TileTrack track in _tracks.Values)
                if (track.Stable(now)) claimed.Add(track.Index);

            float cosExpansion = Mathf.Cos(ExpansionAngleDegrees * Mathf.Deg2Rad);
            float maxResidual = _maxResidualMeters * ExpansionResidualScale;
            float maxLowConfidence = Mathf.Min(0.80f,
                _maxLowConfidenceFraction + ExpansionLowConfidenceExtra);

            for (int depth = 1; depth <= ExpansionMaxDepth; depth++)
            {
                bool added = false;
                foreach (TileEvidence evidence in _currentEvidence.Values)
                {
                    if (claimed.Contains(evidence.Index) ||
                        evidence.SurfaceSamples < _minSurfaceSamples ||
                        evidence.ResidualMeters > maxResidual ||
                        evidence.LowConfidenceFraction > maxLowConfidence)
                        continue;

                    Family bestFamily = null;
                    float bestDistance = float.MaxValue;
                    for (int f = 0; f < _families.Count; f++)
                    {
                        Family family = _families[f];
                        if (!IsMatureFamily(family) ||
                            !TouchesFamilyAtDepth(evidence.Coord, family, depth - 1))
                            continue;

                        float alignment = Vector3.Dot(evidence.Normal, family.Normal);
                        float sign = alignment < 0f ? -1f : 1f;
                        if (Mathf.Abs(alignment) < cosExpansion) continue;
                        float alignedOffset = evidence.Offset * sign;
                        float offsetDistance = Mathf.Abs(alignedOffset - family.Offset);
                        float centerDistance = Mathf.Abs(
                            Vector3.Dot(family.Normal, evidence.CenterLocal) - family.Offset);
                        float distance = Mathf.Max(offsetDistance, centerDistance);
                        if (distance > ExpansionOffsetMeters || distance >= bestDistance) continue;
                        bestDistance = distance;
                        bestFamily = family;
                    }

                    if (bestFamily == null) continue;
                    evidence.ExpansionDepth = depth;
                    bestFamily.Extent.Add(evidence);
                    claimed.Add(evidence.Index);
                    added = true;
                }
                if (!added) break;
            }
        }

        private static bool TouchesFamilyAtDepth(Vector3Int coord, Family family,
            int previousDepth)
        {
            for (int i = 0; i < family.Members.Count; i++)
                if (AreNeighbourTiles(coord, family.Members[i].Coord)) return true;
            if (previousDepth <= 0) return false;
            for (int i = 0; i < family.Extent.Count; i++)
            {
                TileEvidence extent = family.Extent[i];
                if (extent.ExpansionDepth > previousDepth) continue;
                if (AreNeighbourTiles(coord, extent.Coord)) return true;
            }
            return false;
        }

        private void UpdatePlaneTracks(float now)
        {
            var observations = new List<GlobalPlaneObservation>(_families.Count);
            for (int i = 0; i < _families.Count; i++)
            {
                Family family = _families[i];
                family.TrackId = 0;
                if (!IsMatureFamily(family)) continue;
                var observation = new GlobalPlaneObservation
                {
                    SourceId = family.Id,
                    Normal = family.Normal,
                    Offset = family.Offset,
                    Center = family.Center,
                    Horizontal = family.Horizontal,
                    SupportArea = family.SupportArea,
                    AngleSigmaDegrees = Mathf.Max(1f, family.AngleJitter),
                    OffsetSigmaMeters = Mathf.Max(0.01f, family.OffsetJitter),
                };
                for (int m = 0; m < family.Members.Count; m++)
                {
                    TileTrack member = family.Members[m];
                    AppendLandmarkPatches(observation, member.Coord,
                        member.CenterLocal, member.CellSizeVoxels,
                        member.WitnessSubcell, member.SupportMask);
                }
                for (int e = 0; e < family.Extent.Count; e++)
                {
                    TileEvidence extent = family.Extent[e];
                    AppendLandmarkPatches(observation, extent.Coord,
                        extent.CenterLocal, extent.CellSizeVoxels,
                        extent.WitnessSubcell, extent.SupportMask);
                }
                observations.Add(observation);
            }
            _globalPlaneLandmarks.Update(observations, now);
            for (int i = 0; i < _families.Count; i++)
            {
                Family family = _families[i];
                if (_globalPlaneLandmarks.TryGetAssignment(family.Id,
                        out int landmarkId))
                    family.TrackId = landmarkId;
            }
        }

        private void AppendLandmarkPatches(GlobalPlaneObservation observation,
            Vector3Int coord, Vector3 fittedCenter, int cellSizeVoxels,
            int witnessSubcell, int supportMask)
        {
            float span = cellSizeVoxels * _volume.VoxelSize;
            float halfWitness = Mathf.Max(
                GlobalPlaneLandmarkLayer.CellSizeMeters * 0.20f,
                span * 0.18f);
            if (supportMask == 0)
            {
                observation.Patches.Add(new GlobalPlanePatch(fittedCenter,
                    GlobalPlaneLandmarkLayer.CellSizeMeters * 0.20f));
                return;
            }

            Vector3Int vox = new Vector3Int(_volume.VoxelCount.x,
                _volume.VoxelCount.y, _volume.VoxelCount.z);
            Vector3 regionMinVoxel = new Vector3(coord.x * _tileSize,
                coord.y * _tileSize, coord.z * _tileSize);
            if (cellSizeVoxels < _tileSize)
            {
                regionMinVoxel += new Vector3(
                    (witnessSubcell & 1) != 0 ? cellSizeVoxels : 0,
                    (witnessSubcell & 2) != 0 ? cellSizeVoxels : 0,
                    (witnessSubcell & 4) != 0 ? cellSizeVoxels : 0);
            }
            Vector3 regionCenter = (regionMinVoxel +
                Vector3.one * (cellSizeVoxels * 0.5f) -
                new Vector3(vox.x, vox.y, vox.z) * 0.5f) * _volume.VoxelSize;
            for (int bit = 0; bit < 8; bit++)
            {
                if ((supportMask & (1 << bit)) == 0) continue;
                Vector3 offset = new Vector3(
                    (bit & 1) != 0 ? 0.25f : -0.25f,
                    (bit & 2) != 0 ? 0.25f : -0.25f,
                    (bit & 4) != 0 ? 0.25f : -0.25f) * span;
                observation.Patches.Add(new GlobalPlanePatch(
                    regionCenter + offset, halfWitness));
            }
        }

        private void RegisterTrackCompetition(Family family, PlaneTrack winner)
        {
            float cosCompetition = Mathf.Cos(
                PlaneTrackAngleDegrees * Mathf.Deg2Rad);
            foreach (PlaneTrack competitor in _planeTracks.Values)
            {
                if (competitor.Id == winner.Id ||
                    competitor.Horizontal != family.Horizontal ||
                    competitor.LastCompetitionEpoch == _planeEpoch) continue;
                float alignment = Vector3.Dot(competitor.Normal, family.Normal);
                float sign = alignment < 0f ? -1f : 1f;
                if (Mathf.Abs(alignment) < cosCompetition) continue;
                float offsetDelta = Mathf.Abs(
                    competitor.Offset - family.Offset * sign);
                float centerDelta = Mathf.Abs(
                    Vector3.Dot(competitor.Normal, family.Center) -
                    competitor.Offset);
                if (Mathf.Max(offsetDelta, centerDelta) >
                    PlaneTrackOffsetMeters) continue;
                if (DistanceFamilyToPlaneCoverage(competitor, family) >
                    PlaneCellSizeMeters * 2f) continue;
                competitor.CompetitionLosses = Mathf.Min(255,
                    competitor.CompetitionLosses + 1);
                competitor.LastCompetitionEpoch = _planeEpoch;
            }
        }

        private bool IsStrongTrackObservation(PlaneTrack plane, Family family)
        {
            // Strong means this family is good enough to have created a track
            // independently, and it agrees with the stricter plane equation.
            // Merely falling inside the broad association band is weak evidence.
            if (RequiredSeedHits(family) == int.MaxValue) return false;
            float alignment = Vector3.Dot(plane.Normal, family.Normal);
            float sign = alignment < 0f ? -1f : 1f;
            if (Mathf.Abs(alignment) < Mathf.Cos(
                    StrongPlaneTrackAngleDegrees * Mathf.Deg2Rad))
                return false;
            float offsetDelta = Mathf.Abs(
                plane.Offset - family.Offset * sign);
            float centerDelta = Mathf.Abs(
                Vector3.Dot(plane.Normal, family.Center) - plane.Offset);
            return Mathf.Max(offsetDelta, centerDelta) <=
                   StrongPlaneTrackOffsetMeters;
        }

        private static bool IsDurableRoomPlane(PlaneTrack plane)
        {
            if (plane.SeenEpochs < RoomPlaneEpochs) return false;
            int durableCells = 0;
            foreach (PlaneCell cell in plane.Cells.Values)
            {
                if (cell.Votes < DurableCellVotes) continue;
                durableCells++;
                if (durableCells >= RoomPlaneCellCount) return true;
            }
            return false;
        }

        private int RequiredSeedHits(Family family)
        {
            if (family.AngleJitter > StableAngleJitterDegrees ||
                family.OffsetJitter > StableOffsetJitterMeters)
                return int.MaxValue;
            int coarseMembers = 0;
            for (int i = 0; i < family.Members.Count; i++)
                if (family.Members[i].CellSizeVoxels >= _tileSize) coarseMembers++;
            if (coarseMembers >= 3 && family.SupportArea >= StrongBirthAreaMeters)
                return 2;
            if (family.Members.Count >= 2 &&
                family.SupportArea >= MediumBirthAreaMeters)
                return 4;
            if (family.Members.Count == 1)
            {
                TileTrack only = family.Members[0];
                if (only.CellSizeVoxels < _tileSize &&
                    family.SupportArea >= FineBirthAreaMeters &&
                    only.ResidualMeters <= _maxResidualMeters * 0.70f &&
                    only.AngleJitterDegrees <= 3f &&
                    only.OffsetJitterMeters <= 0.035f)
                    return 8;
            }
            return int.MaxValue;
        }

        private PlaneSeed UpdatePlaneSeed(Family family, float now)
        {
            if (!_planeSeeds.TryGetValue(family.Id, out PlaneSeed seed))
            {
                seed = new PlaneSeed
                {
                    FamilyId = family.Id,
                    Normal = family.Normal,
                    Offset = family.Offset,
                    Center = family.Center,
                    Horizontal = family.Horizontal,
                    Hits = 0,
                };
                _planeSeeds.Add(family.Id, seed);
            }
            float alignment = Vector3.Dot(seed.Normal, family.Normal);
            float sign = alignment < 0f ? -1f : 1f;
            float angle = Vector3.Angle(seed.Normal, family.Normal * sign);
            float offsetDelta = Mathf.Abs(seed.Offset - family.Offset * sign);
            if (seed.Horizontal != family.Horizontal ||
                angle > PlaneTrackAngleDegrees ||
                offsetDelta > PlaneTrackOffsetMeters)
            {
                seed.Normal = family.Normal;
                seed.Offset = family.Offset;
                seed.Center = family.Center;
                seed.Horizontal = family.Horizontal;
                seed.Hits = 0;
            }
            else
            {
                seed.Normal = Vector3.Lerp(seed.Normal,
                    family.Normal * sign, 0.20f).normalized;
                seed.Offset = Mathf.Lerp(seed.Offset, family.Offset * sign, 0.20f);
                seed.Center = Vector3.Lerp(seed.Center, family.Center, 0.20f);
            }
            if (seed.LastSeenEpoch != _planeEpoch)
            {
                seed.Hits = Mathf.Min(255, seed.Hits + 1);
                seed.LastSeenEpoch = _planeEpoch;
            }
            seed.LastSeen = now;
            return seed;
        }

        private void MergeCompatiblePlaneTracks()
        {
            bool merged;
            do
            {
                merged = false;
                var planes = new List<PlaneTrack>(_planeTracks.Values);
                planes.Sort((a, b) => a.Id.CompareTo(b.Id));
                float cosStrong = Mathf.Cos(
                    StrongPlaneTrackAngleDegrees * Mathf.Deg2Rad);
                float cosMatch = Mathf.Cos(PlaneTrackAngleDegrees * Mathf.Deg2Rad);
                for (int i = 0; i < planes.Count && !merged; i++)
                {
                    PlaneTrack keep = planes[i];
                    if (!_planeTracks.ContainsKey(keep.Id)) continue;
                    for (int j = i + 1; j < planes.Count; j++)
                    {
                        PlaneTrack remove = planes[j];
                        if (!_planeTracks.ContainsKey(remove.Id) ||
                            keep.Horizontal != remove.Horizontal) continue;
                        float alignment = Vector3.Dot(keep.Normal, remove.Normal);
                        float sign = alignment < 0f ? -1f : 1f;
                        float absAlignment = Mathf.Abs(alignment);
                        float offsetDelta = Mathf.Abs(
                            keep.Offset - remove.Offset * sign);
                        if (absAlignment < cosMatch ||
                            offsetDelta > PlaneTrackOffsetMeters) continue;

                        // Plane identity and coverage connectivity are separate.
                        // A highly consistent equation is the same global plane
                        // even when its observed coverage is split by a doorway,
                        // furniture or an unscanned gap. Borderline equations
                        // still need spatial continuity so folds cannot chain.
                        bool strongSamePlane = absAlignment >= cosStrong &&
                            offsetDelta <= StrongPlaneTrackOffsetMeters;
                        if (!strongSamePlane &&
                            DistanceBetweenPlaneCoverages(keep, remove) >
                            PlaneTrackMergeGapMeters)
                            continue;
                        MergePlaneTrackInto(keep, remove);
                        merged = true;
                        break;
                    }
                }
            } while (merged);
        }

        private void MergePlaneTrackInto(PlaneTrack keep, PlaneTrack remove)
        {
            foreach (KeyValuePair<long, PlaneCell> pair in remove.Cells)
            {
                DecodePlaneCellKey(pair.Key, out int sourceU, out int sourceV);
                Vector3 point = remove.Normal * remove.Offset +
                    remove.U * ((sourceU + 0.5f) * PlaneCellSizeMeters) +
                    remove.V * ((sourceV + 0.5f) * PlaneCellSizeMeters);
                int targetU = Mathf.FloorToInt(Vector3.Dot(point, keep.U) /
                                               PlaneCellSizeMeters);
                int targetV = Mathf.FloorToInt(Vector3.Dot(point, keep.V) /
                                               PlaneCellSizeMeters);
                long targetKey = PlaneCellKey(targetU, targetV);
                if (!keep.Cells.TryGetValue(targetKey, out PlaneCell target))
                {
                    target = new PlaneCell();
                    keep.Cells.Add(targetKey, target);
                }
                PlaneCell source = pair.Value;
                target.Votes = Mathf.Max(target.Votes, source.Votes);
                target.LastSeen = Mathf.Max(target.LastSeen, source.LastSeen);
                target.LastEpoch = Mathf.Max(target.LastEpoch, source.LastEpoch);
                ExpandPlaneBounds(keep, targetU, targetV);
            }
            keep.SeenEpochs = Mathf.Max(keep.SeenEpochs, remove.SeenEpochs);
            keep.LastSeen = Mathf.Max(keep.LastSeen, remove.LastSeen);
            keep.LastSeenEpoch = Mathf.Max(keep.LastSeenEpoch, remove.LastSeenEpoch);
            keep.EvidenceScore = Mathf.Max(keep.EvidenceScore,
                remove.EvidenceScore);
            keep.WeakEvidenceStreak = Mathf.Min(keep.WeakEvidenceStreak,
                remove.WeakEvidenceStreak);
            keep.CompetitionLosses = Mathf.Min(keep.CompetitionLosses,
                remove.CompetitionLosses);
            for (int i = 0; i < _families.Count; i++)
            {
                Family family = _families[i];
                if (family.TrackId == remove.Id) family.TrackId = keep.Id;
            }
            _planeTracks.Remove(remove.Id);
            _paletteSlotByFamily.Remove(remove.Id);
        }

        private static float DistanceBetweenPlaneCoverages(PlaneTrack a,
            PlaneTrack b)
        {
            PlaneTrack source = a.Cells.Count <= b.Cells.Count ? a : b;
            PlaneTrack target = source == a ? b : a;
            if (source.Cells.Count == 0 || target.Cells.Count == 0) return float.MaxValue;
            float best = float.MaxValue;
            foreach (long key in source.Cells.Keys)
            {
                DecodePlaneCellKey(key, out int u, out int v);
                Vector3 point = source.Normal * source.Offset +
                    source.U * ((u + 0.5f) * PlaneCellSizeMeters) +
                    source.V * ((v + 0.5f) * PlaneCellSizeMeters);
                best = Mathf.Min(best, DistanceToPlaneCoverage(target, point));
                if (best <= PlaneCellSizeMeters) break;
            }
            return best;
        }

        private static void ExpandPlaneBounds(PlaneTrack plane, int u, int v)
        {
            if (!plane.HasBounds)
            {
                plane.MinU = plane.MaxU = u;
                plane.MinV = plane.MaxV = v;
                plane.HasBounds = true;
                return;
            }
            plane.MinU = Mathf.Min(plane.MinU, u);
            plane.MaxU = Mathf.Max(plane.MaxU, u);
            plane.MinV = Mathf.Min(plane.MinV, v);
            plane.MaxV = Mathf.Max(plane.MaxV, v);
        }

        private static void RebuildPlaneBounds(PlaneTrack plane)
        {
            plane.HasBounds = false;
            foreach (long key in plane.Cells.Keys)
            {
                DecodePlaneCellKey(key, out int u, out int v);
                ExpandPlaneBounds(plane, u, v);
            }
        }

        private PlaneTrack FindBestPlaneTrack(Family family)
        {
            float cosStrong = Mathf.Cos(
                StrongPlaneTrackAngleDegrees * Mathf.Deg2Rad);
            float cosMatch = Mathf.Cos(PlaneTrackAngleDegrees * Mathf.Deg2Rad);
            PlaneTrack best = null;
            float bestScore = float.MaxValue;
            foreach (PlaneTrack plane in _planeTracks.Values)
            {
                if (plane.Horizontal != family.Horizontal) continue;
                float alignment = Vector3.Dot(plane.Normal, family.Normal);
                float sign = alignment < 0f ? -1f : 1f;
                float absAlignment = Mathf.Abs(alignment);
                if (absAlignment < cosMatch) continue;
                float offsetDelta = Mathf.Abs(plane.Offset - family.Offset * sign);
                float centerDelta = Mathf.Abs(
                    Vector3.Dot(plane.Normal, family.Center) - plane.Offset);
                float planeDelta = Mathf.Max(offsetDelta, centerDelta);
                if (planeDelta > PlaneTrackOffsetMeters) continue;
                float gap = DistanceFamilyToPlaneCoverage(plane, family);
                bool strongSamePlane = absAlignment >= cosStrong &&
                    planeDelta <= StrongPlaneTrackOffsetMeters;
                if (!strongSamePlane && gap > PlaneTrackGapMeters) continue;
                float angle = Mathf.Acos(Mathf.Clamp(absAlignment, -1f, 1f));
                // Always prefer an equation-level match over a merely nearby
                // borderline one. Gap only breaks ties inside the same class.
                float score = (strongSamePlane ? 0f : 1f) + planeDelta +
                    Mathf.Min(gap, PlaneTrackGapMeters) * 0.05f + angle * 0.03f;
                if (score >= bestScore) continue;
                bestScore = score;
                best = plane;
            }
            return best;
        }

        private static float DistanceFamilyToPlaneCoverage(PlaneTrack plane,
            Family family)
        {
            if (plane.Cells.Count == 0) return 0f;
            float best = DistanceToPlaneCoverage(plane, family.Center);
            for (int i = 0; i < family.Members.Count; i++)
                best = Mathf.Min(best,
                    DistanceToPlaneCoverage(plane, family.Members[i].CenterLocal));
            for (int i = 0; i < family.Extent.Count; i++)
                best = Mathf.Min(best,
                    DistanceToPlaneCoverage(plane, family.Extent[i].CenterLocal));
            return best;
        }

        private PlaneTrack CreatePlaneTrack(Family family, float now)
        {
            Vector3 normal = family.Normal.normalized;
            CanonicalizeNormal(ref normal);
            Vector3 u = family.Horizontal
                ? Vector3.right
                : Vector3.Cross(Vector3.up, normal).normalized;
            if (u.sqrMagnitude < 0.5f) u = Vector3.forward;
            Vector3 v = Vector3.Cross(normal, u).normalized;
            return new PlaneTrack
            {
                Id = _nextPlaneTrackId++,
                Normal = normal,
                Offset = Vector3.Dot(normal, family.Center),
                U = u,
                V = v,
                Horizontal = family.Horizontal,
                SeenEpochs = 0,
                LastSeen = now,
                EvidenceScore = InitialTrackEvidence,
            };
        }

        private void StampFamilyEvidence(PlaneTrack plane, Family family, float now)
        {
            for (int i = 0; i < family.Members.Count; i++)
            {
                TileTrack member = family.Members[i];
                StampSupportMask(plane, member.Coord, member.CenterLocal,
                    member.CellSizeVoxels, member.WitnessSubcell,
                    member.SupportMask, now);
            }
            for (int i = 0; i < family.Extent.Count; i++)
            {
                TileEvidence evidence = family.Extent[i];
                StampSupportMask(plane, evidence.Coord, evidence.CenterLocal,
                    evidence.CellSizeVoxels, evidence.WitnessSubcell,
                    evidence.SupportMask, now);
            }
        }

        private void StampSupportMask(PlaneTrack plane, Vector3Int coord,
            Vector3 fittedCenter, int cellSizeVoxels, int witnessSubcell,
            int supportMask, float now)
        {
            float span = cellSizeVoxels * _volume.VoxelSize;
            float halfWitness = Mathf.Max(PlaneCellSizeMeters * 0.20f, span * 0.18f);
            Vector3Int vox = new Vector3Int(_volume.VoxelCount.x,
                _volume.VoxelCount.y, _volume.VoxelCount.z);
            Vector3 regionMinVoxel = new Vector3(coord.x * _tileSize,
                coord.y * _tileSize, coord.z * _tileSize);
            if (cellSizeVoxels < _tileSize)
            {
                regionMinVoxel += new Vector3(
                    (witnessSubcell & 1) != 0 ? cellSizeVoxels : 0,
                    (witnessSubcell & 2) != 0 ? cellSizeVoxels : 0,
                    (witnessSubcell & 4) != 0 ? cellSizeVoxels : 0);
            }
            Vector3 regionCenter = (regionMinVoxel +
                Vector3.one * (cellSizeVoxels * 0.5f) -
                new Vector3(vox.x, vox.y, vox.z) * 0.5f) * _volume.VoxelSize;

            if (supportMask == 0)
            {
                StampPlanePatch(plane, fittedCenter,
                    PlaneCellSizeMeters * 0.20f, now);
                return;
            }
            for (int bit = 0; bit < 8; bit++)
            {
                if ((supportMask & (1 << bit)) == 0) continue;
                Vector3 offset = new Vector3(
                    (bit & 1) != 0 ? 0.25f : -0.25f,
                    (bit & 2) != 0 ? 0.25f : -0.25f,
                    (bit & 4) != 0 ? 0.25f : -0.25f) * span;
                StampPlanePatch(plane, regionCenter + offset, halfWitness, now);
            }
        }

        private void StampPlanePatch(PlaneTrack plane, Vector3 center, float half,
            float now)
        {
            Vector3 onPlane = center - plane.Normal *
                (Vector3.Dot(plane.Normal, center) - plane.Offset);
            float centerU = Vector3.Dot(onPlane, plane.U);
            float centerV = Vector3.Dot(onPlane, plane.V);
            int minU = Mathf.FloorToInt((centerU - half) / PlaneCellSizeMeters);
            int maxU = Mathf.FloorToInt((centerU + half - 0.001f) / PlaneCellSizeMeters);
            int minV = Mathf.FloorToInt((centerV - half) / PlaneCellSizeMeters);
            int maxV = Mathf.FloorToInt((centerV + half - 0.001f) / PlaneCellSizeMeters);
            for (int v = minV; v <= maxV; v++)
            {
                for (int u = minU; u <= maxU; u++)
                {
                    long key = PlaneCellKey(u, v);
                    if (!plane.Cells.TryGetValue(key, out PlaneCell cell))
                    {
                        cell = new PlaneCell();
                        plane.Cells.Add(key, cell);
                    }
                    if (cell.LastEpoch != _planeEpoch)
                    {
                        cell.Votes = Mathf.Min(255, cell.Votes + 1);
                        cell.LastEpoch = _planeEpoch;
                    }
                    cell.LastSeen = now;
                }
            }
            if (!plane.HasBounds)
            {
                plane.MinU = minU;
                plane.MaxU = maxU;
                plane.MinV = minV;
                plane.MaxV = maxV;
                plane.HasBounds = true;
            }
            else
            {
                plane.MinU = Mathf.Min(plane.MinU, minU);
                plane.MaxU = Mathf.Max(plane.MaxU, maxU);
                plane.MinV = Mathf.Min(plane.MinV, minV);
                plane.MaxV = Mathf.Max(plane.MaxV, maxV);
            }
        }

        private static bool HasVisibleCells(PlaneTrack plane)
        {
            if (plane.SeenEpochs < VisiblePlaneEpochs) return false;
            foreach (PlaneCell cell in plane.Cells.Values)
                if (cell.Votes >= VisibleCellVotes) return true;
            return false;
        }

        private static float DistanceToPlaneCoverage(PlaneTrack plane, Vector3 point)
        {
            if (plane.Cells.Count == 0) return 0f;
            float u = Vector3.Dot(point, plane.U) / PlaneCellSizeMeters;
            float v = Vector3.Dot(point, plane.V) / PlaneCellSizeMeters;
            float bestSq = float.MaxValue;
            foreach (long key in plane.Cells.Keys)
            {
                DecodePlaneCellKey(key, out int cellU, out int cellV);
                float du = u < cellU ? cellU - u :
                    (u > cellU + 1 ? u - (cellU + 1) : 0f);
                float dv = v < cellV ? cellV - v :
                    (v > cellV + 1 ? v - (cellV + 1) : 0f);
                bestSq = Mathf.Min(bestSq, du * du + dv * dv);
            }
            return Mathf.Sqrt(bestSq) * PlaneCellSizeMeters;
        }

        private void AccumulateFamily(Family family)
        {
            TileTrack first = family.Members[0];
            Vector3 normalSum = Vector3.zero;
            Vector3 centerSum = Vector3.zero;
            float offsetSum = 0f;
            float angleSum = 0f;
            float driftSum = 0f;
            float freeSum = 0f;
            int sampleSum = 0;
            Bounds bounds = new Bounds(first.CenterLocal, Vector3.zero);
            for (int i = 0; i < family.Members.Count; i++)
            {
                TileTrack member = family.Members[i];
                Vector3 n = member.Normal;
                float memberOffset = member.Offset;
                if (Vector3.Dot(n, first.Normal) < 0f)
                {
                    n = -n;
                    memberOffset = -memberOffset;
                }
                normalSum += n;
                centerSum += member.CenterLocal;
                offsetSum += memberOffset;
                angleSum += member.AngleJitterDegrees;
                driftSum += member.OffsetJitterMeters;
                freeSum += member.FreeSpaceScore;
                sampleSum += member.SurfaceSamples;
                bounds.Encapsulate(member.CenterLocal);
            }
            float inv = 1f / family.Members.Count;
            family.Normal = normalSum.normalized;
            family.Center = centerSum * inv;
            family.Offset = offsetSum * inv;
            family.AngleJitter = angleSum * inv;
            family.OffsetJitter = driftSum * inv;
            family.FreeSpaceScore = freeSum * inv;
            family.Horizontal = Mathf.Abs(family.Normal.y) >=
                                Mathf.Cos(StructuralAngleDegrees * Mathf.Deg2Rad);
            family.Bounds = bounds;
            float bandCells = Mathf.Max(1f,
                1.2f * _volume.VoxelDistance / Mathf.Max(0.001f, _volume.VoxelSize));
            family.SupportArea = sampleSum * _volume.VoxelSize * _volume.VoxelSize / bandCells;
        }

        private void UpdateIntersections(float now)
        {
            var seen = new HashSet<long>();
            for (int i = 0; i < _families.Count; i++)
            {
                Family a = _families[i];
                if (!IsMatureFamily(a) || a.TrackId <= 0) continue;
                for (int j = i + 1; j < _families.Count; j++)
                {
                    Family b = _families[j];
                    if (!IsMatureFamily(b) || b.TrackId <= 0) continue;
                    if (a.TrackId == b.TrackId) continue;
                    float nd = Mathf.Abs(Vector3.Dot(a.Normal, b.Normal));
                    if (nd > 0.35f || !ExpandedBoundsOverlap(a.Bounds, b.Bounds)) continue;
                    Vector3 lineRaw = Vector3.Cross(a.Normal, b.Normal);
                    float denom = lineRaw.sqrMagnitude;
                    if (denom < 0.25f) continue;
                    Vector3 anchor = (a.Offset * Vector3.Cross(b.Normal, lineRaw) +
                                      b.Offset * Vector3.Cross(lineRaw, a.Normal)) / denom;
                    Vector3 direction = lineRaw.normalized;
                    CanonicalizeNormal(ref direction);
                    long key = PairKey(a.TrackId, b.TrackId);
                    seen.Add(key);
                    if (!_intersections.TryGetValue(key, out IntersectionTrack track))
                    {
                        track = new IntersectionTrack
                        {
                            Anchor = anchor,
                            Direction = direction,
                            Hits = 1,
                            LastSeen = now,
                        };
                        _intersections.Add(key, track);
                    }
                    else
                    {
                        float drift = Vector3.Distance(track.Anchor, anchor);
                        float angle = Vector3.Angle(track.Direction, direction);
                        track.DriftMeters = track.Hits <= 1
                            ? drift
                            : Mathf.Lerp(track.DriftMeters, drift, 0.35f);
                        track.AngleJitterDegrees = track.Hits <= 1
                            ? angle
                            : Mathf.Lerp(track.AngleJitterDegrees, angle, 0.35f);
                        track.Anchor = Vector3.Lerp(track.Anchor, anchor, 0.35f);
                        track.Direction = Vector3.Lerp(track.Direction, direction, 0.35f).normalized;
                        track.Hits = Mathf.Min(255, track.Hits + 1);
                        track.LastSeen = now;
                    }
                }
            }
            var stale = new List<long>();
            foreach (KeyValuePair<long, IntersectionTrack> pair in _intersections)
                if (!seen.Contains(pair.Key) && now - pair.Value.LastSeen > 2f) stale.Add(pair.Key);
            for (int i = 0; i < stale.Count; i++) _intersections.Remove(stale[i]);
        }

        private void UpdateStatus(int candidates, float now)
        {
            int stableTiles = 0;
            int fineStableTiles = 0;
            int matureFamilies = 0;
            int visiblePlaneTracks = 0;
            int confirmedPlaneTracks = 0;
            int durableRoomPlanes = 0;
            int dormantPlaneTracks = 0;
            int visiblePlaneCells = 0;
            int walls = 0;
            int horizontals = 0;
            float angle = 0f;
            float drift = 0f;
            float free = 0f;
            int extentTiles = 0;
            foreach (TileTrack track in _tracks.Values)
            {
                if (!track.Stable(now)) continue;
                stableTiles++;
                if (track.CellSizeVoxels < _tileSize) fineStableTiles++;
            }
            for (int i = 0; i < _families.Count; i++)
            {
                Family family = _families[i];
                if (!IsMatureFamily(family)) continue;
                matureFamilies++;
                angle += family.AngleJitter;
                drift += family.OffsetJitter;
                free += family.FreeSpaceScore;
                extentTiles += family.Extent.Count;
            }
            foreach (GlobalPlaneLandmark plane in _globalPlaneLandmarks.Landmarks)
            {
                if (plane.State == GlobalPlaneLandmarkState.Confirmed)
                    confirmedPlaneTracks++;
                else if (plane.State == GlobalPlaneLandmarkState.Dormant)
                    dormantPlaneTracks++;
                if (_globalPlaneLandmarks.IsStructural(plane))
                    durableRoomPlanes++;
                if (!_globalPlaneLandmarks.IsRecentlyVisible(plane, now))
                    continue;
                visiblePlaneTracks++;
                if (plane.Horizontal) horizontals++; else walls++;
                visiblePlaneCells += _globalPlaneLandmarks.CountCellsWithVotes(
                    plane, 4);
            }
            float support = visiblePlaneCells *
                GlobalPlaneLandmarkLayer.CellSizeMeters *
                GlobalPlaneLandmarkLayer.CellSizeMeters;
            int stableIntersections = 0;
            float intersectionDrift = 0f;
            foreach (IntersectionTrack track in _intersections.Values)
            {
                if (track.Hits < 2 || now - track.LastSeen > 1.25f ||
                    track.AngleJitterDegrees > StableAngleJitterDegrees) continue;
                stableIntersections++;
                intersectionDrift += track.DriftMeters;
            }
            float invFamilies = matureFamilies > 0 ? 1f / matureFamilies : 0f;
            float invIntersections = stableIntersections > 0 ? 1f / stableIntersections : 0f;
            string faultSuffix = string.IsNullOrEmpty(_fault) ? string.Empty : " !" + _fault;
            StatusCompact =
                $"地标只读｜候{candidates} 稳{stableTiles} 片{matureFamilies} " +
                $"观{_globalPlaneLandmarks.LastObservationCount} 芽{_globalPlaneLandmarks.SeedCount}→标{visiblePlaneTracks}/确{confirmedPlaneTracks}/固{durableRoomPlanes} " +
                $"眠{dormantPlaneTracks} 库{_globalPlaneLandmarks.RetainedCount} " +
                $"退{_globalPlaneLandmarks.RejectedThisEpoch}（墙{walls}/横{horizontals}）{faultSuffix}\n" +
                $"证据｜支{support:0.0}m² 漂{angle * invFamilies:0.0}°/{drift * invFamilies * 1000f:0}mm " +
                $"格{visiblePlaneCells} 扩{extentTiles} 空{free * invFamilies * 100f:0}% 交{stableIntersections}/{intersectionDrift * invIntersections * 1000f:0}mm";
        }

        private void UpdateVisuals(float now)
        {
            EnsureVisuals();
            if (_visualRoot == null) return;
            AssignActivePaletteSlots();
            for (int i = 0; i < PaletteSize; i++)
            {
                _visualVertices[i].Clear();
                _visualIndices[i].Clear();
            }
            float width = Mathf.Clamp(_volume.VoxelSize * 0.08f, 0.006f, 0.015f);
            float halfCell = GlobalPlaneLandmarkLayer.CellSizeMeters * 0.5f;
            foreach (GlobalPlaneLandmark plane in _globalPlaneLandmarks.Landmarks)
            {
                // The validation view is intentionally stricter than the
                // tracker. Pending/merely-visible tracks stay invisible: the
                // overlay must prove a durable room plane, not expose every
                // internal hypothesis as the retired rectangle view did.
                if (!_globalPlaneLandmarks.IsDisplayable(plane)) continue;
                int palette = _paletteSlotByFamily[plane.Id];
                var visible = new HashSet<long>();
                foreach (KeyValuePair<long, GlobalPlaneLandmarkCell> pair in
                         plane.Cells)
                    if (pair.Value.Votes >=
                        GlobalPlaneLandmarkLayer.DisplayCellVotes)
                        visible.Add(pair.Key);
                Vector3 origin = plane.Normal * plane.Offset;
                foreach (long key in visible)
                {
                    GlobalPlaneLandmarkLayer.DecodeCellKey(key,
                        out int cellU, out int cellV);
                    Vector3 center = origin + plane.U * ((cellU + 0.5f) *
                                     GlobalPlaneLandmarkLayer.CellSizeMeters) +
                                     plane.V * ((cellV + 0.5f) *
                                     GlobalPlaneLandmarkLayer.CellSizeMeters);
                    Vector3 p0 = center - plane.U * halfCell - plane.V * halfCell;
                    Vector3 p1 = center + plane.U * halfCell - plane.V * halfCell;
                    Vector3 p2 = center + plane.U * halfCell + plane.V * halfCell;
                    Vector3 p3 = center - plane.U * halfCell + plane.V * halfCell;
                    AddQuad(_visualVertices[palette], _visualIndices[palette],
                        p0, p1, p2, p3);
                    if (!visible.Contains(GlobalPlaneLandmarkLayer.CellKey(
                            cellU, cellV - 1)))
                        AddRibbon(_visualVertices[palette], _visualIndices[palette],
                            p0, p1, plane.V, width);
                    if (!visible.Contains(GlobalPlaneLandmarkLayer.CellKey(
                            cellU + 1, cellV)))
                        AddRibbon(_visualVertices[palette], _visualIndices[palette],
                            p1, p2, plane.U, width);
                    if (!visible.Contains(GlobalPlaneLandmarkLayer.CellKey(
                            cellU, cellV + 1)))
                        AddRibbon(_visualVertices[palette], _visualIndices[palette],
                            p2, p3, plane.V, width);
                    if (!visible.Contains(GlobalPlaneLandmarkLayer.CellKey(
                            cellU - 1, cellV)))
                        AddRibbon(_visualVertices[palette], _visualIndices[palette],
                            p3, p0, plane.U, width);
                }
            }
            for (int i = 0; i < PaletteSize; i++)
            {
                Mesh mesh = _visualMeshes[i];
                mesh.Clear(false);
                mesh.SetVertices(_visualVertices[i]);
                mesh.SetTriangles(_visualIndices[i], 0, false);
                mesh.RecalculateBounds();
            }
        }

        private void EnsureVisuals()
        {
            if (_visualRoot != null) return;
            // Runtime Shader.Find targets can be stripped from Android builds.
            // The dedicated Resources asset is packaged deterministically and
            // renders above the production mesh without sharing its materials.
            Shader shader = Resources.Load<Shader>("RoomStructureProbeOverlay");
            if (shader == null)
            {
                _fault = "NOSHADER";
                return;
            }
            if (_fault == "NOSHADER") _fault = string.Empty;
            _visualRoot = new GameObject("RoomStructureFeasibilityProbe_ReadOnly");
            _visualRoot.transform.SetParent(_volumeTransform, false);
            _visualRoot.SetActive(_showOverlay);
            for (int i = 0; i < PaletteSize; i++)
            {
                var child = new GameObject("FamilyPalette_" + i);
                child.transform.SetParent(_visualRoot.transform, false);
                var filter = child.AddComponent<MeshFilter>();
                var renderer = child.AddComponent<MeshRenderer>();
                var mesh = new Mesh
                {
                    name = "RoomStructureProbeMesh_" + i,
                    indexFormat = IndexFormat.UInt32,
                };
                var material = new Material(shader) { name = "RoomStructureProbeMaterial_" + i };
                Color overlayColor = Palette[i];
                overlayColor.a = 0.22f;
                if (material.HasProperty("_LineColor")) material.SetColor("_LineColor", overlayColor);
                if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", Palette[i]);
                if (material.HasProperty("_Color")) material.SetColor("_Color", Palette[i]);
                if (material.HasProperty("_Cull")) material.SetFloat("_Cull", 0f);
                filter.sharedMesh = mesh;
                renderer.sharedMaterial = material;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
                _visualMeshes[i] = mesh;
                _visualMaterials[i] = material;
                _visualVertices[i] = new List<Vector3>(512);
                _visualIndices[i] = new List<int>(768);
            }
        }

        private Vector3Int DecodeCoord(int index)
        {
            return new Vector3Int(
                index % _tileCount.x,
                (index / _tileCount.x) % _tileCount.y,
                index / (_tileCount.x * _tileCount.y));
        }

        private static bool AreNeighbourTiles(Vector3Int a, Vector3Int b)
        {
            Vector3Int d = a - b;
            return Mathf.Abs(d.x) <= 1 && Mathf.Abs(d.y) <= 1 && Mathf.Abs(d.z) <= 1;
        }

        private static void CanonicalizeNormal(ref Vector3 n)
        {
            Vector3 a = new Vector3(Mathf.Abs(n.x), Mathf.Abs(n.y), Mathf.Abs(n.z));
            float dominant = a.x >= a.y && a.x >= a.z ? n.x : (a.y >= a.z ? n.y : n.z);
            if (dominant < 0f) n = -n;
        }

        private static int Find(int[] parent, int i)
        {
            while (parent[i] != i)
            {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            return i;
        }

        private static void Union(int[] parent, int a, int b)
        {
            int rootA = Find(parent, a);
            int rootB = Find(parent, b);
            if (rootA != rootB) parent[rootB] = rootA;
        }

        private int PickInheritedFamilyId(List<TileTrack> members, HashSet<int> used)
        {
            var votes = new Dictionary<int, int>();
            for (int i = 0; i < members.Count; i++)
            {
                int id = members[i].FamilyId;
                if (id <= 0 || used.Contains(id)) continue;
                votes.TryGetValue(id, out int count);
                votes[id] = count + 1;
            }
            int bestId = 0;
            int bestVotes = 0;
            foreach (KeyValuePair<int, int> vote in votes)
            {
                if (vote.Value <= bestVotes) continue;
                bestVotes = vote.Value;
                bestId = vote.Key;
            }
            return bestId;
        }

        private bool ExpandedBoundsOverlap(Bounds a, Bounds b)
        {
            float expand = _tileSize * _volume.VoxelSize * 1.5f;
            a.Expand(expand * 2f);
            b.Expand(expand * 2f);
            return a.Intersects(b);
        }

        private static long PairKey(int a, int b)
        {
            uint lo = (uint)Mathf.Min(a, b);
            uint hi = (uint)Mathf.Max(a, b);
            return ((long)hi << 32) | lo;
        }

        private static long PlaneCellKey(int u, int v)
        {
            return ((long)(uint)u << 32) | (uint)v;
        }

        private static void DecodePlaneCellKey(long key, out int u, out int v)
        {
            u = unchecked((int)(uint)(key >> 32));
            v = unchecked((int)(uint)key);
        }

        private static int PositiveModulo(int value, int modulus)
        {
            int result = value % modulus;
            return result < 0 ? result + modulus : result;
        }

        private bool IsMatureFamily(Family family)
        {
            if (family.Members.Count >= 2) return true;
            if (family.Members.Count != 1) return false;
            TileTrack only = family.Members[0];
            return only.CellSizeVoxels < _tileSize &&
                   only.GoodStreak >= StableHitCount + 2 &&
                   only.ResidualMeters <= _maxResidualMeters * 0.8f;
        }

        private void AssignActivePaletteSlots()
        {
            var activeIds = new HashSet<int>();
            foreach (GlobalPlaneLandmark plane in _globalPlaneLandmarks.Landmarks)
                if (_globalPlaneLandmarks.IsDisplayable(plane))
                    activeIds.Add(plane.Id);

            var staleIds = new List<int>();
            foreach (KeyValuePair<int, int> pair in _paletteSlotByFamily)
                if (!activeIds.Contains(pair.Key)) staleIds.Add(pair.Key);
            for (int i = 0; i < staleIds.Count; i++)
                _paletteSlotByFamily.Remove(staleIds[i]);

            var used = new bool[PaletteSize];
            foreach (int slot in _paletteSlotByFamily.Values)
                if (slot >= 0 && slot < PaletteSize) used[slot] = true;
            foreach (int id in activeIds)
            {
                if (_paletteSlotByFamily.ContainsKey(id)) continue;
                int slot = 0;
                while (slot < PaletteSize && used[slot]) slot++;
                if (slot >= PaletteSize) slot = PositiveModulo(id, PaletteSize);
                _paletteSlotByFamily[id] = slot;
                used[slot] = true;
            }
        }

        private static Color[] BuildPalette()
        {
            var colors = new Color[PaletteSize];
            for (int i = 0; i < colors.Length; i++)
            {
                float hue = Mathf.Repeat(0.08f + i * 0.61803398875f, 1f);
                float saturation = 0.72f + 0.22f * ((i % 3) / 2f);
                float value = 0.88f + 0.12f * (i % 2);
                colors[i] = Color.HSVToRGB(hue, saturation, value);
                colors[i].a = 1f;
            }
            return colors;
        }

        private static void AddRibbon(List<Vector3> vertices, List<int> indices,
            Vector3 a, Vector3 b, Vector3 side, float width)
        {
            Vector3 halfSide = side.normalized * (width * 0.5f);
            int start = vertices.Count;
            vertices.Add(a - halfSide);
            vertices.Add(a + halfSide);
            vertices.Add(b + halfSide);
            vertices.Add(b - halfSide);
            indices.Add(start);
            indices.Add(start + 1);
            indices.Add(start + 2);
            indices.Add(start);
            indices.Add(start + 2);
            indices.Add(start + 3);
        }

        private static void AddQuad(List<Vector3> vertices, List<int> indices,
            Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3)
        {
            int start = vertices.Count;
            vertices.Add(p0);
            vertices.Add(p1);
            vertices.Add(p2);
            vertices.Add(p3);
            indices.Add(start);
            indices.Add(start + 1);
            indices.Add(start + 2);
            indices.Add(start);
            indices.Add(start + 2);
            indices.Add(start + 3);
        }

        private void ReleaseGpuResources()
        {
            _records?.Release();
            _records = null;
            _pending = false;
        }

        internal void Reset()
        {
            _generation++;
            _tracks.Clear();
            _currentEvidence.Clear();
            _paletteSlotByFamily.Clear();
            _families.Clear();
            _planeTracks.Clear();
            _planeSeeds.Clear();
            _globalPlaneLandmarks.Reset();
            _planeEpoch = 0;
            _nextPlaneTrackId = 1;
            _lastObservedIntegrationCount = -1;
            _intersections.Clear();
            StatusCompact = "骨架只读: 等待TSDF";
            if (_visualRoot != null)
            {
                _visualRoot.SetActive(false);
                for (int i = 0; i < PaletteSize; i++)
                    _visualMeshes[i]?.Clear(false);
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _generation++;
            ReleaseGpuResources();
            if (_visualRoot != null) UnityEngine.Object.Destroy(_visualRoot);
            _visualRoot = null;
            for (int i = 0; i < PaletteSize; i++)
            {
                if (_visualMaterials[i] != null) UnityEngine.Object.Destroy(_visualMaterials[i]);
                if (_visualMeshes[i] != null) UnityEngine.Object.Destroy(_visualMeshes[i]);
                _visualMaterials[i] = null;
                _visualMeshes[i] = null;
            }
            _tracks.Clear();
            _currentEvidence.Clear();
            _paletteSlotByFamily.Clear();
            _families.Clear();
            _planeTracks.Clear();
            _planeSeeds.Clear();
            _globalPlaneLandmarks.Dispose();
            _intersections.Clear();
        }
    }
}
