// Copyright (c) Meta Platforms, Inc. and affiliates.

using System;
using System.Collections.Generic;
using Meta.XR;
using Meta.XR.Samples;
using UnityEngine;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    [DisallowMultipleComponent]
    [MetaCodeSample("MRMotifs-InstantContentPlacement")]
    public class DepthEffectsGridGuidedCoverageSampler : MonoBehaviour
    {
        public struct DisplayCell
        {
            public Vector3 worldPos;
            public Vector3 normal;
            public Vector3 tangent;
            public Vector3 bitangent;
            public int u;
            public int v;
            public int patchHits;
            public float lastSeenTime;
        }

        private struct SeedCellState
        {
            public float lastSampleTime;
            public int stableHitCount;
        }

        private struct PatchCellKey : IEquatable<PatchCellKey>
        {
            public int u;
            public int v;

            public bool Equals(PatchCellKey other)
            {
                return u == other.u && v == other.v;
            }

            public override bool Equals(object obj)
            {
                return obj is PatchCellKey other && Equals(other);
            }

            public override int GetHashCode()
            {
                unchecked
                {
                    return (u * 397) ^ v;
                }
            }
        }

        private sealed class SurfacePatch
        {
            public Vector3 anchor;
            public Vector3 normal;
            public Vector3 tangent;
            public Vector3 bitangent;
            public float lastSeenTime;
            public int seedHits;
            public int spiralCursor;
            public readonly Dictionary<PatchCellKey, float> cellLastSampleTimes = new Dictionary<PatchCellKey, float>(64);
            public readonly HashSet<PatchCellKey> confirmedCells = new HashSet<PatchCellKey>();
        }

        [Header("Refs")]
        [SerializeField]
        private EnvironmentRaycastManager environmentRaycastManager;

        [SerializeField]
        private Camera sampleCamera;

        [SerializeField]
        private ScanCoverVoxelPointAccumulator voxelPointAccumulator;

        [Header("Seed Window")]
        [SerializeField]
        private float viewportCenterX = 0.5f;

        [SerializeField]
        private float viewportCenterY = 0.55f;

        [SerializeField]
        private float viewportWidth = 0.36f;

        [SerializeField]
        private float viewportHeight = 0.32f;

        [Header("Seed Sampling")]
        [SerializeField]
        private int gridColumns = 12;

        [SerializeField]
        private int gridRows = 8;

        [SerializeField]
        private int seedSamplesPerTick = 6;

        [SerializeField]
        private int samplesPerTick = 18;

        [SerializeField]
        private float sampleIntervalSeconds = 0.03f;

        [SerializeField]
        private float rayMaxDistanceMeters = 3.5f;

        [SerializeField]
        private float minNormalConfidence = 0.2f;

        [Header("Surface Spacing")]
        [SerializeField]
        private float targetSurfaceSpacingMeters = 0.22f;

        [SerializeField]
        private float cellHitToleranceMeters = 0.08f;

        [SerializeField]
        private float patchMergeDistanceMeters = 0.25f;

        [SerializeField]
        private float patchNormalDot = 0.9f;

        [SerializeField]
        private int maxPatches = 24;

        [Header("Patch Memory")]
        [SerializeField]
        private float unstableCellCooldownSeconds = 0.08f;

        [SerializeField]
        private float stableCellCooldownSeconds = 1.25f;

        [SerializeField]
        private float patchHoldSeconds = 8.0f;

        [SerializeField]
        private int stableHitsPerCell = 1;

        [SerializeField]
        private bool debugLog;

        public int LastTickAttemptCount { get; private set; }
        public int LastTickScheduledCount { get; private set; }
        public int LastTickAcceptedHitCount { get; private set; }
        public int LastTickAcceptedOccludedCount { get; private set; }
        public int LastTickRejectedConfidenceCount { get; private set; }
        public float LastTickTime { get; private set; }
        public float TargetSurfaceSpacingMeters => Mathf.Max(0.05f, targetSurfaceSpacingMeters);

        private readonly List<SurfacePatch> m_patches = new List<SurfacePatch>(24);
        private SeedCellState[] m_seedCells;
        private int m_seedCursor;
        private int m_patchCursor;
        private float m_nextSampleTime;
        private float m_seedSampleScale = 1f;
        private float m_patchSampleScale = 1f;
        private float m_intervalScale = 1f;
        private float m_rayDistanceScale = 1f;

        private void OnEnable()
        {
            EnsureSeedBuffer();
            m_nextSampleTime = Time.unscaledTime;
        }

        private void Update()
        {
            ResolveRefs();
            if (environmentRaycastManager == null || sampleCamera == null || voxelPointAccumulator == null)
                return;

            EnsureSeedBuffer();
            if (Time.unscaledTime < m_nextSampleTime)
                return;

            m_nextSampleTime = Time.unscaledTime + ResolveSampleIntervalSeconds();
            SampleSurfaceSpacing();
        }

        public void Configure(EnvironmentRaycastManager raycastManager, Camera camera, ScanCoverVoxelPointAccumulator accumulator)
        {
            environmentRaycastManager = raycastManager;
            sampleCamera = camera;
            voxelPointAccumulator = accumulator;
            EnsureSeedBuffer();
        }

        public void ApplyPerformanceSlice(float seedSampleScale, float patchSampleScale, float intervalScale, float rayDistanceScale)
        {
            m_seedSampleScale = Mathf.Max(0.1f, seedSampleScale);
            m_patchSampleScale = Mathf.Max(0.1f, patchSampleScale);
            m_intervalScale = Mathf.Max(0.1f, intervalScale);
            m_rayDistanceScale = Mathf.Max(0.1f, rayDistanceScale);
        }

        public void GetDisplayCellsNonAlloc(List<DisplayCell> results)
        {
            if (results == null)
                return;

            results.Clear();
            if (m_patches.Count <= 0)
                return;

            for (int i = 0; i < m_patches.Count; i++)
            {
                SurfacePatch patch = m_patches[i];
                foreach (PatchCellKey key in patch.confirmedCells)
                {
                    float spacing = Mathf.Max(0.05f, targetSurfaceSpacingMeters);
                    results.Add(new DisplayCell
                    {
                        worldPos = patch.anchor + patch.tangent * (key.u * spacing) + patch.bitangent * (key.v * spacing),
                        normal = patch.normal,
                        tangent = patch.tangent,
                        bitangent = patch.bitangent,
                        u = key.u,
                        v = key.v,
                        patchHits = patch.seedHits,
                        lastSeenTime = patch.lastSeenTime,
                    });
                }
            }
        }

        private void ResolveRefs()
        {
            if (environmentRaycastManager == null)
                environmentRaycastManager = FindAnyObjectByType<EnvironmentRaycastManager>(FindObjectsInactive.Include);
            if (sampleCamera == null)
                sampleCamera = Camera.main;
            if (voxelPointAccumulator == null)
                voxelPointAccumulator = GetComponent<ScanCoverVoxelPointAccumulator>();
        }

        private void EnsureSeedBuffer()
        {
            int totalCells = Mathf.Max(4, gridColumns) * Mathf.Max(4, gridRows);
            if (m_seedCells != null && m_seedCells.Length == totalCells)
                return;

            m_seedCells = new SeedCellState[totalCells];
            m_seedCursor = 0;
        }

        private void SampleSurfaceSpacing()
        {
            float timeNow = Time.unscaledTime;
            PrunePatches(timeNow);

            int attempts = 0;
            int scheduled = 0;
            int acceptedHits = 0;
            int acceptedOccludedHits = 0;
            int rejectedConfidence = 0;

            int emittedSeedSamples = 0;
            int seedAttemptBudget = Mathf.Max(1, m_seedCells.Length);
            int effectiveSeedSamplesPerTick = Mathf.Max(1, Mathf.RoundToInt(seedSamplesPerTick * m_seedSampleScale));
            while (emittedSeedSamples < effectiveSeedSamplesPerTick && seedAttemptBudget-- > 0)
            {
                int seedIndex = m_seedCursor;
                m_seedCursor = (m_seedCursor + 1) % m_seedCells.Length;
                ref SeedCellState cell = ref m_seedCells[seedIndex];
                float cooldown = cell.stableHitCount >= Mathf.Max(1, stableHitsPerCell)
                    ? Mathf.Max(0.05f, stableCellCooldownSeconds)
                    : Mathf.Max(0.02f, unstableCellCooldownSeconds);
                if (timeNow - cell.lastSampleTime < cooldown)
                    continue;

                cell.lastSampleTime = timeNow;
                scheduled++;
                attempts++;
                emittedSeedSamples++;

                if (TrySampleSeedCell(seedIndex, timeNow, out bool acceptedOccluded, out bool rejectedByConfidence))
                {
                    cell.stableHitCount = Mathf.Min(cell.stableHitCount + 1, 255);
                    acceptedHits++;
                    if (acceptedOccluded)
                        acceptedOccludedHits++;
                }
                else if (rejectedByConfidence)
                {
                    rejectedConfidence++;
                }
            }

            int emittedPatchSamples = 0;
            int effectivePatchSamplesPerTick = Mathf.Max(1, Mathf.RoundToInt(samplesPerTick * m_patchSampleScale));
            int patchAttemptBudget = Mathf.Max(effectivePatchSamplesPerTick * 6, Mathf.Max(1, m_patches.Count) * 3);
            while (emittedPatchSamples < effectivePatchSamplesPerTick && patchAttemptBudget-- > 0)
            {
                if (m_patches.Count <= 0)
                    break;

                SurfacePatch patch = m_patches[m_patchCursor];
                m_patchCursor = (m_patchCursor + 1) % m_patches.Count;

                if (!TrySamplePatchCell(patch, timeNow, out bool attemptedRay, out bool acceptedHit, out bool acceptedOccluded, out bool rejectedByConfidence))
                    continue;

                if (!attemptedRay)
                    continue;

                scheduled++;
                attempts++;
                emittedPatchSamples++;
                if (acceptedHit)
                {
                    acceptedHits++;
                    if (acceptedOccluded)
                        acceptedOccludedHits++;
                }
                else if (rejectedByConfidence)
                {
                    rejectedConfidence++;
                }
            }

            LastTickAttemptCount = attempts;
            LastTickScheduledCount = scheduled;
            LastTickAcceptedHitCount = acceptedHits;
            LastTickAcceptedOccludedCount = acceptedOccludedHits;
            LastTickRejectedConfidenceCount = rejectedConfidence;
            LastTickTime = timeNow;
        }

        private bool TrySampleSeedCell(int cellIndex, float timeNow, out bool acceptedOccludedHit, out bool rejectedByConfidence)
        {
            acceptedOccludedHit = false;
            rejectedByConfidence = false;

            int columns = Mathf.Max(4, gridColumns);
            int rows = Mathf.Max(4, gridRows);
            int column = cellIndex % columns;
            int row = cellIndex / columns;

            float minX = Mathf.Clamp01(viewportCenterX - viewportWidth * 0.5f);
            float minY = Mathf.Clamp01(viewportCenterY - viewportHeight * 0.5f);
            float width = Mathf.Clamp(viewportWidth, 0.05f, 1f);
            float height = Mathf.Clamp(viewportHeight, 0.05f, 1f);

            float u = minX + width * ((column + 0.5f) / columns);
            float v = minY + height * ((row + 0.5f) / rows);
            Ray ray = sampleCamera.ViewportPointToRay(new Vector3(u, v, 0f));

            if (!TryResolveHit(ray, ResolveRayMaxDistanceMeters(), out EnvironmentRaycastHit hit, out acceptedOccludedHit, out rejectedByConfidence))
                return false;

            Vector3 normal = ResolveNormal(hit.normal, -ray.direction);
            voxelPointAccumulator.AddObservation(hit.point, normal, ResolveConfidence(hit), timeNow);
            SurfacePatch patch = ResolveOrCreatePatch(hit.point, normal, timeNow);
            if (patch != null)
                RegisterConfirmedCell(patch, hit.point);

            if (debugLog && UnityEngine.Random.value < 0.0025f)
                Debug.Log($"[DepthEffectsSurfaceSpacedSampler] seed hit={hit.point}, status={hit.status}, patches={m_patches.Count}");

            return true;
        }

        private bool TrySamplePatchCell(
            SurfacePatch patch,
            float timeNow,
            out bool attemptedRay,
            out bool acceptedHit,
            out bool acceptedOccluded,
            out bool rejectedByConfidence)
        {
            attemptedRay = false;
            acceptedHit = false;
            acceptedOccluded = false;
            rejectedByConfidence = false;

            if (sampleCamera == null || patch == null)
                return false;

            int localAttempts = 0;
            while (localAttempts++ < 6)
            {
                PatchCellKey cellKey = SpiralIndexToCellKey(patch.spiralCursor++);
                float cooldown = patch.confirmedCells.Contains(cellKey)
                    ? Mathf.Max(0.05f, stableCellCooldownSeconds)
                    : Mathf.Max(0.02f, unstableCellCooldownSeconds);

                if (patch.cellLastSampleTimes.TryGetValue(cellKey, out float lastSampleTime) &&
                    timeNow - lastSampleTime < cooldown)
                    continue;

                patch.cellLastSampleTimes[cellKey] = timeNow;
                attemptedRay = true;

                Vector3 target = patch.anchor +
                                 patch.tangent * (cellKey.u * Mathf.Max(0.05f, targetSurfaceSpacingMeters)) +
                                 patch.bitangent * (cellKey.v * Mathf.Max(0.05f, targetSurfaceSpacingMeters));
                Vector3 origin = sampleCamera.transform.position;
                Vector3 toTarget = target - origin;
                float distance = toTarget.magnitude;
                if (distance <= 1e-4f)
                    return false;

                Ray ray = new Ray(origin, toTarget / distance);
                if (!TryResolveHit(ray, Mathf.Min(distance + Mathf.Max(0.25f, targetSurfaceSpacingMeters), ResolveRayMaxDistanceMeters()),
                        out EnvironmentRaycastHit hit, out acceptedOccluded, out rejectedByConfidence))
                    return true;

                Vector3 hitNormal = ResolveNormal(hit.normal, -ray.direction);
                if (Vector3.Dot(hitNormal, patch.normal) < Mathf.Clamp(patchNormalDot, -1f, 1f))
                    return true;

                float hitTolerance = Mathf.Max(0.02f, cellHitToleranceMeters);
                if ((hit.point - target).sqrMagnitude > hitTolerance * hitTolerance)
                    return true;

                acceptedHit = true;
                voxelPointAccumulator.AddObservation(hit.point, hitNormal, ResolveConfidence(hit), timeNow);
                UpdatePatchFromHit(patch, hit.point, hitNormal, timeNow);
                patch.confirmedCells.Add(cellKey);
                return true;
            }

            return false;
        }

        private bool TryResolveHit(
            Ray ray,
            float maxDistance,
            out EnvironmentRaycastHit hit,
            out bool acceptedOccludedHit,
            out bool rejectedByConfidence)
        {
            acceptedOccludedHit = false;
            rejectedByConfidence = false;

            bool didHit = environmentRaycastManager.Raycast(ray, out hit, Mathf.Max(0.25f, maxDistance));
            if (!didHit && hit.status != EnvironmentRaycastHitStatus.HitPointOccluded)
                return false;

            if (hit.status != EnvironmentRaycastHitStatus.Hit &&
                hit.status != EnvironmentRaycastHitStatus.HitPointOccluded)
                return false;

            float confidence = ResolveConfidence(hit);
            if (confidence < Mathf.Clamp01(minNormalConfidence))
            {
                rejectedByConfidence = true;
                return false;
            }

            acceptedOccludedHit = hit.status == EnvironmentRaycastHitStatus.HitPointOccluded;
            return true;
        }

        private float ResolveConfidence(EnvironmentRaycastHit hit)
        {
            return hit.status == EnvironmentRaycastHitStatus.Hit
                ? hit.normalConfidence
                : Mathf.Min(hit.normalConfidence, 0.5f);
        }

        private SurfacePatch ResolveOrCreatePatch(Vector3 hitPoint, Vector3 hitNormal, float timeNow)
        {
            SurfacePatch bestPatch = null;
            float bestScore = float.MinValue;
            float mergeDistanceSq = Mathf.Max(0.05f, patchMergeDistanceMeters);
            mergeDistanceSq *= mergeDistanceSq;
            float minNormalDot = Mathf.Clamp(patchNormalDot, -1f, 1f);

            for (int i = 0; i < m_patches.Count; i++)
            {
                SurfacePatch patch = m_patches[i];
                float distanceSq = (patch.anchor - hitPoint).sqrMagnitude;
                if (distanceSq > mergeDistanceSq)
                    continue;

                float normalDot = Vector3.Dot(patch.normal, hitNormal);
                if (normalDot < minNormalDot)
                    continue;

                float score = normalDot * 4f - distanceSq;
                if (score <= bestScore)
                    continue;

                bestScore = score;
                bestPatch = patch;
            }

            if (bestPatch != null)
            {
                UpdatePatchFromHit(bestPatch, hitPoint, hitNormal, timeNow);
                return bestPatch;
            }

            if (m_patches.Count >= Mathf.Max(4, maxPatches))
                return null;

            SurfacePatch created = new SurfacePatch
            {
                anchor = hitPoint,
                normal = hitNormal,
                lastSeenTime = timeNow,
                seedHits = 1,
                spiralCursor = 0,
            };
            RebuildPatchBasis(created);
            m_patches.Add(created);
            return created;
        }

        private void UpdatePatchFromHit(SurfacePatch patch, Vector3 hitPoint, Vector3 hitNormal, float timeNow)
        {
            float weight = 1f / Mathf.Max(1, patch.seedHits + 1);
            patch.anchor = Vector3.Lerp(patch.anchor, hitPoint, weight);
            patch.normal = Vector3.Slerp(patch.normal, hitNormal, weight).normalized;
            patch.lastSeenTime = timeNow;
            patch.seedHits = Mathf.Min(patch.seedHits + 1, 1024);
            RebuildPatchBasis(patch);
        }

        private void RebuildPatchBasis(SurfacePatch patch)
        {
            Vector3 normal = patch.normal.sqrMagnitude > 1e-5f ? patch.normal.normalized : Vector3.up;
            Vector3 projectedUp = Vector3.ProjectOnPlane(Vector3.up, normal);
            if (projectedUp.sqrMagnitude > 1e-4f)
            {
                patch.bitangent = projectedUp.normalized;
                patch.tangent = Vector3.Cross(patch.bitangent, normal).normalized;
            }
            else
            {
                Vector3 projectedForward = Vector3.ProjectOnPlane(Vector3.forward, normal);
                if (projectedForward.sqrMagnitude <= 1e-4f)
                    projectedForward = Vector3.ProjectOnPlane(Vector3.right, normal);

                patch.tangent = projectedForward.sqrMagnitude > 1e-5f
                    ? projectedForward.normalized
                    : Vector3.right;
                patch.bitangent = Vector3.Cross(normal, patch.tangent).normalized;
            }

            if (patch.tangent.sqrMagnitude <= 1e-5f)
                patch.tangent = Vector3.right;
            if (patch.bitangent.sqrMagnitude <= 1e-5f)
                patch.bitangent = Vector3.up;
        }

        private void RegisterConfirmedCell(SurfacePatch patch, Vector3 hitPoint)
        {
            Vector3 delta = hitPoint - patch.anchor;
            float spacing = Mathf.Max(0.05f, targetSurfaceSpacingMeters);
            PatchCellKey key = new PatchCellKey
            {
                u = Mathf.RoundToInt(Vector3.Dot(delta, patch.tangent) / spacing),
                v = Mathf.RoundToInt(Vector3.Dot(delta, patch.bitangent) / spacing),
            };
            patch.confirmedCells.Add(key);
        }

        private void PrunePatches(float timeNow)
        {
            float keepSeconds = Mathf.Max(0.25f, patchHoldSeconds);
            for (int i = m_patches.Count - 1; i >= 0; i--)
            {
                if (timeNow - m_patches[i].lastSeenTime <= keepSeconds)
                    continue;
                m_patches.RemoveAt(i);
                if (m_patchCursor >= m_patches.Count)
                    m_patchCursor = 0;
            }
        }

        private static Vector3 ResolveNormal(Vector3 normal, Vector3 fallback)
        {
            if (normal.sqrMagnitude > 1e-5f)
                return normal.normalized;
            if (fallback.sqrMagnitude > 1e-5f)
                return fallback.normalized;
            return Vector3.up;
        }

        private float ResolveSampleIntervalSeconds()
        {
            return Mathf.Max(0.01f, sampleIntervalSeconds * m_intervalScale);
        }

        private float ResolveRayMaxDistanceMeters()
        {
            return Mathf.Max(0.25f, rayMaxDistanceMeters * m_rayDistanceScale);
        }

        private static PatchCellKey SpiralIndexToCellKey(int index)
        {
            if (index <= 0)
                return new PatchCellKey { u = 0, v = 0 };

            int x = 0;
            int y = 0;
            int dx = 0;
            int dy = -1;
            int side = Mathf.CeilToInt(Mathf.Sqrt(index + 1));
            int maxSteps = side * side;
            for (int step = 0; step < maxSteps; step++)
            {
                if (-side / 2 <= x && x <= side / 2 && -side / 2 <= y && y <= side / 2)
                {
                    if (step == index)
                        return new PatchCellKey { u = x, v = y };
                }

                if (x == y || (x < 0 && x == -y) || (x > 0 && x == 1 - y))
                {
                    int temp = dx;
                    dx = -dy;
                    dy = temp;
                }

                x += dx;
                y += dy;
            }

            return new PatchCellKey { u = x, v = y };
        }
    }
}
