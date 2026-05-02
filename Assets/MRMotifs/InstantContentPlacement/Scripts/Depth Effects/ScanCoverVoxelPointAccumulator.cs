using System.Collections.Generic;
using UnityEngine;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    [DisallowMultipleComponent]
    public sealed class ScanCoverVoxelPointAccumulator : MonoBehaviour
    {
        public struct StablePoint
        {
            public Vector3 worldPos;
            public Vector3 normal;
            public float confidence;
            public float lastSeenTime;
        }

        private struct CellKey
        {
            public int x;
            public int y;
            public int z;
        }

        private struct CellState
        {
            public bool valid;
            public Vector3 worldPos;
            public Vector3 normal;
            public float confidence;
            public float lastSeenTime;
            public int stableHits;
        }

        [Header("Compatibility Settings")]
        public float cellSizeMeters = 0.03f;
        public int minStableHits = 1;
        public float holdSeconds = 12f;
        public int maxStablePoints = 16000;
        public bool enableNeighborMerge = true;
        public float neighborMergeDistanceMeters = 0.06f;
        public float neighborMergeNormalDot = 0.86f;
        public bool debugLog;

        public int TotalVoxelCount => _cells.Count;
        public int Revision { get; private set; }

        private readonly Dictionary<CellKey, CellState> _cells = new(4096);
        private readonly List<StablePoint> _stablePoints = new(4096);

        private void LateUpdate()
        {
            CleanupExpired(Time.unscaledTime);
        }

        public void AddObservation(Vector3 point, Vector3 normal, float confidence, float timeNow)
        {
            float cellSize = Mathf.Max(0.005f, cellSizeMeters);
            CellKey key = new CellKey
            {
                x = Mathf.RoundToInt(point.x / cellSize),
                y = Mathf.RoundToInt(point.y / cellSize),
                z = Mathf.RoundToInt(point.z / cellSize)
            };

            if (_cells.TryGetValue(key, out CellState existing))
            {
                float blend = existing.stableHits <= 0 ? 1f : Mathf.Clamp01(1f / Mathf.Min(6f, existing.stableHits + 1f));
                existing.worldPos = Vector3.Lerp(existing.worldPos, point, blend);
                Vector3 blendedNormal = Vector3.Lerp(existing.normal, SafeNormal(normal), blend);
                existing.normal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : Vector3.up;
                existing.confidence = Mathf.Clamp01(Mathf.Max(existing.confidence, confidence));
                existing.lastSeenTime = timeNow;
                existing.stableHits++;
                existing.valid = true;
                _cells[key] = existing;
            }
            else
            {
                _cells[key] = new CellState
                {
                    valid = true,
                    worldPos = point,
                    normal = SafeNormal(normal),
                    confidence = Mathf.Clamp01(confidence),
                    lastSeenTime = timeNow,
                    stableHits = 1
                };
            }

            Revision++;
        }

        public void ClearAll()
        {
            _cells.Clear();
            _stablePoints.Clear();
            Revision++;
        }

        public void GetStablePointsNonAlloc(List<StablePoint> destination)
        {
            destination.Clear();
            float now = Time.unscaledTime;
            foreach (KeyValuePair<CellKey, CellState> pair in _cells)
            {
                CellState cell = pair.Value;
                if (!cell.valid)
                    continue;
                if (cell.stableHits < Mathf.Max(1, minStableHits))
                    continue;
                if ((now - cell.lastSeenTime) > Mathf.Max(0.05f, holdSeconds))
                    continue;

                destination.Add(new StablePoint
                {
                    worldPos = cell.worldPos,
                    normal = cell.normal,
                    confidence = cell.confidence,
                    lastSeenTime = cell.lastSeenTime
                });
            }

            if (destination.Count > Mathf.Max(64, maxStablePoints))
                destination.RemoveRange(Mathf.Max(64, maxStablePoints), destination.Count - Mathf.Max(64, maxStablePoints));
        }

        public void Prune(float timeNow)
        {
            CleanupExpired(timeNow);
        }

        private void CleanupExpired(float now)
        {
            if (_cells.Count <= 0)
                return;

            List<CellKey> stale = null;
            float keepSeconds = Mathf.Max(0.05f, holdSeconds);
            foreach (KeyValuePair<CellKey, CellState> pair in _cells)
            {
                if ((now - pair.Value.lastSeenTime) <= keepSeconds)
                    continue;
                stale ??= new List<CellKey>();
                stale.Add(pair.Key);
            }

            if (stale == null || stale.Count <= 0)
                return;

            for (int i = 0; i < stale.Count; i++)
                _cells.Remove(stale[i]);

            Revision++;
        }

        private static Vector3 SafeNormal(Vector3 normal)
        {
            if (normal.sqrMagnitude <= 1e-6f)
                return Vector3.up;
            return normal.normalized;
        }
    }
}
