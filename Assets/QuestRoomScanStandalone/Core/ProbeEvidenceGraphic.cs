using UnityEngine;
using UnityEngine.UI;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Fixed-frame shadow-verdict provenance. One UI draw call renders all
    /// markers; it is a pure consumer of diagnostic snapshots.
    /// </summary>
    public sealed class ProbeEvidenceGraphic : Graphic
    {
        private const int Capacity = 96;
        private readonly Vector2[] _positions = new Vector2[Capacity];
        private readonly Color32[] _colors = new Color32[Capacity];
        private readonly float[] _sizes = new float[Capacity];
        private int _count;

        internal void SetEvidence(Camera camera,
            VirtualProbeShadowAdjudicator.GuidanceCellVisual[] cells, int count)
        {
            _count = 0;
            if (camera == null || cells == null || count <= 0)
            {
                SetVerticesDirty();
                return;
            }

            Rect rect = rectTransform.rect;
            int limit = Mathf.Min(Mathf.Min(count, cells.Length), Capacity);
            for (int i = 0; i < limit; i++)
            {
                Vector3 viewport = camera.WorldToViewportPoint(cells[i].WorldPosition);
                if (viewport.z <= 0f || viewport.x < 0f || viewport.x > 1f ||
                    viewport.y < 0f || viewport.y > 1f)
                    continue;
                _positions[_count] = new Vector2(
                    (viewport.x - 0.5f) * rect.width,
                    (viewport.y - 0.5f) * rect.height);
                _colors[_count] = PhaseColor(cells[i].Phase);
                _sizes[_count] = PhaseSize(cells[i].Phase, cells[i].Strength);
                _count++;
            }
            SetVerticesDirty();
        }

        protected override void OnPopulateMesh(VertexHelper vh)
        {
            vh.Clear();
            for (int i = 0; i < _count; i++)
            {
                Vector2 center = _positions[i];
                float half = _sizes[i] * 0.5f;
                float stroke = Mathf.Max(1.5f, _sizes[i] * 0.16f);
                AddQuad(vh, center + new Vector2(-half, -stroke * 0.5f),
                    new Vector2(_sizes[i], stroke), _colors[i]);
                AddQuad(vh, center + new Vector2(-stroke * 0.5f, -half),
                    new Vector2(stroke, _sizes[i]), _colors[i]);
            }
        }

        private static void AddQuad(VertexHelper vh, Vector2 min, Vector2 size,
            Color32 color)
        {
            int start = vh.currentVertCount;
            vh.AddVert(new Vector3(min.x, min.y), color, Vector2.zero);
            vh.AddVert(new Vector3(min.x, min.y + size.y), color, Vector2.up);
            vh.AddVert(new Vector3(min.x + size.x, min.y + size.y), color,
                Vector2.one);
            vh.AddVert(new Vector3(min.x + size.x, min.y), color, Vector2.right);
            vh.AddTriangle(start, start + 1, start + 2);
            vh.AddTriangle(start, start + 2, start + 3);
        }

        private static Color32 PhaseColor(int phase)
        {
            switch (phase)
            {
                case 0: return new Color32(150, 166, 182, 170); // provisional
                case 1: return new Color32(36, 255, 88, 210);   // stable
                case 2: return new Color32(255, 232, 30, 245); // vote 1
                case 3: return new Color32(255, 132, 18, 250); // vote 2
                case 4: return new Color32(255, 24, 20, 255);  // Reject
                case 5: return new Color32(226, 36, 255, 255); // free confirmed
                case 6: return new Color32(28, 225, 255, 255); // reopened
                case 7: return new Color32(190, 172, 30, 150); // vote 1 cleared
                case 8: return new Color32(205, 96, 18, 165);  // vote 2 cleared
                default: return new Color32(255, 255, 255, 220);
            }
        }

        private static float PhaseSize(int phase, float strength)
        {
            float size = phase <= 1 || phase >= 7
                ? 5f + Mathf.Clamp01(strength) * 2f
                : 7f + Mathf.Clamp01(strength) * 4f;
            if (phase >= 4) size += 4f;
            return size;
        }
    }
}
