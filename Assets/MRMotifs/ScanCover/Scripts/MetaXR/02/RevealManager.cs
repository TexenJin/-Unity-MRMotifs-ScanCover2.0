using UnityEngine;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    /// <summary>
    /// RevealManager（Fill-only / 留痕版）
    /// 目标观感：一个逐渐扩大的“圆/球”，圆内空间被揭示，并在到达最大半径后保留（扫过留痕）。
    ///
    /// 机制：
    /// - 每次 TriggerWaveAt(worldPos) 记录一条波：中心点 + startTime
    /// - 半径随时间增长：r(t)=min(maxRadius, speed*(t-startTime))
    /// - 达到 maxRadius 后默认冻结并保留（persistReveal=true）
    ///
    /// Shader 全局变量：
    /// - _RevealWaves[i] = (x,y,z,radius)
    /// - _RevealWaveCount = float
    /// - _RevealWaveParams = (edgeFeather, 0, 0, 0)
    /// </summary>
    public class RevealManager : MonoBehaviour
    {
        public const int MaxWaves = 32; // 与 shader 中 REVEAL_MAX_WAVES 保持一致

        [Header("Capacity")]
        [SerializeField, Range(1, MaxWaves)]
        private int maxWaves = 16;

        [Header("Wave Fill")]
        [Tooltip("扩散速度（米/秒）")]
        [SerializeField, Min(0.01f)]
        private float waveSpeed = 2.5f;

        [Tooltip("最大半径（米）。到达后默认冻结并保留（留痕）。")]
        [SerializeField, Min(0.05f)]
        private float maxRadius = 3.5f;

        [Tooltip("是否保留留痕（建议 true）。false 则波到最大半径后会被移除。")]
        [SerializeField]
        private bool persistReveal = true;

        [Tooltip("圆边缘羽化宽度（米），越大边缘越柔。")]
        [SerializeField, Min(0.0f)]
        private float edgeFeather = 0.25f;

        [Header("Globals")]
        [Tooltip("每帧推送 Shader 全局参数。低频触发下建议保持 true。")]
        [SerializeField]
        private bool pushGlobalsEveryFrame = true;

        private struct Wave
        {
            public Vector3 center;
            public float startTime;
            public float radius;
            public bool frozen;
        }

        private readonly Wave[] _waves = new Wave[MaxWaves];
        private int _count;
        private int _writeIndex;

        private readonly Vector4[] _wavesUpload = new Vector4[MaxWaves];

        private static readonly int RevealWavesId = Shader.PropertyToID("_RevealWaves");
        private static readonly int RevealWaveCountId = Shader.PropertyToID("_RevealWaveCount");
        private static readonly int RevealWaveParamsId = Shader.PropertyToID("_RevealWaveParams");

        private void OnEnable()
        {
            PushGlobals();
        }

        private void OnDisable()
        {
            ClearAll();
        }

        private void LateUpdate()
        {
            UpdateWaves();
            if (pushGlobalsEveryFrame)
            {
                PushGlobals();
            }
        }

        /// <summary>
        /// 触发一条扩散波（逐渐扩大、圆内揭示、留痕）。
        /// </summary>
        public void TriggerWaveAt(Vector3 worldPos)
        {
            int slot = _writeIndex;

            _waves[slot] = new Wave
            {
                center = worldPos,
                startTime = Time.time,
                radius = 0f,
                frozen = false
            };

            _writeIndex = (_writeIndex + 1) % Mathf.Max(1, maxWaves);
            _count = Mathf.Min(_count + 1, maxWaves);

            PushGlobals();
        }

        [ContextMenu("Clear Reveal Waves")]
        public void ClearAll()
        {
            for (int i = 0; i < MaxWaves; i++)
            {
                _waves[i] = default;
                _wavesUpload[i] = Vector4.zero;
            }

            _count = 0;
            _writeIndex = 0;
            PushGlobals();
        }

        private void UpdateWaves()
        {
            if (_count <= 0) return;

            float t = Time.time;
            int valid = Mathf.Min(_count, Mathf.Max(1, maxWaves));
            int start = (_count < maxWaves) ? 0 : _writeIndex; // 环形缓冲最旧索引

            bool anyChanged = false;

            for (int i = 0; i < valid; i++)
            {
                int idx = (start + i) % Mathf.Max(1, maxWaves);
                Wave w = _waves[idx];

                if (!w.frozen)
                {
                    float age = Mathf.Max(0f, t - w.startTime);
                    float r = Mathf.Min(maxRadius, waveSpeed * age);
                    w.radius = r;

                    if (r >= maxRadius - 1e-4f)
                    {
                        if (persistReveal)
                        {
                            w.frozen = true; // 留痕：冻结保留
                        }
                        else
                        {
                            _waves[idx] = default; // 不留痕：清除
                            anyChanged = true;
                            continue;
                        }
                    }

                    _waves[idx] = w;
                    anyChanged = true;
                }
            }

            if (anyChanged && !pushGlobalsEveryFrame)
            {
                PushGlobals();
            }
        }

        /// <summary>
        /// 推送 Shader 全局数组与参数。
        /// </summary>
        public void PushGlobals()
        {
            for (int i = 0; i < MaxWaves; i++)
            {
                _wavesUpload[i] = Vector4.zero;
            }

            int valid = Mathf.Min(_count, Mathf.Max(1, maxWaves));
            int start = (_count < maxWaves) ? 0 : _writeIndex;

            int outCount = 0;
            for (int i = 0; i < valid; i++)
            {
                int idx = (start + i) % Mathf.Max(1, maxWaves);
                Wave w = _waves[idx];
                if (w.radius <= 0.0001f) continue;

                _wavesUpload[outCount] = new Vector4(w.center.x, w.center.y, w.center.z, w.radius);
                outCount++;
                if (outCount >= MaxWaves) break;
            }

            Shader.SetGlobalVectorArray(RevealWavesId, _wavesUpload);
            Shader.SetGlobalFloat(RevealWaveCountId, outCount);
            Shader.SetGlobalVector(RevealWaveParamsId, new Vector4(Mathf.Max(0.0001f, edgeFeather), 0f, 0f, 0f));
        }
    }
}
