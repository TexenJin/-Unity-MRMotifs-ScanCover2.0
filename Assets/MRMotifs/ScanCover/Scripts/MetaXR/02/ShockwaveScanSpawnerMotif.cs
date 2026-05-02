// Copyright (c) Meta Platforms, Inc. and affiliates.
// Custom decoupled scan spawner（方案B：完全解耦、低频、相机前方 + 环境射线命中点）

using Meta.XR;
using UnityEngine;
using static OVRInput;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    /// <summary>
    /// 完全解耦版扫描触发器：
    /// - 不依赖 OrbSpawnerMotif / ShockWaveOrbMotif
    /// - 按键后从相机前方发射环境射线
    /// - 命中真实环境则在 hit.point 生成 ShockWaveEffect
    /// - 低频触发（带冷却），可无限次
    ///
    /// （升级点）
    /// - 在命中点处额外调用 revealManager.TriggerWaveAt(hit.point)
    ///   让“空间显示层”Shader做环形扩散 reveal，与 ShockWave 联动。
    /// </summary>
    public class ShockwaveScanSpawnerMotif : MonoBehaviour
    {
        [Header("Refs")]
        [Tooltip("环境射线管理器（必须）")]
        [SerializeField]
        private EnvironmentRaycastManager environmentRaycast;

        [Tooltip("扫描波 Prefab：请拖 ShockWaveEffect.prefab 上的 AudioSource 组件")]
        [SerializeField]
        private AudioSource shockWaveEffectPrefab;

        [Tooltip("扫描触发音效")]
        [SerializeField]
        private AudioClip scanTrigger;

        [Tooltip("RevealManager（动态环形波）。可为空；为空则只生成 ShockWave 特效")]
        [SerializeField]
        private RevealManager revealManager;

        [Header("Trigger Input")]
        [Tooltip("触发按钮（低频按一下触发一次）")]
        [SerializeField]
        private Button triggerButton = Button.One;

        [Tooltip("冷却时间（秒），防止误触连发")]
        [SerializeField, Min(0f)]
        private float cooldownSeconds = 0.15f;

        [Header("Raycast")]
        [Tooltip("从相机前方发射环境射线的最大距离")]
        [SerializeField, Min(0.1f)]
        private float maxRayDistance = 8.0f;

        [Tooltip("未命中时的后备距离（从相机前方多少米生成）")]
        [SerializeField, Min(0.1f)]
        private float fallbackDistance = 1.2f;

        [Tooltip("若为 true：必须命中真实环境才触发；若为 false：未命中则用 fallback 点触发")]
        [SerializeField]
        private bool requireValidHit = false;

        [Header("Audio Pitch")]
        [Tooltip("是否启用连续触发时轻微升调（和原 OrbSpawner 类似）")]
        [SerializeField]
        private bool usePitchRamp = true;

        [SerializeField, Min(0.1f)]
        private float minPitch = 1.0f;

        [SerializeField, Min(0.1f)]
        private float maxPitch = 3.0f;

        [SerializeField, Min(0f)]
        private float pitchRisePerTrigger = 0.25f;

        [SerializeField, Min(0f)]
        private float pitchRecoverPerSecond = 1.0f;

        [Header("Debug")]
        [SerializeField]
        private bool debugLog = false;

        [SerializeField]
        private bool debugDrawRay = true;

        private Transform m_cameraTransform;
        private float m_nextTriggerTime;
        private float m_pitchLevel = 1.0f;

        private void Awake()
        {
            CacheMainCamera();
        }

        private void Update()
        {
            if (usePitchRamp)
            {
                m_pitchLevel = Mathf.Max(minPitch, m_pitchLevel - pitchRecoverPerSecond * Time.deltaTime);
            }

            if (Time.time < m_nextTriggerTime)
            {
                return;
            }

            if (GetDown(triggerButton))
            {
                TryTriggerScanFromCameraForward();
            }
        }

        /// <summary>
        /// 从相机前方发射环境射线，优先使用命中点触发扫描。
        /// </summary>
        public bool TryTriggerScanFromCameraForward()
        {
            if (!ValidateRefs())
            {
                return false;
            }

            if (!m_cameraTransform)
            {
                CacheMainCamera();
                if (!m_cameraTransform)
                {
                    if (debugLog) Debug.LogWarning("[ShockwaveScanSpawnerMotif] Camera.main 不存在，无法触发扫描。");
                    return false;
                }
            }

            Vector3 origin = m_cameraTransform.position;
            Vector3 dir = m_cameraTransform.forward;
            Ray ray = new Ray(origin, dir);

            bool hitSuccess = environmentRaycast.Raycast(ray, out var hit, maxDistance: maxRayDistance);

            if (debugDrawRay)
            {
                Debug.DrawRay(origin, dir * maxRayDistance, hitSuccess ? Color.green : Color.yellow, 0.2f);
            }

            // 与原工程风格对齐：HitPointOccluded 也可作为有效点使用（如果 hit.point 可用）
            bool treatAsUsableHit =
                hitSuccess ||
                hit.status == EnvironmentRaycastHitStatus.HitPointOccluded;

            if (treatAsUsableHit)
            {
                TriggerScanAt(hit.point);
                // ★只加这一行调用：将命中点写入动态波 reveal（空间显示层会出现环形扩散）
                revealManager?.TriggerWaveAt(hit.point);

                if (debugLog)
                {
                    Debug.Log($"[ShockwaveScanSpawnerMotif] Scan triggered at hit.point = {hit.point}, status = {hit.status}");
                }

                m_nextTriggerTime = Time.time + cooldownSeconds;
                return true;
            }

            if (requireValidHit)
            {
                if (debugLog)
                {
                    Debug.Log($"[ShockwaveScanSpawnerMotif] No valid hit. status = {hit.status}, trigger cancelled.");
                }
                return false;
            }

            // 未命中时 fallback（便于调试链路）
            Vector3 fallbackPos = origin + dir * fallbackDistance;
            TriggerScanAt(fallbackPos);

            if (debugLog)
            {
                Debug.Log($"[ShockwaveScanSpawnerMotif] No valid hit. Fallback scan at {fallbackPos}, status = {hit.status}");
            }

            m_nextTriggerTime = Time.time + cooldownSeconds;
            return true;
        }

        /// <summary>
        /// 在指定世界坐标直接生成扫描波（可被其他系统调用）。
        /// </summary>
        public void TriggerScanAt(Vector3 worldPos)
        {
            if (!shockWaveEffectPrefab)
            {
                if (debugLog) Debug.LogWarning("[ShockwaveScanSpawnerMotif] shockWaveEffectPrefab 未赋值。");
                return;
            }

            var scanEffectAudioSrc = Instantiate(shockWaveEffectPrefab, worldPos, Quaternion.identity);

            if (scanTrigger)
            {
                scanEffectAudioSrc.clip = scanTrigger;
            }

            if (usePitchRamp)
            {
                m_pitchLevel = Mathf.Clamp(m_pitchLevel + pitchRisePerTrigger, minPitch, maxPitch);
                scanEffectAudioSrc.pitch = m_pitchLevel;
            }

            scanEffectAudioSrc.Play();
        }

        private bool ValidateRefs()
        {
            if (!environmentRaycast)
            {
                if (debugLog) Debug.LogWarning("[ShockwaveScanSpawnerMotif] environmentRaycast 未赋值。");
                return false;
            }

            if (!shockWaveEffectPrefab)
            {
                if (debugLog) Debug.LogWarning("[ShockwaveScanSpawnerMotif] shockWaveEffectPrefab 未赋值。");
                return false;
            }

            return true;
        }

        private void CacheMainCamera()
        {
            if (Camera.main != null)
            {
                m_cameraTransform = Camera.main.transform;
            }
        }
    }
}
