// Copyright (c) Meta Platforms, Inc. and affiliates.

#if FUSION2
using System.Collections;
using Fusion;
using UnityEngine;

namespace MRMotifs.ColocatedExperiences.BouncingBall
{
    public class BallSpawnerMotif : NetworkBehaviour
    {
        [Header("References")]
        [Tooltip("Projectile prefab with a NetworkObject component attached.")]
        [SerializeField] private NetworkObject projectilePrefab;

        [Tooltip("Impulse force applied to the projectile.")]
        [SerializeField] private float fireForce = 1.0f;
        
        [Tooltip("Life time of the ball prefab.")]
        [SerializeField] private float lifeTime = 5.0f;

        private bool m_spawned;
        private Transform m_firePoint;
        private NetworkObject m_ballObject;

        public override void Spawned()
        {
            base.Spawned();
            
            if (Camera.main != null)
            {
                m_firePoint = Camera.main.transform;
            }

            m_spawned = true;
        }

        private void Update()
        {
            if (!m_spawned)
            {
                return;
            }

            if (!OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger))
            {
                return;
            }

            if (Object.HasStateAuthority)
            {
                SpawnProjectile();
            }
            else
            {
                Object.RequestStateAuthority();
                SpawnProjectile();
            }
        }

        private void SpawnProjectile()
        {
            m_ballObject = Runner.Spawn(projectilePrefab, m_firePoint.position, m_firePoint.rotation);

            if (!m_ballObject)
            {
                Debug.LogError("[BallSpawnerMotif] Failed to spawn projectile.");
                return;
            }

            var rb = m_ballObject.GetComponent<Rigidbody>();
            if (rb)
            {
                rb.AddForce(m_firePoint.forward * fireForce, ForceMode.Impulse);
            }
            else
            {
                Debug.LogError("[BallSpawnerMotif] Spawned ball is missing Rigidbody.");
            }

            StartCoroutine(DespawnAfterDelay(m_ballObject, lifeTime));
        }

        private IEnumerator DespawnAfterDelay(NetworkObject obj, float delay)
        {
            yield return new WaitForSeconds(delay);

            if (!obj || !Runner.IsRunning || !obj.HasStateAuthority)
            {
                yield break;
            }

            Runner.Despawn(obj);
        }
    }
}
#endif
