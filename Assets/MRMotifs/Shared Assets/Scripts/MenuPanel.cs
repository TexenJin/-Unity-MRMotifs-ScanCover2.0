// Copyright (c) Meta Platforms, Inc. and affiliates.

using UnityEngine;
using UnityEngine.UI;
using Meta.XR.Samples;
using System.Collections;
using Oculus.Interaction;
using System.Collections.Generic;
using Meta.XR.MultiplayerBlocks.Fusion;
using UnityEngine.SceneManagement;

#if FUSION2
using Fusion;
#endif

namespace MRMotifs.SharedAssets
{
    [MetaCodeSample("MRMotifs-SharedAssets")]
    public class MenuPanel : MonoBehaviour
    {
        [Header("MR Motifs - Library: Sample Scenes")]
        [Tooltip("List of buttons that load the scenes.")]
        [SerializeField]
        private List<Button> sceneButtons;

        [Tooltip("List of scene names.")]
        [SerializeField]
        private List<string> sceneNames;

        [Header("Menu Controls")]
        [Tooltip("Root object containing the menu components.")]
        [SerializeField]
        private GameObject menuRoot;

        [Tooltip("Ray Interactable of the canvas.")]
        [SerializeField]
        private RayInteractable rayInteractable;

        [Tooltip("Poke Interactable of the canvas.")]
        [SerializeField]
        private PokeInteractable pokeInteractable;

        [Tooltip("Parent that contains the viewport.")]
        [SerializeField]
        private GameObject menuContent;

        [Tooltip("The button to close the menu.")]
        [SerializeField]
        private Button panelCloseButton;

        [Header("Motif #1 - Passthrough Transitioning")]
        [Tooltip("The button used in the passthrough fader scenes to toggle passthrough on and off.")]
        [SerializeField]
        private Button passthroughFaderButton;

        [Tooltip("The slider used in the passthrough fader slider scene to slowly change visibility.")]
        [SerializeField]
        private Slider passthroughFaderSlider;

        [Header("Motif #2 - Shared Activities")]
        [Tooltip("The slider used in the passthrough fader slider scene to slowly change visibility.")]
        [SerializeField]
        private Button friendsInviteButton;

        public Button PassthroughFaderButton => passthroughFaderButton;
        public Slider PassthroughFaderSlider => passthroughFaderSlider;
        public Button FriendsInviteButton => friendsInviteButton;

        private void Awake()
        {
            panelCloseButton.onClick.AddListener(CloseMenuPanel);
            RegisterSceneButtonListeners();
        }

        private void OnDestroy()
        {
            panelCloseButton.onClick.RemoveListener(CloseMenuPanel);
            DeregisterSceneButtonListeners();
        }

        private void RegisterSceneButtonListeners()
        {
            for (var i = 0; i < sceneButtons.Count; i++)
            {
                var index = i;
                sceneButtons[index].onClick.AddListener(() => LoadScene(index));
            }
        }

        private void DeregisterSceneButtonListeners()
        {
            for (var i = 0; i < sceneButtons.Count; i++)
            {
                var index = i;
                sceneButtons[index].onClick.RemoveListener(() => LoadScene(index));
            }
        }

        private void LoadScene(int sceneIndex)
        {
            if (sceneIndex < 0 || sceneIndex >= sceneNames.Count)
            {
                Debug.LogError($"[MenuPanel] Invalid scene index: {sceneIndex}. Valid range: 0-{sceneNames.Count - 1}");
                return;
            }
#if FUSION2
            var networkRunner = FindAnyObjectByType<NetworkRunner>();

            if (networkRunner != null)
            {
                StartCoroutine(ShutdownFusionAndLoadScene(sceneIndex));
                return;
            }
#endif
            StartCoroutine(LoadSceneAsync(sceneNames[sceneIndex]));
        }

#if FUSION2
        private IEnumerator ShutdownFusionAndLoadScene(int sceneIndex)
        {
            var networkRunner = FindAnyObjectByType<NetworkRunner>();

            if (networkRunner)
            {
                if (!networkRunner.IsRunning)
                {
                    Debug.LogWarning("[MenuPanel] NetworkRunner is not running - may already be shutting down");
                }

                const int MAX_WAIT_FRAMES = 300;
                try
                {
                    var shutdownTask = networkRunner.Shutdown();
                    var frameCount = 0;

                    while (!shutdownTask.IsCompleted && frameCount < MAX_WAIT_FRAMES)
                    {
                        frameCount++;
                    }

                    if (frameCount >= MAX_WAIT_FRAMES)
                    {
                        Debug.LogWarning(
                            $"[MenuPanel] Shutdown timed out after {MAX_WAIT_FRAMES} frames - forcing scene load");
                    }

                    if (shutdownTask.IsCompleted)
                    {
                        if (shutdownTask.IsFaulted)
                        {
                            Debug.LogError($"[MenuPanel] Fusion shutdown failed: {shutdownTask.Exception}");
                            // Continue anyway - we'll force cleanup
                        }
                    }
                }
                catch (System.Exception ex)
                {
                    Debug.LogError($"[MenuPanel] Exception during shutdown: {ex}");
                }

                try
                {
                    var customProvider = FindAnyObjectByType<CustomNetworkObjectProvider>();
                    if (customProvider)
                    {
                        var registryField =
                            typeof(CustomNetworkObjectProvider).GetField("_customNetworkObjects",
                                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                        if (registryField != null)
                        {
                            var registry = registryField.GetValue(customProvider);
                            if (registry is IDictionary dict)
                            {
                                dict.Clear();
                            }
                        }
                    }
                }
                catch (System.Exception ex)
                {
                    Debug.LogError($"[MenuPanel] Error clearing custom registrations: {ex}");
                }

                yield return new WaitForSeconds(0.2f);

                var runnerStillExists = FindAnyObjectByType<NetworkRunner>();
                if (runnerStillExists)
                {
                    try
                    {
                        DestroyImmediate(runnerStillExists.gameObject);
                    }
                    catch (System.Exception ex)
                    {
                        Debug.LogError($"[MenuPanel] Error destroying NetworkRunner: {ex}");
                    }
                }

                var remainingProviders = FindObjectsByType<CustomNetworkObjectProvider>(FindObjectsSortMode.None);
                foreach (var t in remainingProviders)
                {
                    if (t)
                    {
                        DestroyImmediate(t.gameObject);
                    }
                }

                yield return null;
            }
            else
            {
                Debug.LogWarning("[MenuPanel] NetworkRunner was null when coroutine started!");
            }

            yield return StartCoroutine(LoadSceneAsync(sceneNames[sceneIndex]));
        }
#endif

        private IEnumerator LoadSceneAsync(string sceneName)
        {
            var asyncLoad = SceneManager.LoadSceneAsync(sceneName);
            if (asyncLoad == null)
            {
                Debug.LogError($"[MenuPanel] Failed to start async load for scene: {sceneName}");
                yield break;
            }

            while (asyncLoad is { isDone: false })
            {
                yield return null;
            }
        }

        public void ToggleMenu()
        {
            var isMenuActive = menuRoot.activeSelf;
            pokeInteractable.enabled = !isMenuActive;
            rayInteractable.enabled = !isMenuActive;
            menuRoot.SetActive(!isMenuActive);
        }


        private void CloseMenuPanel()
        {
            pokeInteractable.enabled = false;
            rayInteractable.enabled = false;
            menuRoot.SetActive(false);
        }
    }
}
