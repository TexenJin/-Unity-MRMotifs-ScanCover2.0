using UnityEngine;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.RenderGraphModule.Util;
using UnityEngine.Rendering;

/// <summary>
/// DepthRevealOverlayRendererFeature (RenderGraph 长期正确 + MSAA 兼容)
/// - RenderGraph: AddBlitPass(src->dst, material) + resourceData.cameraColor = dst
/// - dst TextureDesc 复制自 src（避免 MSAA mismatch 引发 RG 执行错误）
///
/// IMPORTANT:
/// - 本 Feature 需要配合 SRP Blit 风格 shader（include Blit.hlsl, 使用 FragBlit 从 _BlitTexture 取 base）
/// - Renderer Data 建议设置：Compatibility -> Intermediate Texture = Always（避免 BackBuffer 作为 blit source）
/// </summary>
public class DepthRevealOverlayRendererFeature : ScriptableRendererFeature
{
    [System.Serializable]
    public class Settings
    {
        public Material material;
        public RenderPassEvent passEvent = RenderPassEvent.AfterRenderingTransparents;
        public bool warnOnBackBuffer = true;
    }

    public Settings settings = new Settings();
    private DepthRevealOverlayPass _pass;

    public override void Create()
    {
        _pass = new DepthRevealOverlayPass(settings);
        _pass.renderPassEvent = settings.passEvent;
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        if (settings.material == null) return;
        if (renderingData.cameraData.isPreviewCamera) return;

        _pass.renderPassEvent = settings.passEvent;
        renderer.EnqueuePass(_pass);
    }

    private sealed class DepthRevealOverlayPass : ScriptableRenderPass
    {
        private readonly Settings _settings;
        public DepthRevealOverlayPass(Settings settings) { _settings = settings; }

        // Compatibility Mode 非目标：留空
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData) { }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            if (_settings.material == null) return;

            var resourceData = frameData.Get<UniversalResourceData>();

            if (resourceData.isActiveTargetBackBuffer)
            {
                if (_settings.warnOnBackBuffer)
                {
                    Debug.LogWarning("[DepthRevealOverlay] active target is BackBuffer. Set URP Renderer Data -> Compatibility -> Intermediate Texture = Always.");
                }
                return;
            }

            TextureHandle srcColor = resourceData.activeColorTexture;

            // Copy src descriptor to keep MSAA/sample count consistent.
            TextureDesc desc = renderGraph.GetTextureDesc(srcColor);
            desc.name = "_MR_DepthRevealOverlay_Color";
            desc.depthBufferBits = 0;

            TextureHandle dstColor = renderGraph.CreateTexture(desc);

            var blitParams = new RenderGraphUtils.BlitMaterialParameters(srcColor, dstColor, _settings.material, 0);
            renderGraph.AddBlitPass(blitParams, "MR Depth Reveal Overlay (RG)");

            resourceData.cameraColor = dstColor;
        }
    }
}
