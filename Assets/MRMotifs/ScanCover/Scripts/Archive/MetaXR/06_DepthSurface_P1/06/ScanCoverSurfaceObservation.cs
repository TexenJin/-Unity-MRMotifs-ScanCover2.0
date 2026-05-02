using MyProject.XR;
using UnityEngine;

public enum ScanCoverSurfaceSupportLayer
{
    None = 0,
    StereoConfirmed = 1,
    MonoSupported = 2,
    TemporallyInferred = 3,
}

/// <summary>
/// Standardized world-space surface observation record shared between
/// depth input producers (07) and intermediate surface systems (06).
/// </summary>
public struct ScanCoverSurfaceObservation
{
    public bool valid;
    public Vector3 worldPos;
    public Vector3 worldNormal;
    public float linearDepth;
    public Eye sourceEye;
    public ScanCoverSurfaceSupportLayer supportLayer;
    public float confidence;
    public int frameIndex;
    public Vector2Int sourcePixel;

    public ScanCoverSurfaceObservation(
        bool valid,
        Vector3 worldPos,
        Vector3 worldNormal,
        float linearDepth,
        Eye sourceEye,
        ScanCoverSurfaceSupportLayer supportLayer,
        float confidence,
        int frameIndex,
        Vector2Int sourcePixel)
    {
        this.valid = valid;
        this.worldPos = worldPos;
        this.worldNormal = worldNormal;
        this.linearDepth = linearDepth;
        this.sourceEye = sourceEye;
        this.supportLayer = supportLayer;
        this.confidence = confidence;
        this.frameIndex = frameIndex;
        this.sourcePixel = sourcePixel;
    }
}
