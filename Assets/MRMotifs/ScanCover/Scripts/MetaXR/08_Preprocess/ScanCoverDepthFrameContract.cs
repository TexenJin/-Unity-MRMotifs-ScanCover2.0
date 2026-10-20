using UnityEngine;

/// <summary>
/// Application-side contract for a Meta Environment Depth observation.
/// The provider reprojection matrix already contains the depth camera pose and
/// calibration for the texture generation that produced it.  Recovering the
/// observing eye from that matrix prevents a later Camera.main pose from being
/// mixed into the same TSDF observation.
/// </summary>
public static class ScanCoverDepthFrameContract
{
    public static bool TryRecoverEyeWorldPosition(
        Matrix4x4 worldToDepthClip,
        out Vector3 eyeWorldPosition)
    {
        eyeWorldPosition = Vector3.zero;

        // The projective camera centre is the right null-space of the first,
        // second and fourth rows of the world-to-clip matrix.  This is the same
        // cofactor construction previously used by the read-only provenance
        // audit, now promoted to the production input contract.
        float x = Determinant3x3(
            worldToDepthClip.m01, worldToDepthClip.m02, worldToDepthClip.m03,
            worldToDepthClip.m11, worldToDepthClip.m12, worldToDepthClip.m13,
            worldToDepthClip.m31, worldToDepthClip.m32, worldToDepthClip.m33);
        float y = -Determinant3x3(
            worldToDepthClip.m00, worldToDepthClip.m02, worldToDepthClip.m03,
            worldToDepthClip.m10, worldToDepthClip.m12, worldToDepthClip.m13,
            worldToDepthClip.m30, worldToDepthClip.m32, worldToDepthClip.m33);
        float z = Determinant3x3(
            worldToDepthClip.m00, worldToDepthClip.m01, worldToDepthClip.m03,
            worldToDepthClip.m10, worldToDepthClip.m11, worldToDepthClip.m13,
            worldToDepthClip.m30, worldToDepthClip.m31, worldToDepthClip.m33);
        float w = -Determinant3x3(
            worldToDepthClip.m00, worldToDepthClip.m01, worldToDepthClip.m02,
            worldToDepthClip.m10, worldToDepthClip.m11, worldToDepthClip.m12,
            worldToDepthClip.m30, worldToDepthClip.m31, worldToDepthClip.m32);

        if (!float.IsFinite(w) || Mathf.Abs(w) <= 0.00000001f)
            return false;

        eyeWorldPosition = new Vector3(x / w, y / w, z / w);
        return IsFinite(eyeWorldPosition);
    }

    private static float Determinant3x3(
        float a00, float a01, float a02,
        float a10, float a11, float a12,
        float a20, float a21, float a22)
    {
        return a00 * (a11 * a22 - a12 * a21) -
               a01 * (a10 * a22 - a12 * a20) +
               a02 * (a10 * a21 - a11 * a20);
    }

    private static bool IsFinite(Vector3 value)
    {
        return float.IsFinite(value.x) &&
               float.IsFinite(value.y) &&
               float.IsFinite(value.z);
    }
}
