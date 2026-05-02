using UnityEngine;

public class ScanCoverReferenceFrameRegistry : MonoBehaviour
{
    [Header("Current session reference frame (Phase A)")]
    public Transform sessionReferenceFrame;   // 指向 [ScanCover] SessionReferenceFrame

    [Header("Future upgrade (Phase B)")]
    public Transform anchorReferenceFrame;    // 未来可接 Spatial Anchor root
    public bool useAnchorFrame = false;

    public Transform CurrentFrame
    {
        get
        {
            if (useAnchorFrame && anchorReferenceFrame != null)
                return anchorReferenceFrame;
            return sessionReferenceFrame;
        }
    }

    // 统一空间转换接口（后续累计系统都走这里）
    public Vector3 WorldToRef(Vector3 worldPos)
    {
        return CurrentFrame != null ? CurrentFrame.InverseTransformPoint(worldPos) : worldPos;
    }

    public Vector3 RefToWorld(Vector3 refPos)
    {
        return CurrentFrame != null ? CurrentFrame.TransformPoint(refPos) : refPos;
    }
}