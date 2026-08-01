using System;
using System.IO;
using System.IO.Compression;

/// <summary>
/// Immutable lookup tables used by the Directional Marching Cubes reference
/// implementation from Splietker et al. (IROS 2019).
///
/// The payloads are losslessly compressed copies of kTriangleVertexEdge,
/// kIndexDecomposition and kIndexDirectionCompatibility from the authors'
/// MeshHashingDTSDF reference source.  Keeping the published tables intact
/// avoids silently replacing their topology decisions with a locally derived
/// MC/MC33 variant.
/// </summary>
internal static class ScanCoverPaperDmcTables
{
    internal const int DirectionCount = 6;
    internal const int TriangleEntriesPerIndex = 16;
    internal const int ComponentsPerIndex = 4;

    private const string TriangleEdgesGzipBase64 =
        "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";

    private const string IndexDecompositionGzipBase64 =
        "H4sIAAAAAAAC/03TyUtVYRzG8ZuKJmYpLkJEDAVdBKUGSUR5F6KbhmWrSCpq0WRaSWR1l0maSouKpj8gKC1oHu4itLCRbCIbqDCb59GR0/M9D9fN531+73n98TvvPZHI2P+/CZiEyZjifa1SyWk4EdN9XpUM8iTMxMk4BbMw2+dVyXF/5Wz3V57q/sq5Cf2DSp77az/f/ZULyNM8n1aF5CL3Vz2YP1gVUy9xf+2n+7zmn+7+qmfiDOozsRTLsBxnub+eKGP+2dQr3F/7c3xe/ee6v/bnkedjJUY9v1ZJmIxV3teqmlzj/lHmD8+rkkFewHMLcREu9v1Hud/wvCo57h/ef9hfeUnC/QdP5ib0Dyp57q/nl7q/8jJyrefTajl5Ba70vlaryKvdv5b5w/N6/2vYX4vrcD1uwDrciPXYgJtwM27BRvfXf9jq8+q/jdyE23EH7sSY54vx/mLcv0zB4PcfrFLJabiL883+/lTZTW7BVtyDbdiOHT6vyl7//pU7/P0r72N/Px7weVUOkg/hYTyCR/39h/NpVUgu8vyq6/uPRYqpl3h+fdfN/v41/zHPr3orHqd+AjuxC0/iKc+vJ7qY/zT1M3gWz+F5vIAX8RJexisY9/3Huf849y+vel+rbnIPXsPrfv+q9JJv4E28hbfxDt71eVXukfvwPj7Ah/gIH/u8Kk/I/fgUn+FzfOH5tHpJfoWvva/VAPkNDuJbfIfv8QN+xE/4Gb/gV/yG3/EH/sRf+Bv/4F/8h0M4jCM4imM4DrbeCAsACAAA";

    private const string DirectionCompatibilityGzipBase64 =
        "H4sIAAAAAAAC/51U2xbDIAhL+v8fvSm3gK4P057VUW6SALkW1gPsN/g8dt57LS5ZfA29LdlfzbY0Yj/P8rS2vc2V6yP+5sEeEzUdNjmanCmzjPIW4rvyEusIV4emH1aRe8WyWzV9qH+KC83/m10ZbUWvCltk/7VqijOKHjHjK17UfVMtieU0sXhZnGdHPm5MQS/zXXdJdqSlxqaWjMogCv55Dv+NU47+a6pNwGLQRfvqAojekPvlnZs8u6UYJIEP/1XD6MCsT/E//CennYl/AugRfpGDmGQCBhFPk1Mrev9gImM2SP2TGUdfVAkwIbPuus2fxpcB/pwmEij5hY5ZTEYovg1OtmGmch6T5pxGLs/eZ80bSmY1NypcdYpqKYJajW27J/8HuCOVUAAGAAA=";

    private static readonly sbyte[] TriangleEdges =
        DecodeSByteTable(TriangleEdgesGzipBase64, 256 * TriangleEntriesPerIndex);
    private static readonly short[] IndexDecomposition =
        DecodeInt16Table(IndexDecompositionGzipBase64, 256 * ComponentsPerIndex);
    private static readonly byte[] DirectionCompatibility =
        DecodeByteTable(DirectionCompatibilityGzipBase64, 256 * DirectionCount);

    internal static int TriangleEdge(int index, int entry)
    {
        return TriangleEdges[index * TriangleEntriesPerIndex + entry];
    }

    internal static int Component(int index, int component)
    {
        return IndexDecomposition[index * ComponentsPerIndex + component];
    }

    internal static int Compatibility(int index, int paperDirection)
    {
        return DirectionCompatibility[index * DirectionCount + paperDirection];
    }

    private static byte[] Inflate(string payload, int expectedBytes)
    {
        byte[] compressed = Convert.FromBase64String(payload);
        using (MemoryStream input = new MemoryStream(compressed, false))
        using (GZipStream gzip = new GZipStream(input, CompressionMode.Decompress))
        using (MemoryStream output = new MemoryStream(expectedBytes))
        {
            gzip.CopyTo(output);
            byte[] result = output.ToArray();
            if (result.Length != expectedBytes)
            {
                throw new InvalidDataException(
                    "Directional MC reference table length mismatch: " +
                    result.Length + " != " + expectedBytes);
            }
            return result;
        }
    }

    private static byte[] DecodeByteTable(string payload, int count)
    {
        return Inflate(payload, count);
    }

    private static sbyte[] DecodeSByteTable(string payload, int count)
    {
        byte[] raw = Inflate(payload, count);
        sbyte[] result = new sbyte[count];
        Buffer.BlockCopy(raw, 0, result, 0, raw.Length);
        return result;
    }

    private static short[] DecodeInt16Table(string payload, int count)
    {
        byte[] raw = Inflate(payload, count * sizeof(short));
        short[] result = new short[count];
        Buffer.BlockCopy(raw, 0, result, 0, raw.Length);
        return result;
    }
}
