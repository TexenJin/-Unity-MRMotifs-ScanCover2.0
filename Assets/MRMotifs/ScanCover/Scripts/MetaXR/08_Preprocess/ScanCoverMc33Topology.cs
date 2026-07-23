using System;
using System.Collections.Generic;

// Topology-only C# port of the MC33 v5.3 case selector by David Vega.
// The original implementation is MIT licensed and accompanies:
// Vega, Abache and Coll, "A Fast and Memory-Saving Marching Cubes 33
// implementation with the correct interior test", JCGT 8(3), 2019.
//
// ScanCover deliberately consumes only the selected triangle pattern.  Edge and
// optional centre vertices are reduced to connected components; geometry remains
// positioned by the existing Surface Nets extractor.
internal static class ScanCoverMc33Topology
{
    internal struct Result
    {
        public int ComponentCount;
        public int AmbiguousFaces;
        public int InteriorTests;
        public int PatternOffset;
    }

    // Current ScanCover cube edge -> MC33 v5.3 edge.
    private static readonly int[] CurrentToMcEdge = { 8, 4, 9, 0, 11, 6, 10, 2, 3, 7, 5, 1 };

    // Generated verbatim from the MIT-licensed mc33cpp.h initializer.  The first
    // 256 entries classify sign masks; the remainder store selected triangle patterns.
    private static readonly ushort[] Table =
    {
        0X0000, 0X0885, 0X0886, 0X0895, 0X0883, 0X1816, 0X089D, 0X0943, 0X0884, 0X0897, 0X1814, 0X0916,
        0X0891, 0X094C, 0X091F, 0X048F, 0X0882, 0X089B, 0X1808, 0X0934, 0X2803, 0X3817, 0X3814, 0X0525,
        0X180E, 0X0928, 0X4802, 0X049D, 0X3815, 0X0541, 0X6004, 0X0110, 0X0881, 0X180A, 0X0899, 0X0922,
        0X1806, 0X4800, 0X0913, 0X0499, 0X2802, 0X380E, 0X3811, 0X053D, 0X380D, 0X6002, 0X0521, 0X010D,
        0X088B, 0X0946, 0X0937, 0X0493, 0X3813, 0X6003, 0X0545, 0X012E, 0X380F, 0X0531, 0X600A, 0X011C,
        0X5001, 0X3001, 0X3009, 0X0087, 0X0880, 0X2801, 0X1804, 0X3807, 0X088F, 0X380B, 0X092B, 0X0539,
        0X1802, 0X3806, 0X4806, 0X6001, 0X0925, 0X051D, 0X0495, 0X010A, 0X1812, 0X380A, 0X4805, 0X6009,
        0X3816, 0X5002, 0X6008, 0X3010, 0X4801, 0X6000, 0X7000, 0X4003, 0X6006, 0X3004, 0X4007, 0X1010,
        0X0889, 0X3808, 0X093A, 0X052D, 0X0949, 0X6005, 0X0491, 0X0131, 0X380C, 0X5000, 0X600B, 0X3012,
        0X0549, 0X3002, 0X0119, 0X008D, 0X0907, 0X0535, 0X04A1, 0X013D, 0X0529, 0X3005, 0X0140, 0X0093,
        0X6007, 0X3000, 0X4004, 0X1000, 0X3003, 0X2000, 0X100C, 0X007F, 0X0380, 0X0109, 0X021A, 0X0B32,
        0X0945, 0X0748, 0X07B6, 0X06A5, 0X1189, 0X0138, 0X129A, 0X0092, 0X1B3A, 0X031A, 0X12B0, 0X0B80,
        0X1045, 0X0105, 0X1975, 0X0987, 0X1340, 0X0374, 0X1BA5, 0X07B5, 0X1486, 0X0B68, 0X1615, 0X0216,
        0X1726, 0X0732, 0X146A, 0X094A, 0X1945, 0X0038, 0X1109, 0X0748, 0X16A5, 0X0109, 0X1945, 0X021A,
        0X16A5, 0X032B, 0X17B6, 0X021A, 0X17B6, 0X0038, 0X1B32, 0X0748, 0X121A, 0X0038, 0X1B32, 0X0109,
        0X16A5, 0X0874, 0X1945, 0X07B6, 0X1905, 0X1035, 0X1453, 0X0843, 0X1974, 0X1917, 0X1871, 0X0081,
        0X1605, 0X1950, 0X116A, 0X0106, 0X1A45, 0X1942, 0X124A, 0X0192, 0X13A5, 0X1B56, 0X132A, 0X0B35,
        0X11A6, 0X17B2, 0X1217, 0X0671, 0X1786, 0X1806, 0X1B60, 0X03B0, 0X1834, 0X1324, 0X1742, 0X0B72,
        0X123A, 0X138A, 0X11A8, 0X0018, 0X1129, 0X12B9, 0X109B, 0X030B, 0X14A5, 0X1A86, 0X148A, 0X0768,
        0X1B65, 0X1794, 0X17B9, 0X059B, 0X16A5, 0X0380, 0X17B6, 0X0109, 0X121A, 0X0748, 0X1945, 0X0B32,
        0X10A5, 0X1805, 0X1863, 0X1685, 0X16A3, 0X03A0, 0X1796, 0X11B6, 0X1169, 0X1970, 0X17B0, 0X00B1,
        0X174A, 0X17A2, 0X1872, 0X1A41, 0X1481, 0X0182, 0X12B5, 0X1B34, 0X1943, 0X15B4, 0X1592, 0X0293,
        0X1B9A, 0X1930, 0X09B3, 0X180A, 0X101A, 0X0B8A, 0X189B, 0X1B12, 0X0B91, 0X128A, 0X1382, 0X09A8,
        0X1246, 0X1419, 0X0421, 0X18A5, 0X1485, 0X0BA8, 0X1026, 0X1067, 0X0078, 0X1845, 0X1538, 0X0135,
        0X176A, 0X1A87, 0X098A, 0X1715, 0X11B2, 0X017B, 0X1175, 0X1708, 0X0710, 0X1426, 0X1832, 0X0248,
        0X106A, 0X1460, 0X010A, 0X1149, 0X1174, 0X0371, 0X142B, 0X174B, 0X0024, 0X12A5, 0X1532, 0X0735,
        0X1635, 0X136B, 0X0153, 0X1695, 0X1609, 0X0206, 0X1935, 0X1390, 0X0753, 0X13B6, 0X1360, 0X0406,
        0X19BA, 0X17B4, 0X0B94, 0X17A6, 0X171A, 0X0317, 0X1A25, 0X1245, 0X0042, 0X1965, 0X19B6, 0X08B9,
        0X146A, 0X1A94, 0X0380, 0X16A5, 0X1189, 0X0138, 0X16A5, 0X180B, 0X002B, 0X1BA5, 0X157B, 0X0038,
        0X1615, 0X1621, 0X0803, 0X16A5, 0X1034, 0X0374, 0X18B6, 0X1648, 0X0109, 0X1BA5, 0X17B5, 0X0109,
        0X129A, 0X17B6, 0X0092, 0X17B6, 0X1189, 0X0138, 0X1726, 0X1732, 0X0109, 0X1045, 0X17B6, 0X0105,
        0X129A, 0X1209, 0X0748, 0X1975, 0X121A, 0X0879, 0X1486, 0X121A, 0X08B6, 0X131A, 0X1B3A, 0X0874,
        0X121A, 0X1374, 0X0403, 0X1615, 0X1216, 0X0874, 0X1945, 0X12B0, 0X0B80, 0X1945, 0X1B3A, 0X031A,
        0X146A, 0X194A, 0X032B, 0X1975, 0X1987, 0X02B3, 0X1045, 0X1510, 0X0B32, 0X1945, 0X1726, 0X0732,
        0X136A, 0X190A, 0X1094, 0X1804, 0X1684, 0X1386, 0X03A0, 0X1895, 0X11A5, 0X136A, 0X1591, 0X13A1,
        0X1863, 0X0856, 0X10A5, 0X1AB6, 0X102A, 0X1A2B, 0X186B, 0X1568, 0X0058, 0X10A5, 0X1785, 0X187B,
        0X138B, 0X1A3B, 0X103A, 0X0058, 0X1685, 0X1236, 0X1321, 0X1031, 0X1501, 0X1805, 0X0863, 0X10A5,
        0X1456, 0X1376, 0X1674, 0X1054, 0X13A0, 0X0A36, 0X1496, 0X1948, 0X1098, 0X1B08, 0X110B, 0X161B,
        0X0169, 0X1795, 0X11A5, 0X11BA, 0X1915, 0X1097, 0X1B07, 0X00B1, 0X19A6, 0X16A2, 0X1B62, 0X10B2,
        0X17B0, 0X1970, 0X0796, 0X1796, 0X113B, 0X1B38, 0X17B8, 0X1978, 0X1169, 0X01B6, 0X1796, 0X1730,
        0X1032, 0X1102, 0X1612, 0X1916, 0X0970, 0X1165, 0X1745, 0X1756, 0X1704, 0X1B61, 0X10B1, 0X0B07,
        0X149A, 0X1208, 0X1809, 0X1489, 0X174A, 0X127A, 0X0728, 0X19A5, 0X175A, 0X11A9, 0X1819, 0X1218,
        0X1728, 0X027A, 0X14A6, 0X18B2, 0X12B6, 0X1A26, 0X11A4, 0X1814, 0X0182, 0X141A, 0X1B7A, 0X17B3,
        0X1873, 0X1183, 0X1481, 0X04A7, 0X141A, 0X1014, 0X1103, 0X1213, 0X1723, 0X1A27, 0X04A7, 0X1645,
        0X1415, 0X1746, 0X1276, 0X1872, 0X1182, 0X0814, 0X1925, 0X1B84, 0X1480, 0X1940, 0X1290, 0X1B52,
        0X05B4, 0X19A5, 0X1319, 0X191A, 0X1B5A, 0X145B, 0X134B, 0X0439, 0X1B6A, 0X1B46, 0X12BA, 0X192A,
        0X1329, 0X1439, 0X034B, 0X12B5, 0X1398, 0X1387, 0X1B37, 0X15B7, 0X1925, 0X0293, 0X1B45, 0X1125,
        0X1210, 0X1320, 0X1430, 0X1B34, 0X0B52, 0X1265, 0X1574, 0X1567, 0X1347, 0X1943, 0X1293, 0X0925,
        0X136A, 0X190A, 0X13A0, 0X1684, 0X0386, 0X1685, 0X136A, 0X1895, 0X113A, 0X0863, 0X10A5, 0X1856,
        0X102A, 0X186B, 0X0058, 0X10A5, 0X1785, 0X1058, 0X1A3B, 0X003A, 0X1685, 0X1236, 0X1863, 0X1501,
        0X0805, 0X10A5, 0X1A36, 0X1054, 0X1376, 0X03A0, 0X1496, 0X1169, 0X1B08, 0X110B, 0X061B, 0X1795,
        0X11BA, 0X10B1, 0X1097, 0X0B07, 0X19A6, 0X1796, 0X10B2, 0X17B0, 0X0970, 0X1796, 0X113B, 0X161B,
        0X1978, 0X0169, 0X1796, 0X1126, 0X1730, 0X1970, 0X0916, 0X1165, 0X1704, 0X1B07, 0X1B61, 0X00B1,
        0X149A, 0X1208, 0X1728, 0X174A, 0X027A, 0X1A75, 0X127A, 0X1819, 0X1218, 0X0728, 0X14A6, 0X18B2,
        0X1182, 0X11A4, 0X0814, 0X174A, 0X1B7A, 0X1183, 0X1481, 0X0A41, 0X141A, 0X1014, 0X1723, 0X1A27,
        0X04A7, 0X1415, 0X1276, 0X1814, 0X1872, 0X0182, 0X1925, 0X1B84, 0X15B4, 0X1290, 0X0B52, 0X1AB5,
        0X1319, 0X1439, 0X145B, 0X034B, 0X1B46, 0X192A, 0X134B, 0X1329, 0X0439, 0X12B5, 0X1398, 0X1293,
        0X15B7, 0X0925, 0X12B5, 0X1125, 0X1430, 0X1B34, 0X05B4, 0X1265, 0X1925, 0X1734, 0X1943, 0X0293,
        0X1945, 0X121A, 0X07B6, 0X12B3, 0X1874, 0X0109, 0X16A5, 0X1B32, 0X0874, 0X1945, 0X121A, 0X0038,
        0X1945, 0X17B6, 0X0038, 0X16A5, 0X1B32, 0X0109, 0X16A5, 0X1109, 0X0748, 0X121A, 0X17B6, 0X0380,
        0X1B65, 0X19B5, 0X121A, 0X1794, 0X0B97, 0X1B92, 0X1874, 0X1B30, 0X19B0, 0X0912, 0X14A5, 0X1A86,
        0X1B32, 0X1876, 0X08A4, 0X1945, 0X138A, 0X1801, 0X1A81, 0X0A23, 0X1B65, 0X1380, 0X1794, 0X1B97,
        0X09B5, 0X16A5, 0X1129, 0X1B92, 0X1B30, 0X09B0, 0X14A5, 0X1876, 0X1109, 0X18A4, 0X0A86, 0X17B6,
        0X123A, 0X18A3, 0X1801, 0X0A81, 0X1A45, 0X17B6, 0X124A, 0X1219, 0X0429, 0X1109, 0X1483, 0X1243,
        0X12B7, 0X0427, 0X16A5, 0X1274, 0X12B7, 0X1483, 0X0243, 0X1A45, 0X1038, 0X1219, 0X1429, 0X024A,
        0X1945, 0X1806, 0X1678, 0X103B, 0X060B, 0X1605, 0X1095, 0X116A, 0X132B, 0X0061, 0X1605, 0X1095,
        0X116A, 0X1748, 0X0061, 0X1786, 0X121A, 0X103B, 0X160B, 0X0068, 0X1945, 0X11A6, 0X1716, 0X17B2,
        0X0172, 0X132B, 0X1749, 0X1108, 0X1718, 0X0179, 0X13A5, 0X1B56, 0X1748, 0X135B, 0X032A, 0X1905,
        0X121A, 0X1350, 0X1384, 0X0534, 0X1905, 0X1534, 0X17B6, 0X1384, 0X0350, 0X13A5, 0X1B56, 0X1109,
        0X132A, 0X035B, 0X16A5, 0X1749, 0X1179, 0X1108, 0X0718, 0X11A6, 0X1380, 0X17B2, 0X1172, 0X0716,
        0X1C65, 0X1C5A, 0X17C4, 0X1C7B, 0X1C94, 0X1C19, 0X1C21, 0X1CA2, 0X0CB6, 0X1C74, 0X1CB7, 0X1C2B,
        0X11C9, 0X1C12, 0X1C09, 0X1C30, 0X1C83, 0X0C48, 0X1CA5, 0X1C6A, 0X1C32, 0X1C83, 0X1C48, 0X1C54,
        0X1C76, 0X1CB7, 0X0C2B, 0X1AC5, 0X1C38, 0X1C23, 0X1CA2, 0X1C45, 0X1C94, 0X1C19, 0X1C01, 0X0C80,
        0X1C65, 0X19C5, 0X1CB6, 0X1C3B, 0X1C03, 0X1C80, 0X17C4, 0X1C78, 0X0C94, 0X16C5, 0X1C6A, 0X1C95,
        0X1C09, 0X1C30, 0X1CB3, 0X1C2B, 0X1C12, 0X0CA1, 0X1C95, 0X1C6A, 0X1C10, 0X1CA1, 0X1C48, 0X1C76,
        0X1C87, 0X1C54, 0X0C09, 0X1C1A, 0X17C6, 0X1C01, 0X1C80, 0X1C78, 0X1CB6, 0X1C3B, 0X1C23, 0X0CA2,
        0X1C65, 0X1CA6, 0X1C59, 0X11C2, 0X1CB2, 0X17C4, 0X1C7B, 0X1C94, 0X0C1A, 0X1C2B, 0X11C9, 0X1C12,
        0X1C49, 0X1C74, 0X1C87, 0X1C08, 0X1C30, 0X0CB3, 0X1CA5, 0X1C54, 0X1C76, 0X1C48, 0X1C2A, 0X1C32,
        0X1CB3, 0X1C6B, 0X0C87, 0X1C45, 0X12CA, 0X1C84, 0X1C38, 0X1C23, 0X1C59, 0X1C1A, 0X1C01, 0X0C90,
        0X1C65, 0X1C03, 0X1C90, 0X1C59, 0X1CB6, 0X17C4, 0X1C7B, 0X1C84, 0X0C38, 0X1CA5, 0X1C56, 0X1C09,
        0X1C30, 0X1CB3, 0X1C6B, 0X1C2A, 0X1C12, 0X0C91, 0X1CA5, 0X1C6A, 0X1C54, 0X1C76, 0X1C87, 0X1C08,
        0X11C9, 0X1C10, 0X0C49, 0X1CA6, 0X17C6, 0X1C1A, 0X1C01, 0X1C80, 0X1C38, 0X1C23, 0X1CB2, 0X0C7B,
        0X1AC5, 0X1CA6, 0X1C94, 0X1C19, 0X1C21, 0X1CB2, 0X1C7B, 0X1C67, 0X0C45, 0X1C2B, 0X11C9, 0X1C49,
        0X1C74, 0X1CB7, 0X1C32, 0X1C83, 0X1C08, 0X0C10, 0X1CA5, 0X1C56, 0X1C2A, 0X1C32, 0X1C83, 0X1C48,
        0X1C74, 0X1CB7, 0X0C6B, 0X1AC5, 0X12CA, 0X1C45, 0X1C84, 0X1C38, 0X1C03, 0X1C90, 0X1C19, 0X0C21,
        0X1C45, 0X1CB6, 0X1C3B, 0X1C03, 0X1C90, 0X1C59, 0X1C84, 0X1C78, 0X0C67, 0X16C5, 0X1C2A, 0X13CB,
        0X1C6B, 0X1C95, 0X1C09, 0X1C10, 0X1CA1, 0X0C32, 0X16C5, 0X1C6A, 0X1C95, 0X1C74, 0X17C8, 0X1C08,
        0X1C10, 0X1CA1, 0X0C49, 0X1CA6, 0X1C80, 0X1C78, 0X1C67, 0X1C1A, 0X1C21, 0X1CB2, 0X1C3B, 0X0C03,
        0X1A65, 0X1419, 0X11B2, 0X17B4, 0X04B1, 0X174B, 0X1149, 0X12B1, 0X11B4, 0X0830, 0X12A5, 0X1485,
        0X1B76, 0X1832, 0X0582, 0X1A25, 0X1238, 0X1458, 0X1528, 0X0019, 0X1965, 0X1390, 0X1B63, 0X1693,
        0X0784, 0X1695, 0X112A, 0X1093, 0X1B36, 0X0396, 0X1495, 0X176A, 0X110A, 0X1870, 0X07A0, 0X17A6,
        0X1780, 0X11A0, 0X1A70, 0X03B2, 0X1465, 0X1459, 0X121A, 0X1AB2, 0X1BA6, 0X17B6, 0X1476, 0X15A1,
        0X0951, 0X1084, 0X1748, 0X18B7, 0X1B83, 0X12B3, 0X1109, 0X1130, 0X1123, 0X0904, 0X16A5, 0X132B,
        0X1B83, 0X1874, 0X18B7, 0X1576, 0X1547, 0X16B2, 0X0A62, 0X1945, 0X115A, 0X1038, 0X1023, 0X1201,
        0X1A21, 0X1519, 0X1908, 0X0498, 0X1465, 0X1380, 0X1890, 0X1984, 0X1594, 0X1647, 0X1B67, 0X1783,
        0X0B73, 0X16A5, 0X1A95, 0X19A1, 0X1091, 0X1312, 0X1301, 0X1B32, 0X12A6, 0X0B26, 0X16A5, 0X1109,
        0X19A1, 0X1A95, 0X1754, 0X1765, 0X1874, 0X1490, 0X0840, 0X1BA6, 0X1380, 0X1378, 0X173B, 0X167B,
        0X1AB2, 0X11A2, 0X1230, 0X0120, 0X1B9A, 0X09B8, 0X1426, 0X0024, 0X1375, 0X0135, 0X17A6, 0X1180,
        0X1781, 0X0A71, 0X1B42, 0X1129, 0X1492, 0X04B7, 0X1A35, 0X13A2, 0X1853, 0X0584, 0X1965, 0X13B0,
        0X10B6, 0X0069, 0X146A, 0X194A, 0X1028, 0X082B, 0X17A5, 0X1189, 0X1381, 0X07BA, 0X1625, 0X1340,
        0X1743, 0X0521, 0X1846, 0X190A, 0X186B, 0X002A, 0X1795, 0X11BA, 0X1789, 0X03B1, 0X1015, 0X1236,
        0X1540, 0X0763, 0X126A, 0X1029, 0X12A9, 0X1B62, 0X16B8, 0X1468, 0X1489, 0X0809, 0X19A5, 0X1789,
        0X1957, 0X11A9, 0X1A13, 0X1BA3, 0X1B37, 0X0387, 0X1405, 0X1756, 0X1015, 0X1021, 0X1320, 0X1237,
        0X1627, 0X0745, 0X146A, 0X1A94, 0X1904, 0X1408, 0X1B80, 0X1B02, 0X1AB2, 0X0A6B, 0X1BA5, 0X1189,
        0X1138, 0X13B8, 0X18B7, 0X157B, 0X115A, 0X0195, 0X1465, 0X1156, 0X1621, 0X1231, 0X1130, 0X1403,
        0X1437, 0X0647, 0X19CA, 0X1AC6, 0X190C, 0X102C, 0X12BC, 0X18CB, 0X184C, 0X046C, 0X1C95, 0X1CBA,
        0X1C7B, 0X1C57, 0X19C8, 0X1C38, 0X1C13, 0X01CA, 0X1C15, 0X12C6, 0X1C21, 0X14C5, 0X1C76, 0X17C3,
        0X1C03, 0X0C40, 0X1C46, 0X1C2A, 0X1C94, 0X1CA9, 0X12C0, 0X1C80, 0X1CB8, 0X0BC6, 0X1CA5, 0X17C5,
        0X1CBA, 0X1C3B, 0X11C9, 0X13C1, 0X1C89, 0X08C7, 0X16C5, 0X12C6, 0X123C, 0X174C, 0X137C, 0X10C4,
        0X101C, 0X015C, 0X16B5, 0X1B80, 0X15B0, 0X0150, 0X1786, 0X1189, 0X1126, 0X0681, 0X129A, 0X1974,
        0X1792, 0X0372, 0X14A5, 0X13BA, 0X13A4, 0X0340, 0X1975, 0X1902, 0X1927, 0X072B, 0X116A, 0X1846,
        0X1861, 0X0813, 0X176A, 0X1A90, 0X17A0, 0X0370, 0X1B1A, 0X1140, 0X11B4, 0X074B, 0X1125, 0X1285,
        0X12B8, 0X0458, 0X1695, 0X1236, 0X1396, 0X0389, 0X1496, 0X1139, 0X1369, 0X063B, 0X12A5, 0X1785,
        0X1825, 0X0802, 0X1246, 0X1192, 0X1429, 0X0038, 0X1945, 0X181A, 0X1180, 0X0B8A, 0X16A5, 0X1129,
        0X1B92, 0X089B, 0X16A5, 0X1749, 0X1179, 0X0371, 0X123A, 0X17B6, 0X18A3, 0X09A8, 0X16A5, 0X1274,
        0X172B, 0X0024, 0X1715, 0X17B2, 0X1172, 0X0380, 0X19BA, 0X1794, 0X1B97, 0X0803, 0X1406, 0X121A,
        0X103B, 0X060B, 0X1905, 0X121A, 0X1350, 0X0753, 0X1345, 0X17B6, 0X1384, 0X0135, 0X1945, 0X1786,
        0X1068, 0X0260, 0X1246, 0X1384, 0X1342, 0X0190, 0X1A45, 0X14A8, 0X18AB, 0X0019, 0X16B5, 0X15B9,
        0X112A, 0X09B8, 0X1495, 0X116A, 0X1617, 0X0713, 0X1786, 0X168A, 0X1A89, 0X023B, 0X14A5, 0X1B76,
        0X1A42, 0X0240, 0X1715, 0X1180, 0X1817, 0X0B23, 0X19BA, 0X13B0, 0X10B9, 0X0784, 0X11A6, 0X1160,
        0X1064, 0X03B2, 0X1A35, 0X123A, 0X1537, 0X0019, 0X1B65, 0X1B53, 0X1351, 0X0784, 0X1065, 0X1905,
        0X1602, 0X0784, 0X1846, 0X1948, 0X1980, 0X1901, 0X1863, 0X1362, 0X1132, 0X0103, 0X1AB5, 0X1159,
        0X11A5, 0X1190, 0X15B4, 0X14B8, 0X1048, 0X0094, 0X1685, 0X126A, 0X1589, 0X11A5, 0X12B6, 0X16B8,
        0X12A1, 0X0159, 0X19A5, 0X1A36, 0X191A, 0X1A13, 0X1954, 0X1637, 0X1467, 0X0456, 0X126A, 0X1738,
        0X137B, 0X1789, 0X13B2, 0X1796, 0X169A, 0X02B6, 0X10A5, 0X1B6A, 0X1745, 0X1756, 0X1540, 0X176B,
        0X1A02, 0X0BA2, 0X1015, 0X1102, 0X1203, 0X123B, 0X1058, 0X1857, 0X1B87, 0X0B38, 0X13BA, 0X1784,
        0X137B, 0X1738, 0X13A0, 0X10A9, 0X1409, 0X0480, 0X1B6A, 0X1BA2, 0X1A64, 0X1B23, 0X1A41, 0X1140,
        0X1310, 0X0321, 0X19A5, 0X1320, 0X1021, 0X1237, 0X1019, 0X127A, 0X1A75, 0X091A, 0X1165, 0X1456,
        0X1467, 0X1478, 0X161B, 0X1B13, 0X18B3, 0X087B, 0X1265, 0X1809, 0X1894, 0X1902, 0X1847, 0X1925,
        0X1756, 0X0745, 0X1946, 0X1312, 0X1013, 0X1621, 0X1803, 0X1961, 0X1498, 0X0908, 0X1945, 0X115A,
        0X1084, 0X1904, 0X1B80, 0X11B0, 0X1AB1, 0X0195, 0X16A5, 0X1195, 0X1A15, 0X1891, 0X1281, 0X1B82,
        0X1B26, 0X02A6, 0X16A5, 0X1764, 0X1546, 0X1374, 0X1934, 0X1139, 0X119A, 0X095A, 0X12A6, 0X1B26,
        0X19A2, 0X17B6, 0X1392, 0X1893, 0X1837, 0X03B7, 0X16A5, 0X1B2A, 0X16BA, 0X102B, 0X1754, 0X170B,
        0X1407, 0X0765, 0X1B25, 0X178B, 0X13B8, 0X157B, 0X1038, 0X1152, 0X1120, 0X0230, 0X194A, 0X1490,
        0X1840, 0X1380, 0X17A4, 0X1BA7, 0X1B73, 0X0783, 0X1BA6, 0X1130, 0X1231, 0X1403, 0X1A21, 0X1B43,
        0X164B, 0X0B2A, 0X1A95, 0X119A, 0X1759, 0X121A, 0X1079, 0X1370, 0X1302, 0X0012, 0X1465, 0X13B8,
        0X178B, 0X1138, 0X167B, 0X1418, 0X1514, 0X0476, 0X1945, 0X1765, 0X1754, 0X1267, 0X1827, 0X1028,
        0X1089, 0X0849, 0X12C6, 0X1C19, 0X11C0, 0X13C2, 0X180C, 0X18C3, 0X194C, 0X046C, 0X1AC5, 0X119C,
        0X14C9, 0X145C, 0X1ABC, 0X10C8, 0X1B8C, 0X01C0, 0X1CA5, 0X156C, 0X12AC, 0X16BC, 0X1B8C, 0X11C9,
        0X189C, 0X02C1, 0X1CA5, 0X1AC6, 0X14C5, 0X16C7, 0X137C, 0X11C9, 0X113C, 0X09C4, 0X12CA, 0X1CB6,
        0X13BC, 0X178C, 0X167C, 0X189C, 0X19AC, 0X03C2, 0X1CA5, 0X154C, 0X176C, 0X1AC6, 0X140C, 0X1BC2,
        0X102C, 0X07CB, 0X1C15, 0X123C, 0X101C, 0X18C3, 0X180C, 0X1BC7, 0X157C, 0X02CB, 0X1CBA, 0X1C38,
        0X17C4, 0X1BC7, 0X1C84, 0X1C90, 0X1CA9, 0X03C0, 0X16CA, 0X11C2, 0X1C03, 0X12CB, 0X1C3B, 0X1C40,
        0X1C64, 0X0AC1, 0X1AC5, 0X1C19, 0X11C2, 0X13C0, 0X10C9, 0X1C37, 0X1C75, 0X02CA, 0X1C45, 0X17C6,
        0X178C, 0X1BC3, 0X16CB, 0X113C, 0X151C, 0X04C8, 0X1C45, 0X1C26, 0X184C, 0X190C, 0X159C, 0X102C,
        0X17C6, 0X08C7, 0X146C, 0X190C, 0X184C, 0X13C0, 0X138C, 0X11C2, 0X162C, 0X09C1, 0X1C45, 0X10C9,
        0X14C8, 0X159C, 0X1B8C, 0X11AC, 0X1ABC, 0X01C0, 0X16C5, 0X16AC, 0X15C9, 0X11CA, 0X189C, 0X12BC,
        0X1B8C, 0X02C1, 0X16C5, 0X16AC, 0X195C, 0X1A1C, 0X113C, 0X1C74, 0X137C, 0X09C4, 0X16CA, 0X12CB,
        0X17BC, 0X17C6, 0X19AC, 0X138C, 0X189C, 0X03C2, 0X16C5, 0X15CA, 0X1BC6, 0X1AC2, 0X102C, 0X174C,
        0X140C, 0X07CB, 0X17C5, 0X1C3B, 0X18C7, 0X103C, 0X10C8, 0X121C, 0X115C, 0X02CB, 0X19CA, 0X1C80,
        0X1C94, 0X18C7, 0X1C47, 0X1BC3, 0X1CBA, 0X03C0, 0X12CA, 0X1CB6, 0X1C23, 0X1BC3, 0X1C64, 0X1C01,
        0X1C40, 0X0AC1, 0X19C5, 0X1C1A, 0X11C0, 0X1C90, 0X1C75, 0X13C2, 0X1C37, 0X02CA, 0X1C65, 0X17C4,
        0X1BC7, 0X1B6C, 0X151C, 0X18C3, 0X113C, 0X04C8, 0X1C65, 0X17C4, 0X194C, 0X19C5, 0X126C, 0X180C,
        0X102C, 0X08C7, 0X1945, 0X121A, 0X17B6, 0X0380, 0X1A65, 0X1190, 0X1B23, 0X0784, 0X1B65, 0X1B59,
        0X121A, 0X1380, 0X1794, 0X0B97, 0X1945, 0X17B6, 0X181A, 0X1801, 0X1A38, 0X0A23, 0X1945, 0X121A,
        0X10B6, 0X103B, 0X1680, 0X0678, 0X1945, 0X11A6, 0X1038, 0X1172, 0X17B2, 0X0167, 0X1A45, 0X17B6,
        0X1380, 0X1429, 0X1219, 0X04A2, 0X1A65, 0X1794, 0X1197, 0X13B2, 0X1817, 0X0801, 0X1905, 0X121A,
        0X1345, 0X17B6, 0X1350, 0X0384, 0X1065, 0X1590, 0X11A6, 0X1784, 0X1B23, 0X0016, 0X1B65, 0X135A,
        0X1784, 0X1019, 0X1B53, 0X0A23, 0X1A65, 0X1190, 0X1342, 0X1384, 0X1472, 0X07B2, 0X1A65, 0X1784,
        0X10B9, 0X103B, 0X1B29, 0X0219, 0X1A45, 0X18A6, 0X1190, 0X13B2, 0X14A8, 0X0678, 0X1C65, 0X1C59,
        0X121A, 0X14C9, 0X10C8, 0X1C78, 0X1C47, 0X1C3B, 0X16CB, 0X0C03, 0X1945, 0X1CA6, 0X13C0, 0X11C2,
        0X1CB2, 0X1C3B, 0X1C80, 0X1C78, 0X17C6, 0X0C1A, 0X1C65, 0X1CA6, 0X19C5, 0X11AC, 0X1C21, 0X1CB2,
        0X17C4, 0X1BC7, 0X1C94, 0X0803, 0X1945, 0X12CA, 0X1CB6, 0X1C3B, 0X1C23, 0X1C1A, 0X1C01, 0X1C78,
        0X10C8, 0X0C67, 0X1C65, 0X1A6C, 0X17C4, 0X159C, 0X194C, 0X178C, 0X180C, 0X11AC, 0X11C0, 0X03B2,
        0X1945, 0X1CA6, 0X17BC, 0X18C3, 0X1C23, 0X1CB2, 0X1C67, 0X1C01, 0X1AC1, 0X0C80, 0X1C65, 0X1C5A,
        0X1CB6, 0X12CA, 0X17C4, 0X1C7B, 0X1C19, 0X14C9, 0X1C21, 0X0038, 0X1AC5, 0X1CA6, 0X1C45, 0X17C6,
        0X1C94, 0X1C19, 0X1CB2, 0X11C2, 0X1C7B, 0X0380, 0X1A65, 0X1C3B, 0X17C4, 0X18C7, 0X180C, 0X103C,
        0X1B2C, 0X1C19, 0X121C, 0X094C, 0X1AC5, 0X1C45, 0X17B6, 0X10C8, 0X14C9, 0X1C19, 0X1C01, 0X1C38,
        0X1C23, 0X02CA, 0X1A65, 0X119C, 0X11C0, 0X13C2, 0X138C, 0X180C, 0X194C, 0X17BC, 0X17C4, 0X0B2C,
        0X1AC5, 0X1A6C, 0X1C19, 0X194C, 0X145C, 0X167C, 0X180C, 0X18C7, 0X101C, 0X0B23, 0X1C65, 0X16CA,
        0X12CB, 0X159C, 0X121C, 0X11AC, 0X103C, 0X10C9, 0X13BC, 0X0784, 0X1C45, 0X17C6, 0X1C59, 0X121A,
        0X1C84, 0X1C78, 0X1CB6, 0X1C3B, 0X1C90, 0X03C0, 0X1C65, 0X19C5, 0X121A, 0X1C38, 0X17C4, 0X1BC7,
        0X1C84, 0X1C03, 0X1C90, 0X0CB6, 0X1C45, 0X16CA, 0X10C9, 0X14C8, 0X159C, 0X101C, 0X11AC, 0X167C,
        0X178C, 0X023B, 0X1C65, 0X12CA, 0X13C2, 0X159C, 0X11C0, 0X11AC, 0X13BC, 0X1B6C, 0X190C, 0X0784,
        0X19C5, 0X12CA, 0X1C45, 0X17B6, 0X1AC1, 0X1C01, 0X1C90, 0X1C84, 0X1C23, 0X08C3, 0X1A65, 0X14C8,
        0X10C9, 0X103C, 0X138C, 0X147C, 0X17BC, 0X121C, 0X12CB, 0X019C, 0X1AC5, 0X15C4, 0X17B6, 0X1C19,
        0X11C2, 0X13C0, 0X1C90, 0X1CA2, 0X1C84, 0X0C38, 0X1AC5, 0X1B6C, 0X1C19, 0X1A2C, 0X121C, 0X190C,
        0X103C, 0X1BC3, 0X165C, 0X0784, 0X1AC5, 0X16CA, 0X12CB, 0X145C, 0X167C, 0X17BC, 0X123C, 0X138C,
        0X14C8, 0X0019, 0X1C65, 0X15AC, 0X17C4, 0X17BC, 0X1B6C, 0X1A2C, 0X138C, 0X13C2, 0X184C, 0X0190,
        0X1AC5, 0X1B6C, 0X145C, 0X1C78, 0X1BC3, 0X167C, 0X184C, 0X1A2C, 0X123C, 0X0019, 0X1C65, 0X1C5A,
        0X1C90, 0X1C03, 0X1C38, 0X1C84, 0X1C47, 0X1C7B, 0X1CB6, 0X1CA2, 0X1C21, 0X0C19, 0X1AC5, 0X1A6C,
        0X138C, 0X123C, 0X1B2C, 0X145C, 0X17BC, 0X167C, 0X194C, 0X119C, 0X101C, 0X080C, 0X1C65, 0X1CA6,
        0X1CB2, 0X1C59, 0X1C21, 0X1C1A, 0X1C94, 0X1C47, 0X1C78, 0X1C80, 0X1C03, 0X0C3B, 0X1C45, 0X1B6C,
        0X159C, 0X11AC, 0X101C, 0X190C, 0X184C, 0X178C, 0X167C, 0X13BC, 0X123C, 0X0A2C, 0X1A65, 0X1784,
        0X1B87, 0X18B3, 0X1804, 0X1094, 0X1190, 0X1310, 0X1213, 0X0B23, 0X1945, 0X115A, 0X17B6, 0X1380,
        0X1302, 0X1012, 0X1908, 0X1498, 0X1951, 0X0A21, 0X1A65, 0X19A5, 0X1A91, 0X1A26, 0X12B6, 0X13B2,
        0X1132, 0X1031, 0X1901, 0X0784, 0X1945, 0X1BA6, 0X1380, 0X1837, 0X13B7, 0X1230, 0X1120, 0X121A,
        0X12AB, 0X067B, 0X1465, 0X1BA6, 0X1519, 0X121A, 0X12AB, 0X15A1, 0X1594, 0X1476, 0X17B6, 0X0803,
        0X1A65, 0X13B2, 0X18B3, 0X1574, 0X1B87, 0X1B62, 0X16A2, 0X1756, 0X1847, 0X0190, 0X1465, 0X1459,
        0X121A, 0X1380, 0X1089, 0X1849, 0X1783, 0X1B73, 0X17B6, 0X0764, 0X1A65, 0X1190, 0X1A91, 0X19A5,
        0X1940, 0X1480, 0X1784, 0X1574, 0X1675, 0X03B2, 0X1A65, 0X1380, 0X1219, 0X1942, 0X14B2, 0X07B4,
        0X1A35, 0X1584, 0X17B6, 0X1190, 0X123A, 0X0385, 0X1965, 0X121A, 0X1B03, 0X1B60, 0X1690, 0X0784,
        0X1945, 0X17A6, 0X13B2, 0X1018, 0X1178, 0X01A7,
    };

    internal static bool TryBuildComponents(
        float[] currentValues,
        bool[] currentEdgeActive,
        int[] currentEdgeComponents,
        out Result result)
    {
        result = default(Result);
        if (currentValues == null || currentValues.Length < 8 ||
            currentEdgeActive == null || currentEdgeActive.Length < 12 ||
            currentEdgeComponents == null || currentEdgeComponents.Length < 12)
            return false;

        for (int edge = 0; edge < 12; edge++)
            currentEdgeComponents[edge] = -1;

        // MC33 order: 0(000),1(010),2(011),3(001),4(100),5(110),6(111),7(101).
        // Its scalar v is iso-field, hence -TSDF for iso=0.
        float[] v =
        {
            -currentValues[0], -currentValues[3], -currentValues[7], -currentValues[4],
            -currentValues[1], -currentValues[2], -currentValues[6], -currentValues[5]
        };
        int signMask = 0;
        for (int corner = 0; corner < 8; corner++)
        {
            if (v[corner] < 0f)
                signMask |= 1 << (7 - corner);
        }
        if (signMask == 0 || signMask == 0xFF)
            return false;

        result.AmbiguousFaces = CountAmbiguousFaces(signMask);
        int patternOffset = SelectPattern(v, signMask, ref result.InteriorTests);
        result.PatternOffset = patternOffset;
        if (patternOffset < 0 || patternOffset + 1 >= Table.Length)
            return false;

        int[] parent = new int[13];
        for (int node = 0; node < parent.Length; node++)
            parent[node] = -1;

        int cursor = patternOffset;
        int guard = 0;
        while (++cursor < Table.Length && guard++ < 32)
        {
            int triangle = Table[cursor];
            int a = triangle & 0xF;
            int b = (triangle >> 4) & 0xF;
            int c = (triangle >> 8) & 0xF;
            if (a > 12 || b > 12 || c > 12)
                return false;
            Activate(parent, a);
            Activate(parent, b);
            Activate(parent, c);
            Union(parent, a, b);
            Union(parent, b, c);
            if ((triangle & 0xF000) == 0)
                break;
        }
        if (guard <= 0 || guard >= 32)
            return false;

        Dictionary<int, int> components = new Dictionary<int, int>(4);
        for (int currentEdge = 0; currentEdge < 12; currentEdge++)
        {
            if (!currentEdgeActive[currentEdge])
                continue;
            int mcEdge = CurrentToMcEdge[currentEdge];
            int root = Find(parent, mcEdge);
            if (root < 0)
                return false;
            if (!components.TryGetValue(root, out int component))
            {
                component = components.Count;
                components[root] = component;
            }
            currentEdgeComponents[currentEdge] = component;
        }
        result.ComponentCount = components.Count;
        return result.ComponentCount > 0;
    }

    private static int SelectPattern(float[] v, int i, ref int interiorTests)
    {
        int c;
        bool m;
        if ((i & 0x80) != 0)
        {
            c = Table[i ^ 0xFF];
            m = (c & 0x800) == 0;
        }
        else
        {
            c = Table[i];
            m = (c & 0x800) != 0;
        }
        int k = c & 0x7FF;
        int orientedMask = m ? i : i ^ 0xFF;
        int[] f = new int[6];
        switch (c >> 12)
        {
            case 0:
                return k;
            case 1:
                return ((orientedMask & FaceTest1(v, k >> 2)) != 0 ? 183 + (k << 1) : 159 + k);
            case 2:
                return InteriorTest(v, k, 0, ref interiorTests) != 0 ? 239 + 6 * k : 231 + (k << 1);
            case 3:
                if ((orientedMask & FaceTest1(v, k % 6)) != 0)
                    return 575 + 5 * k;
                return InteriorTest(v, k / 6, 0, ref interiorTests) != 0 ? 407 + 7 * k : 335 + 3 * k;
            case 4:
            {
                int sum = FaceTests(v, orientedMask, f);
                if (sum == -3) return 695 + 3 * k;
                if (sum == -1) return (f[4] + f[5] < 0 ? (f[0] + f[2] < 0 ? 759 : 799) : 719) + 5 * k;
                if (sum == 1) return (f[4] + f[5] < 0 ? 983 : (f[0] + f[2] < 0 ? 839 : 911)) + 9 * k;
                return InteriorTest(v, k >> 1, 0, ref interiorTests) != 0 ? 1095 + 9 * k : 1055 + 5 * k;
            }
            case 5:
            {
                int sum = FaceTests(v, orientedMask, f);
                if (sum == -2)
                {
                    bool connected = (k & 2) != 0
                        ? InteriorTest(v, 0, 0, ref interiorTests) != 0
                        : InteriorTest(v, 0, 0, ref interiorTests) != 0 ||
                          InteriorTest(v, k != 0 ? 1 : 3, 0, ref interiorTests) != 0;
                    return connected ? 1213 + (k << 3) : 1189 + (k << 2);
                }
                if (sum == 0)
                    return (f[2 + k] < 0 ? 1261 : 1285) + (k << 3);
                bool otherConnected = (k & 2) != 0
                    ? InteriorTest(v, 1, 0, ref interiorTests) != 0
                    : InteriorTest(v, 2, 0, ref interiorTests) != 0 ||
                      InteriorTest(v, k != 0 ? 3 : 1, 0, ref interiorTests) != 0;
                return otherConnected ? 1237 + (k << 3) : 1201 + (k << 2);
            }
            case 6:
            {
                int sum = FaceTests(v, orientedMask, f);
                if (sum == -2)
                {
                    int diagonal = (0xDA010C >> (k << 1)) & 3;
                    return InteriorTest(v, diagonal, 0, ref interiorTests) != 0 ? 1453 + (k << 3) : 1357 + (k << 2);
                }
                if (sum == 0)
                    return (f[k >> 1] < 0 ? 1645 : 1741) + (k << 3);
                int otherDiagonal = (0xA7B7E5 >> (k << 1)) & 3;
                return InteriorTest(v, otherDiagonal, 0, ref interiorTests) != 0 ? 1549 + (k << 3) : 1405 + (k << 2);
            }
            default:
            {
                int sum = Math.Abs(FaceTests(v, 165, f));
                if (sum == 0)
                {
                    k = ((f[1] < 0 ? 1 : 0) << 1) | (f[5] < 0 ? 1 : 0);
                    if (f[0] * f[1] == f[5])
                        return 2157 + 12 * k;
                    int interior = InteriorTest(v, k, 1, ref interiorTests);
                    return 2285 + (interior != 0 ? 10 * k - 40 * interior : 6 * k);
                }
                if (sum == 2)
                {
                    int offset = 1917 + 10 * ((f[0] < 0 ? (f[2] > 0 ? 1 : 0) : 12 + (f[2] < 0 ? 1 : 0)) +
                                              (f[1] < 0 ? (f[3] < 0 ? 1 : 0) : 6 + (f[3] > 0 ? 1 : 0)));
                    if (f[4] > 0) offset += 30;
                    return offset;
                }
                if (sum == 4)
                {
                    k = 21 + 11 * f[0] + 4 * f[1] + 3 * f[2] + 2 * f[3] + f[4];
                    if ((k >> 4) != 0)
                        k -= (k & 32) != 0 ? 20 : 10;
                    return 1845 + 3 * k;
                }
                return 1839 + 2 * f[0];
            }
        }
    }

    private static int CountAmbiguousFaces(int i)
    {
        int count = 0;
        int[] masks = { 0xCC, 0x66, 0x33, 0x99, 0xF0, 0x0F };
        int[] alternatingA = { 0x84, 0x42, 0x12, 0x81, 0xA0, 0x0A };
        int[] alternatingB = { 0x48, 0x24, 0x21, 0x18, 0x50, 0x05 };
        for (int face = 0; face < 6; face++)
        {
            int faceMask = i & masks[face];
            if (faceMask == alternatingA[face] || faceMask == alternatingB[face])
                count++;
        }
        return count;
    }

    private static int FaceTests(float[] v, int i, int[] face)
    {
        Array.Clear(face, 0, face.Length);
        if ((i & 0x80) != 0)
        {
            face[0] = (i & 0xCC) == 0x84 ? (v[0] * v[5] < v[1] * v[4] ? -1 : 1) : 0;
            face[3] = (i & 0x99) == 0x81 ? (v[0] * v[7] < v[3] * v[4] ? -1 : 1) : 0;
            face[4] = (i & 0xF0) == 0xA0 ? (v[0] * v[2] < v[1] * v[3] ? -1 : 1) : 0;
        }
        else
        {
            face[0] = (i & 0xCC) == 0x48 ? (v[0] * v[5] < v[1] * v[4] ? 1 : -1) : 0;
            face[3] = (i & 0x99) == 0x18 ? (v[0] * v[7] < v[3] * v[4] ? 1 : -1) : 0;
            face[4] = (i & 0xF0) == 0x50 ? (v[0] * v[2] < v[1] * v[3] ? 1 : -1) : 0;
        }
        if ((i & 0x02) != 0)
        {
            face[1] = (i & 0x66) == 0x42 ? (v[1] * v[6] < v[2] * v[5] ? -1 : 1) : 0;
            face[2] = (i & 0x33) == 0x12 ? (v[3] * v[6] < v[2] * v[7] ? -1 : 1) : 0;
            face[5] = (i & 0x0F) == 0x0A ? (v[4] * v[6] < v[5] * v[7] ? -1 : 1) : 0;
        }
        else
        {
            face[1] = (i & 0x66) == 0x24 ? (v[1] * v[6] < v[2] * v[5] ? 1 : -1) : 0;
            face[2] = (i & 0x33) == 0x21 ? (v[3] * v[6] < v[2] * v[7] ? 1 : -1) : 0;
            face[5] = (i & 0x0F) == 0x05 ? (v[4] * v[6] < v[5] * v[7] ? 1 : -1) : 0;
        }
        return face[0] + face[1] + face[2] + face[3] + face[4] + face[5];
    }

    private static int FaceTest1(float[] v, int face)
    {
        switch (face)
        {
            case 0: return v[0] * v[5] < v[1] * v[4] ? 0x48 : 0x84;
            case 1: return v[1] * v[6] < v[2] * v[5] ? 0x24 : 0x42;
            case 2: return v[3] * v[6] < v[2] * v[7] ? 0x21 : 0x12;
            case 3: return v[0] * v[7] < v[3] * v[4] ? 0x18 : 0x81;
            case 4: return v[0] * v[2] < v[1] * v[3] ? 0x50 : 0xA0;
            default: return v[4] * v[6] < v[5] * v[7] ? 0x05 : 0x0A;
        }
    }

    private static int InteriorTest(float[] v, int diagonal, int flag13, ref int testCount)
    {
        testCount++;
        float at = v[4] - v[0];
        float bt = v[5] - v[1];
        float ct = v[6] - v[2];
        float dt = v[7] - v[3];
        float t = at * ct - bt * dt;
        if (t < 0f)
        {
            if ((diagonal & 1) != 0) return 0;
        }
        else if ((diagonal & 1) == 0 || t == 0f)
        {
            return 0;
        }
        t = 0.5f * (v[3] * bt + v[1] * dt - v[2] * at - v[0] * ct) / t;
        if (!(t > 0f && t < 1f))
            return 0;
        at = v[0] + at * t;
        bt = v[1] + bt * t;
        ct = (v[2] + ct * t) * at;
        dt = (v[3] + dt * t) * bt;
        if ((diagonal & 1) != 0)
        {
            if (ct < dt && dt >= 0f)
                return ((bt < 0f) == (v[diagonal] < 0f) ? 1 : 0) + flag13;
        }
        else if (ct > dt && ct >= 0f)
        {
            return ((at < 0f) == (v[diagonal] < 0f) ? 1 : 0) + flag13;
        }
        return 0;
    }

    private static void Activate(int[] parent, int node)
    {
        if (parent[node] < 0) parent[node] = node;
    }

    private static int Find(int[] parent, int node)
    {
        if (node < 0 || node >= parent.Length || parent[node] < 0) return -1;
        int root = node;
        while (parent[root] != root) root = parent[root];
        while (parent[node] != node)
        {
            int next = parent[node];
            parent[node] = root;
            node = next;
        }
        return root;
    }

    private static void Union(int[] parent, int a, int b)
    {
        int rootA = Find(parent, a);
        int rootB = Find(parent, b);
        if (rootA >= 0 && rootB >= 0 && rootA != rootB)
            parent[rootB] = rootA;
    }
}

