import cv2
import numpy as np

img = cv2.imread("dataset/training/flow_noc/000005_10.png", cv2.IMREAD_ANYCOLOR)

paths = [
    "dataset/training/flow_noc/000000_10.png",
    "dataset/training/flow_noc/000001_10.png",
    "dataset/training/flow_noc/000002_10.png",
    "dataset/training/flow_noc/000003_10.png",
    "dataset/training/flow_noc/000004_10.png",
    "dataset/training/flow_noc/000005_10.png",
    "dataset/training/flow_noc/000006_10.png",
    "dataset/training/flow_noc/000007_10.png",
    "dataset/training/flow_noc/000008_10.png",
    "dataset/training/flow_noc/000009_10.png",
    "dataset/training/flow_noc/000010_10.png",
    "dataset/training/flow_noc/000011_10.png",
    "dataset/training/flow_noc/000012_10.png",
    "dataset/training/flow_noc/000013_10.png",
    "dataset/training/flow_noc/000014_10.png",
    "dataset/training/flow_noc/000015_10.png",
    "dataset/training/flow_noc/000016_10.png",
    "dataset/training/flow_noc/000017_10.png",
    "dataset/training/flow_noc/000018_10.png",
    "dataset/training/flow_noc/000019_10.png",
    "dataset/training/flow_noc/000020_10.png",
    "dataset/training/flow_noc/000021_10.png",
    "dataset/training/flow_noc/000022_10.png",
    "dataset/training/flow_noc/000023_10.png",
    "dataset/training/flow_noc/000024_10.png",
    "dataset/training/flow_noc/000025_10.png",
    "dataset/training/flow_noc/000026_10.png",
    "dataset/training/flow_noc/000027_10.png",
    "dataset/training/flow_noc/000028_10.png",
    "dataset/training/flow_noc/000029_10.png",
    "dataset/training/flow_noc/000030_10.png",
    "dataset/training/flow_noc/000031_10.png",
    "dataset/training/flow_noc/000032_10.png",
    "dataset/training/flow_noc/000033_10.png",
    "dataset/training/flow_noc/000034_10.png",
    "dataset/training/flow_noc/000035_10.png",
    "dataset/training/flow_noc/000036_10.png",
    "dataset/training/flow_noc/000037_10.png",
    "dataset/training/flow_noc/000038_10.png",
    "dataset/training/flow_noc/000039_10.png",
    "dataset/training/flow_noc/000040_10.png",
    "dataset/training/flow_noc/000041_10.png",
    "dataset/training/flow_noc/000042_10.png",
    "dataset/training/flow_noc/000043_10.png",
    "dataset/training/flow_noc/000044_10.png",
    "dataset/training/flow_noc/000045_10.png",
    "dataset/training/flow_noc/000046_10.png",
    "dataset/training/flow_noc/000047_10.png",
    "dataset/training/flow_noc/000048_10.png",
    "dataset/training/flow_noc/000049_10.png",
    "dataset/training/flow_noc/000050_10.png",
    "dataset/training/flow_noc/000051_10.png",
    "dataset/training/flow_noc/000052_10.png",
    "dataset/training/flow_noc/000053_10.png",
    "dataset/training/flow_noc/000054_10.png",
    "dataset/training/flow_noc/000055_10.png",
    "dataset/training/flow_noc/000056_10.png",
    "dataset/training/flow_noc/000057_10.png",
    "dataset/training/flow_noc/000058_10.png",
    "dataset/training/flow_noc/000059_10.png",
    "dataset/training/flow_noc/000060_10.png",
    "dataset/training/flow_noc/000061_10.png",
    "dataset/training/flow_noc/000062_10.png",
    "dataset/training/flow_noc/000063_10.png",
    "dataset/training/flow_noc/000064_10.png",
    "dataset/training/flow_noc/000065_10.png",
    "dataset/training/flow_noc/000066_10.png",
    "dataset/training/flow_noc/000067_10.png",
    "dataset/training/flow_noc/000068_10.png",
    "dataset/training/flow_noc/000069_10.png",
    "dataset/training/flow_noc/000070_10.png",
    "dataset/training/flow_noc/000071_10.png",
    "dataset/training/flow_noc/000072_10.png",
    "dataset/training/flow_noc/000073_10.png",
    "dataset/training/flow_noc/000074_10.png",
    "dataset/training/flow_noc/000075_10.png",
    "dataset/training/flow_noc/000076_10.png",
    "dataset/training/flow_noc/000077_10.png",
    "dataset/training/flow_noc/000078_10.png",
    "dataset/training/flow_noc/000079_10.png",
    "dataset/training/flow_noc/000080_10.png",
    "dataset/training/flow_noc/000081_10.png",
    "dataset/training/flow_noc/000082_10.png",
    "dataset/training/flow_noc/000083_10.png",
    "dataset/training/flow_noc/000084_10.png",
    "dataset/training/flow_noc/000085_10.png",
    "dataset/training/flow_noc/000086_10.png",
    "dataset/training/flow_noc/000087_10.png",
    "dataset/training/flow_noc/000088_10.png",
    "dataset/training/flow_noc/000089_10.png",
    "dataset/training/flow_noc/000090_10.png",
    "dataset/training/flow_noc/000091_10.png",
    "dataset/training/flow_noc/000092_10.png",
    "dataset/training/flow_noc/000093_10.png",
    "dataset/training/flow_noc/000094_10.png",
    "dataset/training/flow_noc/000095_10.png",
    "dataset/training/flow_noc/000096_10.png",
    "dataset/training/flow_noc/000097_10.png",
    "dataset/training/flow_noc/000098_10.png",
    "dataset/training/flow_noc/000099_10.png",
    "dataset/training/flow_noc/000100_10.png",
    "dataset/training/flow_noc/000101_10.png",
    "dataset/training/flow_noc/000102_10.png",
    "dataset/training/flow_noc/000103_10.png",
    "dataset/training/flow_noc/000104_10.png",
    "dataset/training/flow_noc/000105_10.png",
    "dataset/training/flow_noc/000106_10.png",
    "dataset/training/flow_noc/000107_10.png",
    "dataset/training/flow_noc/000108_10.png",
    "dataset/training/flow_noc/000109_10.png",
    "dataset/training/flow_noc/000110_10.png",
    "dataset/training/flow_noc/000111_10.png",
    "dataset/training/flow_noc/000112_10.png",
    "dataset/training/flow_noc/000113_10.png",
    "dataset/training/flow_noc/000114_10.png",
    "dataset/training/flow_noc/000115_10.png",
    "dataset/training/flow_noc/000116_10.png",
    "dataset/training/flow_noc/000117_10.png",
    "dataset/training/flow_noc/000118_10.png",
    "dataset/training/flow_noc/000119_10.png",
    "dataset/training/flow_noc/000120_10.png",
    "dataset/training/flow_noc/000121_10.png",
    "dataset/training/flow_noc/000122_10.png",
    "dataset/training/flow_noc/000123_10.png",
    "dataset/training/flow_noc/000124_10.png",
    "dataset/training/flow_noc/000125_10.png",
    "dataset/training/flow_noc/000126_10.png",
    "dataset/training/flow_noc/000127_10.png",
    "dataset/training/flow_noc/000128_10.png",
    "dataset/training/flow_noc/000129_10.png",
    "dataset/training/flow_noc/000130_10.png",
    "dataset/training/flow_noc/000131_10.png",
    "dataset/training/flow_noc/000132_10.png",
    "dataset/training/flow_noc/000133_10.png",
    "dataset/training/flow_noc/000134_10.png",
    "dataset/training/flow_noc/000135_10.png",
    "dataset/training/flow_noc/000136_10.png",
    "dataset/training/flow_noc/000137_10.png",
    "dataset/training/flow_noc/000138_10.png",
    "dataset/training/flow_noc/000139_10.png",
    "dataset/training/flow_noc/000140_10.png",
    "dataset/training/flow_noc/000141_10.png",
    "dataset/training/flow_noc/000142_10.png",
    "dataset/training/flow_noc/000143_10.png",
    "dataset/training/flow_noc/000144_10.png",
    "dataset/training/flow_noc/000145_10.png",
    "dataset/training/flow_noc/000146_10.png",
    "dataset/training/flow_noc/000147_10.png",
    "dataset/training/flow_noc/000148_10.png",
    "dataset/training/flow_noc/000149_10.png",
    "dataset/training/flow_noc/000150_10.png",
    "dataset/training/flow_noc/000151_10.png",
    "dataset/training/flow_noc/000152_10.png",
    "dataset/training/flow_noc/000153_10.png",
    "dataset/training/flow_noc/000154_10.png",
    "dataset/training/flow_noc/000155_10.png",
    "dataset/training/flow_noc/000156_10.png",
    "dataset/training/flow_noc/000157_10.png",
    "dataset/training/flow_noc/000158_10.png",
    "dataset/training/flow_noc/000159_10.png",
    "dataset/training/flow_noc/000160_10.png",
    "dataset/training/flow_noc/000161_10.png",
    "dataset/training/flow_noc/000162_10.png",
    "dataset/training/flow_noc/000163_10.png",
    "dataset/training/flow_noc/000164_10.png",
    "dataset/training/flow_noc/000165_10.png",
    "dataset/training/flow_noc/000166_10.png",
    "dataset/training/flow_noc/000167_10.png",
    "dataset/training/flow_noc/000168_10.png",
    "dataset/training/flow_noc/000169_10.png",
    "dataset/training/flow_noc/000170_10.png",
    "dataset/training/flow_noc/000171_10.png",
    "dataset/training/flow_noc/000172_10.png",
    "dataset/training/flow_noc/000173_10.png",
    "dataset/training/flow_noc/000174_10.png",
    "dataset/training/flow_noc/000175_10.png",
    "dataset/training/flow_noc/000176_10.png",
    "dataset/training/flow_noc/000177_10.png",
    "dataset/training/flow_noc/000178_10.png",
    "dataset/training/flow_noc/000179_10.png",
    "dataset/training/flow_noc/000180_10.png",
    "dataset/training/flow_noc/000181_10.png",
    "dataset/training/flow_noc/000182_10.png",
    "dataset/training/flow_noc/000183_10.png",
    "dataset/training/flow_noc/000184_10.png",
    "dataset/training/flow_noc/000185_10.png",
    "dataset/training/flow_noc/000186_10.png",
    "dataset/training/flow_noc/000187_10.png",
    "dataset/training/flow_noc/000188_10.png",
    "dataset/training/flow_noc/000189_10.png",
    "dataset/training/flow_noc/000190_10.png",
    "dataset/training/flow_noc/000191_10.png",
    "dataset/training/flow_noc/000192_10.png",
    "dataset/training/flow_noc/000193_10.png",
    "dataset/training/flow_noc/000194_10.png",
    "dataset/training/flow_noc/000195_10.png",
    "dataset/training/flow_noc/000196_10.png",
    "dataset/training/flow_noc/000197_10.png",
    "dataset/training/flow_noc/000198_10.png",
    "dataset/training/flow_noc/000199_10.png"
]


for path in paths:
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # print(img.dtype)
    print("max", np.max(img[:, :, 1]), " min ", np.min(img[:, :, 1]))
    # print(np.min(img[:, :, 1]))
