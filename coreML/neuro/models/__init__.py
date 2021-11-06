from . import ResNet


MODEL_MAP = {
    "binary18" : ResNet.binary18,
    "binary18" : ResNet.binary34,
    "binary18" : ResNet.binary50,
    "binary18" : ResNet.binary50_32x4d,
    "binary18" : ResNet.binary101,
    "binary18" : ResNet.binary152
}