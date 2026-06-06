from .spec import ModelSpec
from .vm import VelocityModule
from .pointnet2 import PointNet2VelocityModule
from .edgeconv_baseline import EdgeConvBaselineModule

def get_model(model_config, **kwargs) -> ModelSpec:
    MAP = {
        'VelocityModule': VelocityModule,
        'PointNet2VelocityModule': PointNet2VelocityModule,
        'EdgeConvBaselineModule': EdgeConvBaselineModule,
        'EdgeConv_Baseline': EdgeConvBaselineModule,
    }
    __target__ = model_config['__target__']
    del model_config['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](model_config=model_config, **kwargs)
