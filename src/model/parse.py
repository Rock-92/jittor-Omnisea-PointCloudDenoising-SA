from .spec import ModelSpec
from .vm import VelocityModule
from .edgeconv_baseline import EdgeConvBaselineModule
from .shape_ssl import MaskedShapePretrainModule, ShapeContextVelocityModule

def get_model(model_config, **kwargs) -> ModelSpec:
    MAP = {
        'VelocityModule': VelocityModule,
        'EdgeConvBaselineModule': EdgeConvBaselineModule,
        'EdgeConv_Baseline': EdgeConvBaselineModule,
        'MaskedShapePretrainModule': MaskedShapePretrainModule,
        'ShapeContextVelocityModule': ShapeContextVelocityModule,
    }
    __target__ = model_config['__target__']
    del model_config['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](model_config=model_config, **kwargs)
