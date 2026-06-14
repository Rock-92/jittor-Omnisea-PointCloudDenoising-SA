from .spec import DummySystem, DummyWriter
from .vm import VMSystem, VMSSLSystem, VMWriter
from .shape_ssl import ShapePretrainSystem, ShapeContextVMSystem

def get_system(**kwargs) -> DummySystem:
    MAP = {
        'dummy': DummySystem,
        'vm': VMSystem,
        'vm_ssl': VMSSLSystem,
        'shape_pretrain': ShapePretrainSystem,
        'shape_context_vm': ShapeContextVMSystem,
    }
    __target__ = kwargs['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    del kwargs['__target__']
    return MAP[__target__](**kwargs)

def get_writer(**kwargs) -> DummyWriter:
    MAP = {
        'dummy': DummyWriter,
        'vm': VMWriter,
    }
    __target__ = kwargs['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    del kwargs['__target__']
    return MAP[__target__](**kwargs)
