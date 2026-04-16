Refactoring ``layer2c``
=======================

The current implementation of :mod:`keras2c.layer2c` exposes a single
``Layers2C`` class that contains more than five hundred lines of methods for
serialising each layer type to C code.  This monolithic class is difficult to
maintain and test because new behaviour must be added via additional private
methods and the internal state is managed through a single ``self.layers``
string.

A cleaner approach would be to move to a functional design where each layer is
handled by a standalone function.  These functions could accept a typed context
object describing the model and return the generated C code.  A simple dispatch
table can then be used in place of the current ``getattr`` calls.

Using :class:`pydantic.BaseModel` for the context allows the parameters passed to
layer writer functions to be validated and documented.  It also enables static
type checking across the code base.  Below is a rough outline of what this might
look like:

.. code-block:: python

    from pydantic import BaseModel
    from typing import List, Callable, Dict

    class LayerContext(BaseModel):
        name: str
        inputs: List[str]
        outputs: List[str]
        is_model_input: bool = False
        is_model_output: bool = False

    LayerWriter = Callable[[LayerContext], str]

    def write_dense(ctx: LayerContext) -> str:
        return f"k2c_dense({ctx.outputs[0]}, {ctx.inputs[0]}, {ctx.name}_kernel);" 

    LAYER_DISPATCH: Dict[str, LayerWriter] = {
        "Dense": write_dense,
        # additional layer functions
    }

    def write_layers(model) -> str:
        code_segments: List[str] = []
        for layer in model.layers:
            ctx = LayerContext(
                name=layer.name,
                inputs=[inp.name for inp in layer.input] if hasattr(layer, 'input') else [],
                outputs=[out.name for out in layer.output] if hasattr(layer, 'output') else [],
            )
            writer = LAYER_DISPATCH[layer.__class__.__name__]
            code_segments.append(writer(ctx))
        return "\n".join(code_segments)

The ``LayerContext`` model can be expanded to include extra attributes required
by individual layer implementations (activation functions, strides, etc.).  By
defining these fields explicitly the intent of each function becomes clearer and
unit tests can be written directly for them.

Moving ``layer2c`` to this functional style would isolate the logic for each
layer and drastically reduce the size of any single file.  The use of Pydantic
models gives strong typing and validation while remaining lightweight.
