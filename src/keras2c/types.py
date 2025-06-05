from typing import List, Union, TYPE_CHECKING
from pydantic import BaseModel
from .backend import keras

if TYPE_CHECKING:
    from .backend import keras as _keras_typing  # noqa: F401


class LayerIO(BaseModel):
    """Input/output details for a layer."""

    name: str
    pointer: str
    inputs: Union[str, List[str]]
    outputs: Union[str, List[str]]
    is_model_input: bool = False
    is_model_output: bool = False


class Keras2CConfig(BaseModel):
    """Configuration for :func:`keras2c_main.k2c`."""

    model: Union["keras.Model", str]
    function_name: str
    malloc: bool = False
    num_tests: int = 10
    verbose: bool = True

    class Config:
        arbitrary_types_allowed = True


Keras2CConfig.update_forward_refs(keras=keras)
