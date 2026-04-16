from typing import List, Union, TYPE_CHECKING
from pydantic import BaseModel
from .backend import keras

if TYPE_CHECKING:
    from .backend import keras as _keras_typing  # noqa: F401

# Check pydantic version for compatibility
try:
    import pydantic
    # Check version - v2 is 2.0.0+
    try:
        version = pydantic.__version__
        major_version = int(version.split('.')[0])
        PYDANTIC_V2 = major_version >= 2
    except (AttributeError, ValueError, IndexError):
        # Fallback: assume v1 if we can't determine
        PYDANTIC_V2 = False
except ImportError:
    PYDANTIC_V2 = False


class LayerIO(BaseModel):
    """Input/output details for a layer."""

    name: str
    pointer: str
    inputs: Union[str, List[str]]
    outputs: Union[str, List[str]]
    is_model_input: bool = False
    is_model_output: bool = False


# Create config based on pydantic version
if PYDANTIC_V2:
    # pydantic v2
    try:
        from pydantic import ConfigDict
        _config_dict = ConfigDict(arbitrary_types_allowed=True)
    except ImportError:
        _config_dict = None
else:
    _config_dict = None


class Keras2CConfig(BaseModel):
    """Configuration for :func:`keras2c_main.k2c`."""

    model: Union["keras.Model", str]
    function_name: str
    malloc: bool = False
    num_tests: int = 10
    verbose: bool = True

    if _config_dict is not None:
        # pydantic v2
        model_config = _config_dict
    else:
        # pydantic v1
        class Config:
            arbitrary_types_allowed = True


# Handle pydantic v1 vs v2 forward refs compatibility
if not PYDANTIC_V2:
    # pydantic v1 API
    try:
        Keras2CConfig.update_forward_refs(keras=keras)
    except TypeError:
        # Fallback if it fails
        pass
else:
    # pydantic v2 - forward refs are usually resolved automatically
    # but we can try model_rebuild if needed
    try:
        Keras2CConfig.model_rebuild()
    except (AttributeError, TypeError):
        # Forward refs should be resolved automatically in v2
        pass
