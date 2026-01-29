from typing import Any, List, Union, Optional

def value_only(value: Any = None, **kwargs: Any) -> Any:
    """Return the input value directly (for static inputs)."""
    return value

def first_float(value: Union[List[Any], Any], **kwargs: Any) -> Optional[float]:
    """Return the first element as a float."""
    if isinstance(value, list):
        if not value:
            return None
        val = value[0]
    else:
        val = value
    
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def ms_to_s(value: Any, **kwargs: Any) -> Optional[float]:
    """Convert milliseconds to seconds."""
    val = first_float(value)
    if val is not None:
        return val / 1000.0
    return None
