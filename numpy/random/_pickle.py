from typing import Optional, Type

from .mtrand import RandomState
from ._philox import Philox
from ._pcg64 import PCG64, PCG64DXSM
from ._sfc64 import SFC64
from .bit_generator import BitGenerator
from ._generator import Generator
from ._mt19937 import MT19937

BitGenerators: dict[str, Type[BitGenerator]]

BitGenerators = {'MT19937': MT19937,
                 'PCG64': PCG64,
                 'PCG64DXSM': PCG64DXSM,
                 'Philox': Philox,
                 'SFC64': SFC64,
                 }


def __generator_ctor(bit_generator_name:str='MT19937') -> Generator:
    """
    Pickling helper function that returns a Generator object

    Parameters
    ----------
    bit_generator_name : str
        String containing the core BitGenerator. Can also be the
        full path (module.class) of a third-party bit generator.

    Returns
    -------
    rg : Generator
        Generator using the named core BitGenerator
    """
    bit_generator: Optional[Type[BitGenerator]] = None
    if bit_generator_name in BitGenerators:
        bit_generator = BitGenerators[bit_generator_name]
    elif "." in bit_generator_name:
        parts = bit_generator_name.split(".")
        mod_name = ".".join(parts[:-1])
        bit_generator_name = parts[-1]

        import importlib
        try:
            mod = importlib.import_module(mod_name)
            bit_generator = getattr(mod, bit_generator_name, None)
        except ImportError:
            pass

    if bit_generator is None:
        raise ValueError(
            f"{bit_generator_name}  is not a known NumPy BitGenerator class "
            f"and cannot be looked up using it's fully qualified name."
        )
    assert bit_generator is not None

    return Generator(bit_generator())


def __bit_generator_ctor(bit_generator_name='MT19937'):
    """
    Pickling helper function that returns a bit generator object

    Parameters
    ----------
    bit_generator_name : str
        String containing the name of the BitGenerator

    Returns
    -------
    bit_generator : BitGenerator
        BitGenerator instance
    """
    if bit_generator_name in BitGenerators:
        bit_generator = BitGenerators[bit_generator_name]
    else:
        raise ValueError(str(bit_generator_name) + ' is not a known '
                                                   'BitGenerator module.')

    return bit_generator()


def __randomstate_ctor(bit_generator_name='MT19937'):
    """
    Pickling helper function that returns a legacy RandomState-like object

    Parameters
    ----------
    bit_generator_name : str
        String containing the core BitGenerator

    Returns
    -------
    rs : RandomState
        Legacy RandomState using the named core BitGenerator
    """
    if bit_generator_name in BitGenerators:
        bit_generator = BitGenerators[bit_generator_name]
    else:
        raise ValueError(str(bit_generator_name) + ' is not a known '
                                                   'BitGenerator module.')

    return RandomState(bit_generator())
