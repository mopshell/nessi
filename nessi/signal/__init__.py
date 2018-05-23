from __future__ import (absolute_import,
                        division,
                        print_function,
                        unicode_literals)

from .windowing import time_window
from .windowing import space_window

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
