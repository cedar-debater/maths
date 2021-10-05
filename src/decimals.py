"""Precision base-10 decimal numbers"""

from fractions import Fraction as _Frac
import inspect
from operator import add, sub

DEBUG = True
QUICK_PRINT = True

if QUICK_PRINT:
    logs = []
    log = lambda*args,sep=' ',end='\n':logs.append(''.join([sep.join([str(i) for i in args]),end]))
elif DEBUG:
    log = lambda*a,**k:print(*a,**k)
else:
    log = lambda*a,**k:None

class Number(object):
    """Precise base 10 whole numbers

Can be built from a normal number (int)
Can be built from a string representatoin of a int
Not allowed to be built from a float (due to issues)
Why not? Run `int(float(1e+100))` to see why floats are not allowed.
10000000000000000159028911097599180468360808563945281389781327557747838772170381060813469985856815104
"""
    def __init__(self, arg):
        if isinstance(arg, float):
            raise TypeError('argument passed to Number() cannot be float\n\nsee docstring','decimals.Number.__init__()')
        elif isinstance(arg, int):
            self._string = str(arg)
        elif isinstance(arg, str):
            for char in arg:
                if char not in '123456789.0':
                    raise ValueError('Invalid character passed to Number()','decimals.Number.__init__()')
                if char == '.':
                    raise ValueError('No full stops allowed in strings passed to Number() due to float inaccuracies','decimals.Number.__init__()')
            arg = list(arg)
            while (arg[0] == '0'):
                arg = arg[1:]
            self._string = ''.join(arg)
        else:
            raise TypeError('argument passed to Number() should be int or its repr()')
    def _op(self, other, is_plus = True, reverse = False):
        if isinstance(other, int):
            other = Number(other)
        if not isinstance(other, type(self)):
            raise TypeError('Numbers can only be added/subtracted with other Number instances or integers','decimals.Number._op()')
        if reverse:
            self, other = other, self
        op_func = [sub, add][int(is_plus)]
        carry = 0
        new = []
        for a,b in zip(*[i._string[::-1] for i in [self, other]]):
            carry, num = divmod(op_func(int(a) + carry,int(b)), 10)
            new.append(num)
        new.append(carry)
        return type(self)(''.join([str(i) for i in new[-2::-1]]))
        
    def __add__(self, other):
        return self._op(other)

    def __radd__(self, other):
        return self._op(other, reverse = True)

    def __sub__(self, other):
        return self._op(other, is_plus = False)

    def __rsub__(self, other):
        return self._op(other, is_plus = False, reverse = True)

    def __mul__(self, other):
        if isinstance(other, int):
            other = Number(other)
        if not isinstance(other, type(self)):
            raise TypeError('Numbers can only be multiplied with other Number instances or integers','decimals.Number.__mul__()')
        max_length = max(len(self._string),len(other._string))
    def __repr__(self):
        return f"{{{self._string}}}"

class Decimal(object):
    """Precise base 10 decimal numbers

Can be built from a float (precise or unprecise)
Can be built from a fractions.Fraction()
Can be built from a string representation of a float/integer
Can be built from an iterable of length 2 (a fraction)
Can be built from 2 arguments (similar to passing in iterable)
Can be built from an integer

Examples:

Decimal(3.3434) # unprecise (string)
Decimal(3.3434, precise=False) # specified unprecise (string)

Decimal(3.3434, precise=True) # specified "precise"
# This uses Fraction.from_float() - so be careful

# Normal has no problems
# Decimal(3,10) == Decimal(0.3)

# But built-in floats are made with bits and not strings...
# Decimal(3,10) != Decimal(0.3, precise=True)

# Decimal(0.3, precise=True) == Decimal(5404319552844595, 18014398509481984)
"""
    def __init__(self, *args, precise=False):
        if len(args) == 2:
            arg = args
        elif len(args) == 1:
            arg = args[0]
        else:
            raise ValueError("Decimal() takes 1 or 2 arguments",'decimals.Decimal.__init__()')
        if isinstance(arg, str):
            log('STRING',arg)
            self._string = arg
            self._repeat = ''
        else:
            built = []
            try:
                new = arg.__iter__()
                while True:
                    try:
                        built.append(Decimal(next(new)))
                    except StopIteration:
                        break
                if len(built) != 2:
                    raise ValueError(f"Iterables should be of length 2 and not {len(built)}","decimals.Decimal.__init__()")
                log('LIST',built)
                self._string = "*unfinished Decimal*"
                #self._string = (lambda a,b:(a/b))(*built)
            except (AttributeError, TypeError):
                if not isinstance(arg, (float, int, _Frac)):
                    raise TypeError(
                        'Bad argument type passed to Decimal()',
                        'Decimal.__init__()'
                    )
                elif isinstance(arg, float):
                    if precise:
                        log('pass control (precise => 2 args)')
                        type(self).__init__(self, _Frac.from_float(arg).as_integer_ratio())
                    else:
                        log('pass control (unprecise => string)')
                        type(self).__init__(self, repr(arg))
                elif isinstance(arg, int):
                    log('pass control (int => string)')
                    type(self).__init__(self, repr(arg))
                elif isinstance(arg, _Frac):
                    type(self).__init__(self, arg.as_integer_ratio())
    def __repr__(self):
        return f"<{self._string}>"

def _main():
    log('Running tests')
    log()

    log('SETUP')
    decimals = {}
    log()

    # Built from a float (precise or unprecise)
    log('float, precise')
    decimals['float_precise'] = Decimal(0.1, precise=True)
    log('float, unprecise')
    decimals['float_unprecise'] = Decimal(0.2, precise=False)
    log()

    # Built from a fractions.Fraction()
    log('fraction')
    decimals['Fraction'] = Decimal(_Frac(3,10))
    log()

    # Built from a string representation of a float
    log('string')
    decimals['string'] = Decimal('0.4')
    log()

    # Built from a iterable of length 2 (a fraction)
    log('iterable, normal')
    decimals['iterable'] = Decimal((6,10))
    log('iterable, unusual')
    arg = (6,10).__iter__
    if hasattr(arg,'__call__'):
        arg = arg.__call__()
    decimals['iterableWEIRD'] = Decimal(arg)
    log()

    # Built from 2 arguments (similar to iterable)
    log('2 arguments')
    decimals['2args'] = Decimal(7,10)
    log()

    # Built from integer
    log('integer')
    decimals['int'] = Decimal(8)
    log()

    # Display all
    for k,v in decimals.items():
        log(k,v,sep=': ')
    log()

    # Number test 3+4 and 4-3
    three = Number(3)
    log('three',three)
    four = Number('4')
    log('four',four)
    seven = three + four
    log('seven',seven)
    one = four - three
    log('one',one)
    log()

    # Number test 36+19 and 36-19
    left = Number('36')
    log('left',left)
    right = Number(19)
    log('right',right)
    add_ = left + right
    log('add', add_)
    sub_ = left - right
    log('sub', sub_)
    log()

    # Number test - Number(3) + 2
    three = Number(3)
    log('five',three+2)
    log('one',three-2)
    log()

if __name__ == "__main__":
    _main()

if QUICK_PRINT:
    print(''.join(logs),end='')
