"""Testing"""

import os
try:
    from . import maths
except ImportError:
    import maths

decimals_module = [0]
try:
    from .maths import decimals
except ImportError:
    try:
        from maths import decimals
    except ImportError:
        import maths.decimals as decimals

import fractions

DEBUG = True
QUICK_PRINT = True

if QUICK_PRINT:
    logs = []
    log = lambda*args,sep=' ',end='\n':logs.append(''.join([sep.join([str(i) for i in args]),end]))
elif DEBUG:
    log = lambda*a,**k:print(*a,**k)
else:
    log = lambda*a,**k:None

cwd = ['.']

def _init():
    try:
        cwd[0] = os.getcwd()
    except Exception:
        pass

def _decimals():
    log('Running tests for decimals.py')
    log()

    log('SETUP')
    decimal_dict = {}
    Decimal = decimals.Decimal
    Number = decimals.Number
    log()

    # Built from a float (precise or unprecise)
    log('float, precise')
    decimal_dict['float_precise'] = Decimal(0.1, precise=True)
    log('float, unprecise')
    decimal_dict['float_unprecise'] = Decimal(0.2, precise=False)
    log()

    # Built from a fractions.Fraction()
    log('fraction')
    decimal_dict['Fraction'] = Decimal(fractions.Fraction(3,10))
    log()

    # Built from a string representation of a float
    log('string')
    decimal_dict['string'] = Decimal('0.4')
    log()

    # Built from a iterable of length 2 (a fraction)
    log('iterable, normal')
    decimal_dict['iterable'] = Decimal((6,10))
    log('iterable, unusual')
    arg = (6,10).__iter__
    if hasattr(arg,'__call__'):
        arg = arg.__call__()
    decimal_dict['iterableWEIRD'] = Decimal(arg)
    log()

    # Built from 2 arguments (similar to iterable)
    log('2 arguments')
    decimal_dict['2args'] = Decimal(7,10)
    log()

    # Built from integer
    log('integer')
    decimal_dict['int'] = Decimal(8)
    log()

    # Display all
    for k,v in decimal_dict.items():
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

def _main():
    _decimals()

def _cleanup():
    os.chdir(cwd[0])

if __name__ == "__main__":
    _init()
    try:
        _main()
    finally:
        _cleanup()

if QUICK_PRINT:
    print(''.join(logs),end='')
