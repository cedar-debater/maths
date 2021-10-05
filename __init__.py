"""Math extension module"""

import math as _math

import functools

import ast

from .decimals import Decimal

## ************************************

## Setup and Settings

__all__ = [
    'pi', 'Pi', 'PI',
    'e', 'E',
    'use_degrees',
    'use_radians',
    'toggle_degrees',
    'toggle_radians',
    'set_degrees',
    'set_radians',
    'Expression',
    'wrap',
    'Variable',
    'Add',
    'Subtract',
    'Multiply',
    'Divide',
    'Exponent',
    'sin', 'cos', 'tan', 'ln',
    'Function',
    'derivative',
    'd', 'dx', 'dn',
    'evaluate',
    'simplify',
    'VariableBond',
    'x', 'n'
]

pi = Pi = PI = getattr(_math,'pi',
    getattr(
        _math,'PI',getattr(
            _math,
            'Pi',
            3.1415926535897932384626433832795029
        )
    )
)

e = E = getattr(_math,'e',
    getattr(
        _math,
        'E',
        2.71828182845904523536
    )
)

_NUMBER_TYPE = (int, float, complex)

_settings = {'degrees': False}

class repr_call(object):
    def __init__(self, func):
        self._func = func
    def __call__(self, *a, **k):
        return self._func(*a, **k)
    def __repr__(self, *a, **k):
        self._func(*a, **k)
        return repr(self._func)

@repr_call
def use_degrees():
    _settings['degrees'] = True
@repr_call
def use_radians():
    _settings['degrees'] = False

@repr_call
def toggle_degrees():
    _settings['degrees'] = not _settings['degrees']
@repr_call
def toggle_radians():
    _settings['degrees'] = not _settings['degrees']

@repr_call
def set_degrees(boolean):
    _settings['degrees'] = boolean
@repr_call
def set_radians(boolean):
    _settings['degrees'] = boolean

def _angle(n):
    """Convert to radians if it is degrees"""
    if _settings['degrees']:
        n = n*pi/180
    return n

PLEASE_CONVERT = (
    "\n".join([
        "\n",
        "Please convert variables with either of the following:",
        "VariableBond()",
        "Variable.set(<value>) and Variable.clear()",
        "See documentation for both with help(VariableBond) or help(Variable)",
    ]),
    "evaluate()"
)

# Q: What happens if you do help(VariableBond) or help(Variable)
# A: The following. (basically)

##thing = <VariableBond or Variable>
##name = thing.__name__
##sys.stdout.write(f'Help on class {name} in module maths:\n\n')
##sys.stdout.write(pydoc.plaintext.document(thing, name))

# In pydoc... (line 1730)

##plaintext = _PlainTextDoc()

# In pydoc._PlainTextDoc:

# self.bold(text) now returns text. That's it.

# ...

# pydoc.getdoc(<VariableBond or Variable>)

# heres the code

##def getdoc(obj):
##    """Get the doc string or comments for an object."""
##    result = _getdoc(obj) or inspect.getcomments(obj)
##    return result and re.sub('^ *\n', '', result.rstrip()) or ''

# or sometimes

DIFF_TYPE = " ".join([
    "Part of expression",
    "is not a number or",
    "Expression - from:",
])

TYPE_EE = (DIFF_TYPE, "evaluate()")

TYPE__D = (DIFF_TYPE, "_derivative()")

TYPE_S = (DIFF_TYPE, "simplify()")

## ************************************

## Expression classes and subclasses

class ExpressionMeta(type):
    """Base metaclass for expression objects"""

class ExpressionBase(object, metaclass=ExpressionMeta):
    """Base object for expressions"""
    def __init__(self):
        self.vars = tuple([])
    def _get(self, name):
        return '='.join([name,repr(getattr(self, name))])
    def __repr__(self):
        n = [self._get(var) for var in self.vars]
        n = ', '.join(n)
        return f"{type(self).__name__}({n})"
    def __add__(self, other):
        return Add(self, other)
    def __radd__(other, self):
        return Add(self, other)
    def __sub__(self, other):
        return Subtract(self, other)
    def __rsub__(other, self):
        return Subtract(self, other)
    def __mul__(self, other):
        return Multiply(self, other)
    def __rmul__(other, self):
        return Multiply(self, other)
    def __truediv__(self, other):
        return Divide(self, other)
    __div__ = __truediv__
    def __rtruediv__(other, self):
        return Divide(self, other)
    __rdiv__ = __rtruediv__
    def __pow__(self, other):
        return Exponent(self, other)
    def __rpow__(other, self):
        return Exponent(self, other)
    def __xor__(self, other):
        return Exponent(self, other)
    def __rxor__(other, self):
        return Exponent(self, other)
    def __neg__(self):
        return Negative(self)
    def __pos__(self):
        return self
    def __matmul__(self, other):
        empty = getattr(
            __import__('_queue'),
            'Empty',
            getattr(
                __import__('queue'),
                'Empty',
                type(
                    'Empty',
                    tuple(Exception.mro()),
                    {}
                )
            )
        )
        class_ = [None]
        try:
            class new(UserWarning, empty):
                pass
            class_[0] = new
        except Exception:
            try:
                class new(empty):
                    pass
                class_[0] = new
            except Exception:
                class new(UserWarning):
                    pass
                class_[0] = new
        children = type
        raise children (#Why am I doing this?!?#?!?siht gniod I ma yhW#) nerdlihc esiar        
            ''.join(
                [chr(i) for i in (
                    69,
                    97,
                    115,
                    116,
                    101,
                    114,
                    32,
                    69,
                    103,
                    103,
                    33,
                    32,
                    87,
                    101,
                    108,
                    108,
                    32,
                    100,
                    111,
                    110,
                    101,
                    59,
                    32,
                    73,
                    32,
                    104,
                    111,
                    110,
                    101,
                    115,
                    116,
                    108,
                    121,
                    32,
                    100,
                    105,
                    100,
                    110,
                    39,
                    116,
                    32,
                    101,
                    120,
                    112,
                    101,
                    99,
                    116,
                    32,
                    121,
                    111,
                    117,
                    32,
                    116,
                    111,
                    32,
                    102,
                    105,
                    110,
                    100,
                    32,
                    116,
                    104,
                    105,
                    115,
                    46,
                    46,
                    46
                )]
            ),
            tuple(class_[0].mro()),
            {}
        )
    __rmatmul__ = __matmul__

class Expression(ExpressionBase):
    """Wrapper for all expressions"""
    def __init__(self, body):
        self.body = body
        self.vars = ('body',)
    def __repr__(self):
        return f'<< {self.body} >>'

wrap = Expression

def _get_name(obj, default=None):
    if hasattr(obj, '__name__'):
        return obj.__name__
    if hasattr(obj, '__qualname__'):
        return obj.__qualname__
    return default

def add_wrap(func, name=None):
    """Add a wrap around the output of a function."""
    @functools.wraps(func)
    def _internal(*args, **kwargs):
        return wrap(func(*args, **kwargs))
    if name is None:
        name = _get_name(func, default='')
        while name.startswith('_') and len(name) > 1:
            name = name[1:]
    _internal.__name__ = name
    _internal.__qualname__ = name
    return _internal

class Variable(ExpressionBase):
    """Class for all variables.
Variables can be bonded with a value within
the function evaluate().

To bond variables, use one of the following:

with VariableBond(3):
    pass # do stuff here where variables are 3
# do stuff here where variables are no longer 3

OR

Variable.set(3)
# do stuff here where variables are 3
Variable.clear()
# do stuff here where variables are no longer 3
"""
    _all = [False, None]
    def __init__(self, name):
        self.name = name
        self.vars = ('name',)
    def __repr__(self):
        return self.name
    @classmethod
    def set(cls, new):
        cls._all[0] = True
        cls._all[1] = new
    @classmethod
    def clear(cls):
        cls._all[0] = False
        cls._all[1] = None
    @classmethod
    def _get(cls):
        return cls._all[1]
    @classmethod
    def _resolved(cls):
        return cls._all[0]

class OperatorBase(ExpressionBase):
    """Base object for operators"""
    def __init__(self, *args, **kwargs):
        self._init(*args, **kwargs)
        self.vars = ('left', 'right')
    def __repr__(self):
        return ''.join([
            '(',
            repr(self.left),
            self.op,
            repr(self.right),
            ')'
        ])

class OperatorMeta(ExpressionMeta):
    """Metaclass for operators"""
    def _init(self, left, right):
        if (self.op == '/') and (
            right == 0 or (
                isinstance(right, Variable) and Variable._get() == 0
            )
        ):
            # DIVISION BY ZERO IS NOT ALLOWED.
            try:
                ArithmeticError = ZeroDivisionError
            except NameError:
                pass
            try:
                raise ArithmeticError("division by zero")
            except NameError:
                pass
            raise Exception("division by zero")
        self.left = left
        self.right = right
    def __init__(self, first, other = None):
        type.__init__(self, first)
    def __new__(self, operator, second = None):
        if second is None:
            second = operator
        retval = type.__new__(
            self,
            operator,
            tuple(OperatorBase.mro()),
            {
                'op': operator,
                'sec': second,
                '_init': self._init
            }
        )
        return retval

Add = OperatorMeta('+')
Subtract = OperatorMeta('-')
Multiply = OperatorMeta('*')
Divide = OperatorMeta('/')
Exponent = OperatorMeta('^','**')

def Negative(expr):
    return Subtract(0, expr)

## ************************************

## Mathematical Functions

class angle(object):
    def __init__(self, func, state=1):
        self._func = func
        self._state = state
    def __call__(self, arg):
        if not isinstance(arg, str):
            arg = _angle(arg)
        return self._state * self._func(arg)
    def __neg__(self):
        return type(self)(self._func, -self._state)
    def __repr__(self):
        return repr(self._func)

@angle
def _sin(k):
    """Evaluation of sin(x)"""
    return _math.sin(k)

@angle
def sin(k):
    """Expression of sin(x)"""
    if isinstance(k, str):
        return Function('sin',Variable(k))
    return Function('sin',k)

# **********

@angle
def _cos(k):
    """Evaluation of cos(x)"""
    return _math.cos(k)

@angle
def cos(k):
    """Expression of cos(x)"""
    if isinstance(k, str):
        return Function('cos',Variable(k))
    return Function('cos',k)

# **********

@angle
def _tan(k):
    """Evaluation of tan(x)"""
    return _math.tan(k)

@angle
def tan(k):
    """Expression of tan(x)"""
    if isinstance(k, str):
        return Function('tan',Variable(k))
    return Function('tan',k)

@angle
def _d_tan(k):
    """Derivative of tan(k)
Find the actual value"""
    return _cos(k) ** (-2)
    # return expr_eval(d/dx(Divide(sin(x), cos(x))))

# **********

def ln(k):
    """Expression of: ln(x)"""
    if isinstance(k, str):
        return Function('ln',Variable(k))
    return Function('ln',k)

def _ln(k):
    """Evaluation of: ln(x)"""
    return _math.log(k)

def _d_ln(k):
    """Derivative of: ln(x)
Basically `_d_ln = lambda k:(1/k)`
"""
    return 1/k

# **********

_f_sin = cos

def _f_cos(k):
    return Negative(sin(k))

def _f_tan(k):
    return Divide(1, Multiply(cos(k), cos(k)))

def _f_ln(k):
    return Divide(1, k)

## ************************************

## Function class

class Function(ExpressionBase):
    """A function"""
    func_dict = {
        'sin':  ( _sin , _cos   , _f_sin )
        ,'cos': ( _cos , -_sin  , _f_cos )
        ,'tan': ( _tan , _d_tan , _f_tan )
        ,'ln':  ( _ln  , _d_ln  , _f_ln  )
    }
    def __init__(self, func_name, arg):
        self.name = func_name
        self.arg = arg
        self.vars = ('name','arg')
    def __repr__(self):
        return f"{self.name}({self.arg!r})"

## ************************************

## Derivatives

def _derivative(*args):
    exp = args[-1]
    ist = isinstance
    if ist(exp, _NUMBER_TYPE):
        return 0
    if not ist(exp, ExpressionBase):
        raise TypeError(*TYPE__D)
    if ist(exp, Expression):
        exp = exp.body
    if ist(exp, Variable):
        return 1
    if ist(exp, (Add, Subtract)):
        # exp is Add or Subtract object
        return type(exp)(
            d(exp.left),
            d(exp.right)
        )
    if ist(exp, (Multiply, Divide)):
        # Product rule = (f' * g) + (f * g')

        # Quotient rule = (f' * g) - (f * g')
        # but all multiplied by 1 / g^2

        k = [exp.left, exp.right]
        a = [Multiply(
            d(e),
            k[1-i]
        ) for i,e in enumerate(k)]

        if ist(exp, Multiply):
            return Add(*a)
        return Divide(Subtract(*a),Multiply(k[1], k[1]))
    if ist(exp, Function):
        # Chain rule - return f'(g(x)) * g'(x)
        func_tuple = exp.func_dict[exp.name]
        left = func_tuple[2](exp.arg)
        right = d[exp.arg]
        return Multiply(left, right)
    if ist(exp, Exponent):
        # Exponent rule make: y = f^g
        # F = f' and G = g'
        # ln(y) = ln(f^g)
        # ln(y) = g*ln(f)
        # ln(y(x)) = g(x) * ln(f(x))

        # ln(k(x)) => K(x)/k(x)

        # d(x)/y(x) = d/dx [ g(x) * ln(f(x)) ]
        # d(x) = y(x) * (G(x) * ln(f(x)) + g(x) * F(x)/f(x))

        # A = G(x) * ln(f(x))
        # B = g(x) * F(x) / f(x)
        # R = A + B
        # return y(x) * R

        # f and g
        f = exp.left
        g = exp.right

        # A = G(x) * ln(f(x))
        a = Multiply(d(g), ln(f))

        # fl = F(x) / f(x)
        fl = Divide(d(f), f)

        # B = g(x) * fl
        b = Multiply(g, fl)

        # returning y(x) * (a+b)
        return Multiply(Exponent(f, g), Add(a, b))
    raise NotImplementedError
    
derivative = add_wrap(_derivative)

## ************************************

## Derivative shorthand

class _ReferenceBase(object):
    """Base type for _Reference creator"""

def _Reference(attributes={}, overload={}):
    """Create _Reference object
attributes is the attributes
overload is a method overloader

Example:

two = _Reference(overload={'add':lambda*a:4})
two + two // 4
two + two + two // four
two + two + 2 // four
two + two + two + (-2) // four"""
    return type('_Reference',tuple(_ReferenceBase.mro()),{
        **attributes,
        **{
            '__'.join(['',k,'']): v for k,v in overload.items()
        }
    })()

def _d_div(self, other):
    print('in _d_div')
    if other is dx:
        return derivative
    return derivative(other.expression)

d = _Reference(overload={
    'truediv':_d_div,
    'floordiv':_d_div,
    'call': _derivative,
    'getitem': _derivative
})

def _dx_call(self, other):
    print('in _dx_call')
    return _Reference(attributes={'expression': other})

dx = _Reference(overload={'call':_dx_call,'getitem':_dx_call})
dn = dx

## ************************************

## Evaluating

def evaluate(expr):
    ist = isinstance
    if ist(expr, Expression):
        e = expr.body
    elif ist(expr, ExpressionBase):
        e = expr
    elif ist(expr, _NUMBER_TYPE):
        return expr
    else:
        raise TypeError(*TYPE_EE)
    if ist(e, OperatorBase):
        return eval(f'evaluate(e.left) {e.sec} evaluate(e.right)')
    if ist(e, Variable):
        if e._resolved():
            return e._get()
        raise ValueError(*PLEASE_CONVERT)
    if ist(e, Function):
        return e.func_dict[e.name][0](evaluate(e.arg))
    raise NotImplementedError

## ************************************

## Simplifying

def simplify(expr):
    ist = isinstance
    if ist(expr, Expression):
        e = expr.body
    elif ist(expr, ExpressionBase):
        e = expr
    elif ist(expr, _NUMBER_TYPE):
        return expr
    else:
        raise TypeError(*TYPE_S)
    if ist(e, OperatorBase):
        a = simplify(e.left)
        b = simplify(e.right)
        if b == 0:
            return ({
                      Add: lambda:[a]
                ,Subtract: lambda:[a]
                ,Multiply: lambda:[0]
                  ,Divide: lambda:[exec("Divide(1,0)")]
                ,Exponent: lambda:[
                    ({
                        0: lambda:exec("raise ValueError(\"Cannot raise 0 to the power of 0\")")
                    }).get(a, lambda:1)()
                ]
            })[type(e)]()[0]
        if b == 1:
            return ({
                      Add: e
                ,Subtract: e
            }).get(type(e),a)
        if a == 0:
            return ({
                      Add: b
                ,Subtract: e
            }).get(type(e), 0)
        if a == 1:
            return ({
                 Multiply: b
                ,Exponent: 1
            }).get(type(e),e)
        if a == b and isinstance(e, (Subtract, Divide)):
            return 0 if isinstance(e, Subtract) else (1 if a == 0 else ValueError)
        return type(e)(a,b)
    if ist(e, Variable):
        return e
    if ist(e, Function):
        return Function(e.name, simplify(e.arg))
    raise NotImplementedError

## ************************************

## Parsing

def _parse_get():
    case_letters = ''.join([chr(i) for i in range(65,91)])

    letters = ''.join([case_letters.upper(), case_letters.lower()])

    punctuation = '()[]{}<>≤≥=+-*/x÷!.,%'

    numbers = '1234567890'

    other = 'πß∆¬≈√‰∏'

    return ''.join([
        letters,
        punctuation,
        numbers,
        other
    ])

_PARSE_WHITELIST = _parse_get()

def _parse(string):
    # Problem case
    # (1+2)
    # (1)+(2)

    # Step 1: Ensure it is a string
    if not isinstance(string, str):
        raise TypeError('Non-string passed to parse()','_parse()')
    if type(string) != str:
        raise TypeError(
            'Strings must be of type str and not a subclass',
            '_parse()'
        )

    # Step 2: Remove whitespace
    string = ''.join([i for i in string if i not in '\t\n \r'])

    # Step 3: Remove non-whitelisted characters
    for index, char in enumerate(string):
        if char not in _PARSE_WHITELIST:
            raise ValueError(
                ' '.join([
                    "Illegal char - at position",
                    index
                ]),
                '_parse()'
            )
        # char is okay, so go to next char

    # Step 4: I CANNOT DO THIS ANYMORE

    raise NotImplementedError("ITS TOO HARD I CANNOT PARSE ANYTHING I GIVE UP")
        
    return 'UNFINISHED'

parse = add_wrap(_parse)

## ************************************

## Variable bonding within evaluate()

class VariableBond(object):
    """A class allowing variable bonding within evaluate()
To bond variables, use one of the following:

with VariableBond(3):
    pass # do stuff here where variables are 3
# do stuff here where variables are no longer 3

OR:

Variable.set(3)
# do stuff here where variables are 3
Variable.clear()
# do stuff here where variables are no longer 3"""
    def __init__(self, value):
        self._var_val = value
    def __enter__(self):
        Variable.set(self._var_val)
        return None
    def __exit__(self, *args):
        Variable.clear()
        return False

x = Variable('x')
n = Variable('n')

if __name__ == '__main__':
    with VariableBond(3):
        # all Variable() instances here are 3 in evaluate()
        print('d/dx',
            d/dx[
                wrap(tan(tan(x)))
            ]
        )
        print('evaluate',
            evaluate(
                wrap(tan(tan(x)))
            )
        )
        print('simplify',
            simplify(
                wrap(tan(tan(x)))
            )
        )
        print('parse',
            parse(
                "tan(tan(x))"
            )
        )
    # all Variable() instances have no value.
