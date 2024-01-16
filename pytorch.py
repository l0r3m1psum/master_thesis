import torch

# here [7,8] should be a row
# torch.kron(torch.tensor([7,8]), torch.eye(2,dtype=torch.int64))
T = torch.tensor([
 [[7,0],
  [0,7]],
 [[8,0],
  [0,8]]
])
# x should be a column
x = torch.tensor([56,118])
A = T@x
print(A)

T = torch.arange(6).reshape(3, 2)
T = torch.stack([T]*3)
T = torch.stack([T]*2)
print("For some reason the shape is different from the one in the notes. But "
	"the calculations work as expected:", T.size(), sep="\n")
a = torch.tensor([1, 0, 0])
b = torch.tensor([1, 2])
print("a@T@b", a@T@b, sep="\n")

# https://github.com/alugowski/matrepr
# TODO: add support for format specifiers
def to_latex(T: torch.Tensor) -> str:
	if T.ndim == 0:
		return str(T.item())
	else:
		has_even_dimensions = T.ndim%2 == 0
		# TODO: the optional extra-space argument shoul be relative to the font
		# size and the dimension we are printing the bigger one of the two the
		# bigger the space.
		new_line = "\\\\" + ("[20pt]" if T.ndim > 2 else "") + "\n"
		if has_even_dimensions:
			dim_size = T.size()[1]
			res = "\\left [\\begin{array}{" + "c"*dim_size + "}\n"
			res += new_line.join(
				"\n&".join(to_latex(A) for A in T[i]) for i in range(T.size()[0])
			)
			res += "\n\\end{array}\\right ]"
		else:
			res = "\\left [\\begin{array}{c}\n"
			res += new_line.join(to_latex(A) for A in T)
			res += "\n\\end{array}\\right ]"
		return res

if False:
	print(to_latex(torch.tensor(1)))
	print(to_latex(torch.arange(5)))
	print(to_latex(torch.eye(5)))
	print()
	print(to_latex(torch.zeros(2,3,2)))
	print()
	print(to_latex(T))
	print()
	print(to_latex(torch.zeros(2,3,2,4,3)))

LEFT_SQUARE_BRACKET               = "\u005B"
RIGHT_SQUARE_BRACKET              = "\u005D"
LEFT_SQUARE_BRACKET_UPPER_CORNER  = "\u23A1"
LEFT_SQUARE_BRACKET_EXTENSION     = "\u23A2"
LEFT_SQUARE_BRACKET_LOWER_CORNER  = "\u23A3"
RIGHT_SQUARE_BRACKET_UPPER_CORNER = "\u23A4"
RIGHT_SQUARE_BRACKET_EXTENSION    = "\u23A5"
RIGHT_SQUARE_BRACKET_LOWER_CORNER = "\u23A6"

size = T.size()
tot_rows = size[::-2].numel() * (1 if T.ndim%2 else size[0])

def left_bracket(row_index: int, row_number: int) -> str:
	assert 0 <= row_index < row_number
	if row_number == 1: return LEFT_SQUARE_BRACKET
	if row_index == 0: return LEFT_SQUARE_BRACKET_UPPER_CORNER
	if row_index == row_number - 1: return LEFT_SQUARE_BRACKET_LOWER_CORNER
	return LEFT_SQUARE_BRACKET_EXTENSION


# https://en.m.wikipedia.org/wiki/Z-order_curve
# https://stackoverflow.com/questions/12157685/z-order-curve-coordinates
# https://stackoverflow.com/questions/65313532/morton-curve-for-non-cubic-areas-that-are-a-power-of-two
# https://observablehq.com/@stwind/z-order-curve-3d-in-glsl
# https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
# https://demonstrations.wolfram.com/ComparingXYCurvesAndZOrderCurvesForTextureCoding/

def to_unicode_int(T: torch.Tensor, ndim: int, n_digits: int, row_index: int, row_number: int) -> str:
	assert ndim >= 0 and n_digits > 0
	if T.ndim == 0: return "{0:>{n_digits}}".format(T.item(), n_digits=n_digits)
	if T.ndim == 1:
		left = left_bracket(row_index, row_number)
		right = "]"
		num_generator = (to_unicode_int(n, ndim, n_digits, row_index, row_number) for n in T)
		return left + " ".join(num_generator) + right
	first_dimension_is_column = T.ndim%2 == 1
	if first_dimension_is_column:
		left = left_bracket(row_index, row_number)
		right = "]"
		col_generator = (to_unicode_int(col, ndim, n_digits, row_index, row_number) for i, col in enumerate(T))
		return left + "".join(col_generator) + right
	else:
		if T.ndim != ndim:
			left, right = left_bracket(row_index, row_number), "]"
		else:
			left, right = "", ""
		row_generator = (to_unicode_int(row, ndim, n_digits, i, len(T)) for i, row in enumerate(T))
		return left + ("\n" if T.ndim == ndim else "\n").join(row_generator) + right

def to_unicode(T: torch.Tensor) -> str:
	n_digits = (T.max().log10()+1).floor().int().item()
	# This method would work only for natural numbers different from 0. So for
	# now we fix the right allign with 3 spaces.
	n_digits = 3
	if T.ndim > 0:
		first_row_dimension = 0 if T.ndim%2 else 1
		row_number = T.size()[first_row_dimension]
	else:
		row_number = 1
	return to_unicode_int(T, T.ndim, n_digits, 0, row_number)
	
def to_unicode(T: torch.Tensor) -> str:
	if T.ndim == 0: return "{0:>{n_digits}}".format(T.item(), n_digits=3)
	if T.ndim == 1: return "[" + " ".join(to_unicode(n) for n in T) + "]"
	if T.ndim%2 == 1:
		return "\n".join(to_unicode(A) for A in T)
	else:
		return " ".join(to_unicode(T[:,i]) for i in range(len(T)))

if False:
	T = torch.zeros(3,3,3, dtype=torch.int64)
	print(T)
	print(to_unicode(T), end="\n\n")
	print(to_unicode(torch.tensor(1)))
	print(to_unicode(torch.tensor([1,2,3])))
	T = torch.arange(81, dtype=torch.int64).reshape(3,3,3,3)
	print(to_unicode(T), end="\n\n")
	T = torch.arange(27, dtype=torch.int64).reshape(3,3,3)
	print(to_unicode(T), end="\n\n")
	T = torch.arange(18, dtype=torch.int64).reshape(1,2,1,3,1,3)
	print(to_unicode(T), end="\n\n")
	T = torch.arange(162, dtype=torch.int64).reshape(9,1,9,2)
	print(to_unicode(T), end="\n\n")

import numpy as np

def mode_n_product(x, m, mode):
	"https://stackoverflow.com/a/65230131"
   x = np.asarray(x)
   m = np.asarray(m)
   if mode <= 0 or mode % 1 != 0:
       raise ValueError('`mode` must be a positive interger')
   if x.ndim < mode:
       raise ValueError('Invalid shape of X for mode = {}: {}'.format(mode, x.shape))
   if m.ndim != 2:
       raise ValueError('Invalid shape of M: {}'.format(m.shape))
   return np.swapaxes(np.swapaxes(x, mode - 1, -1).dot(m.T), mode - 1, -1)

res = np.ones((3,3,3)) @ np.eye(3) == mode_n_product(np.ones((3,3,3)), np.eye(3), 2)
print(res)

T = (torch.arange(18)+1).reshape(3,3,2).float()
I = torch.eye(2)
res = torch.matmul(T, I) == (I @ T.reshape(2, -1)).reshape(3,3,2)
print(res)
