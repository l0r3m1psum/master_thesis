class Value:
	def __init__(self, data, _children=()):
		self.data = data
		self._prev = set(_children)
	def __repr__(self):
		return f"{self.data}"

topo = []
visited = set()
def build_topo(v):
	if v not in visited:
		visited.add(v)
		for child in v._prev:
			build_topo(child)
		topo.append(v)

def build_topo2(v):
	linear_order = []
	visited = set()
	def build_topo2_internal(v, linear_order, visited):
		if v in visited:
			return
		visited.add(v)
		for child in v._prev:
			build_topo2_internal(child, linear_order, visited)
		linear_order.append(v)
	build_topo2_internal(v, linear_order, visited)
	return linear_order

inner_dag = Value(4, (Value(5), Value(6)))
dag = Value(1, (Value(2, (inner_dag,)), Value(3, (inner_dag,))))

build_topo(dag)
print(topo)
print(build_topo2(dag))
