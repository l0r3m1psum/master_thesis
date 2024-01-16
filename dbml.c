#include <stdbool.h>
#include <stddef.h>
#include <limits.h>

#define BREAKPOINT __builtin_debugtrap()
#define MAX(a, b) (a) > (b) ? (a) : (b);

#pragma Assertion

#include <stdio.h>
#ifdef __APPLE__
#include <execinfo.h>
#define PRINT_BACKTRACE do { \
		void* callstack[128]; \
		int frames = backtrace(callstack, 128); \
		int stderr = 2; \
		backtrace_symbols_fd(callstack, frames, stderr); \
	} while (false)
#else
#define PRINT_BACKTRACE
#endif

#define ASSERT(test) do { \
		if (!(test)) { \
			fprintf(stderr, "ASSERT: %s:%s:%d: %s\n", __FILE__, __func__, __LINE__, #test); \
			fflush(stderr); \
			PRINT_BACKTRACE; \
			*((volatile int *)0) = 0; \
		} \
	} while (false)

#pragma mark Memory Allocator

typedef struct Arena {
	void  *data;
	size_t size;
	size_t used;
} Arena;

#define Arena_FROM_ARRAY(a) \
	(Arena){.data = (a), .size = sizeof (a), .used = 0}

static bool
Arena_invariant(const Arena *alloc) {
	if (!alloc) return false;
	if (alloc->data == NULL) {
		return (alloc->size == 0) & (alloc->used == 0);
	}
	return alloc->used <= alloc->size;
}

#if 0
// NOTE: the operation like this is a bit unsafe in the sense that we could
// forget that we have created a sub allocator (with from_unsused) and keep
// using the original arena (which effectivelly should have all of it's memory
// used). This can create some subtle memory corruption bugs. A possible
// solution can be to set alloc->used = alloc->size when we create the sub
// allocator to "exahust" the memory in the original allocator and then use
// anoter method to give back said memeory (something like return_unsused). In
// this metod we would just set alloc->size -= unused->size and unused->size =
// unused->used = 0 (just to be sure). This is all great but if we start
// stacking sub allocators on top of sub allocatos keeping a record of the order
// in which operations have to be performed could be beneficial. Something like
// a static stack (of size N) that keeps track of all allocations contexts.
static Arena
Arena_from_unused(const Arena alloc[static 1]) {
	ASSERT(Arena_invariant(alloc));

	Arena res = {
		.data = alloc->data + alloc->used,
		.size = alloc->size - alloc->used,
		.used = 0
	};

	ASSERT(Arena_invariant(&res));
	return res;
}
#endif

static void *
Arena_alloc(Arena alloc[static 1], size_t req_size) {
	ASSERT(Arena_invariant(alloc));
	if (alloc->size - alloc->used < req_size) {
		return NULL;
	}
	void *res = alloc->data + alloc->used;
	alloc->used += req_size;
	return res;
}

static void
Arena_reset(Arena alloc[static 1]) {
	ASSERT(Arena_invariant(alloc));
	alloc->used = 0;
	ASSERT(Arena_invariant(alloc));
}

#pragma mark Hash Set

typedef struct HashSet {
	uintptr_t *table;
	size_t length;
} HashSet;

static bool HashSet_invariant(struct HashSet *ht) {
	if (!ht) return false;
	if (!ht->table) return false;
	size_t m = ht->length;
	// Testing if it is a power of 2.
	// http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
	return m && !(m & (m - 1));
}

// Design rationale: we are going to allocate pretty much all of our pointers
// from the same base address (because we use an arena) therefore the high part
// of the address is not going to change while the lower part will (also the
// high part is going to be discarded by the modulo operation). Since we do not
// allocate byte sized object some of the initial bits of the address are going
// to be 0. We therefore discard some of them with a right shift. Below some
// example pointers:
//   * 0x10FB91010
//   * 0x10FB91088
//   * 0x10FB91100
//   * 0x10FB91178
//   * 0x10FB911F0
//   * 0x10FB91268
//   * 0x10FB912E0
static size_t hash(uintptr_t k) {
	// NOTE: is this in any way necessary???
#if SIZE_MAX < UINTPTR_MAX
    k %= SIZE_MAX;
#endif
	return k >> 3;
}

static bool HashSet_insert(struct HashSet ht[static 1], uintptr_t key) {
	ASSERT(HashSet_invariant(ht));
	ASSERT(key);
	size_t index = hash(key) & (ht->length - 1);
	size_t start = index;
	while (ht->table[index]) {
		if (ht->table[index] == key) {
			// We don't have to insert the key since it is already in the table.
			return true;
		}
		// printf("collision!\n");
		index = (index + 1) & (ht->length - 1);
		if (index == start) {
			return false; // We have wrapped around and we haven't found it.
		}
	}
	// We have found an empty spot.
	ht->table[index] = key;
	return true;
}

static bool HashSet_search(struct HashSet ht[static 1], uintptr_t key) {
	// benhoyt.com style
	ASSERT(HashSet_invariant(ht));
	ASSERT(key);
	size_t index = hash(key) & (ht->length - 1);
	size_t start = index;
	while (ht->table[index]) {
		if (ht->table[index] == key) {
			return true;
		}
		index = (index + 1) & (ht->length - 1);
		// printf("collision!\n");
		if (index == start) return false; // We have wrapped around and we haven't found it.
	}
	return false;
}

static void HashSet_reset(struct HashSet ht[static 1]) {
	ASSERT(HashSet_invariant(ht));
	for (size_t i = 0; i < ht->length; i++) {
		ht->table[i] = 0;
	}
}

#pragma mark Tensor

typedef enum Type {
	TYPE_FLOAT32,
	TYPE_INT16,
} Type;

size_t Type_sizeof[] = {4, 2};

typedef enum Backend {
	BACKEND_CPU,
	BACKEND_GPU,
} Backend;

typedef enum Operation {
	Operation_nop,
	OPERATION_ADD,
	OPERATION_MUL,
	OPERATION_SUB,
	OPERATION_DIV,
	OPERATION_POW,
	OPERATION_RELU, // NOTE: How do I differentiate this if it it implemented with max?
	OPERATION_MATMUL,
} Operation;

typedef enum TensorError {
	TensorError_OK,
	TensorError_OOM, // Out Of Memory
	TensorError_TYPE_MISMATCH,
	TensorError_SHAPE_MISMATCH,
	TensorError_NULL_WITHOUT_ERROR,
	TensorError_SMALL_DIM,
} TensorError;

const char *TensorError_str[] = {
	"TensorError_OK",
	"TensorError_OOM",
	"TensorError_TYPE_MISMATCH",
	"TensorError_SHAPE_MISMATCH",
	"TensorError_NULL_WITHOUT_ERROR",
	"TensorError_SMALL_DIM",
};

enum Maximums {
	MAX_DIMS = 4,
	MAX_SRC = 6,
};

// NOTE: right now we only support tensors that are contiguous in memory.
// NOTE: right now we only support tensors that own their own memory i.e.
// tensors can't be views of other tensors.
typedef struct Tensor {
	Type    type;
	Backend backend;

	unsigned dim_len;
	unsigned dim_size[MAX_DIMS];

	void *data;
	void *grad;
	struct Tensor *src[MAX_SRC];
	union {
		float exponent;
	};
	Operation operation;

	// TODO: add union for constant parameters. Should this be a union with src?
	// NOTE: A cool thing could be to have prev and next pointer to other
	// tensors as a way to represent the topological sort.
} Tensor;

// NOTE: This should be both "const" and "pure"
static bool
Tensor_invariant(const Tensor *t) {
	if (!t) return true;
	if (t->dim_len == 0) return false;
	if (t->dim_len >= MAX_DIMS) return false;
	if (t->data == NULL) return false;

	// Only the dimensions used shall be non zero.
	for (unsigned i = 0; i < t->dim_len; i++) {
		if (t->dim_size[i] == 0) return false;
	}
	for (unsigned i = t->dim_len; i < MAX_DIMS; i++) {
		if (t->dim_size[i] != 0) return false;
	}

	unsigned not_null_count = 0;
	for (unsigned i = 0; i < MAX_SRC; i++) {
		if (t->src[i] != NULL) not_null_count++;
	}
	// Sources can be non null only in an ordered fashion.
	ASSERT(not_null_count <= MAX_SRC);
	for (unsigned i = 0; i < not_null_count; i++) {
		if (t->src[i] == NULL) return false;
	}
	// If we have a gradient than all of our sources should have it to.
	if (t->grad != NULL) {
		for (unsigned i = 0; i < not_null_count; i++) {
			if (t->src[i]->grad == NULL) return false;
		}
	}

	// TODO: mixing types is allowed in the DAG but it has to be explicitly
	// stated by a node (e.g. a type conversion node or one for mixed precision
	// operation like Nvidia's tensor cores).

	// TODO: check same shape here.
	switch (t->operation) {
	case Operation_nop:
		if (not_null_count != 0) return false;
		break;
	case OPERATION_ADD:
	case OPERATION_MUL:
	case OPERATION_SUB:
	case OPERATION_DIV:
	case OPERATION_MATMUL:
		if (not_null_count != 2) return false;
		break;
	case OPERATION_POW:
	case OPERATION_RELU:
		if (not_null_count != 1) return false;
		break;
	}

	// TODO: add checks for backends.
	// TODO: add checks for the missing union.

	return true;
}

// NOTE: another thing that could be done for errors is to have some statically
// allocated tensors object that represent erroneous computations like in
// IEEE-754 Inf and NaN. In this way NULL wouldn't be the only invalid pointer.
typedef struct Context {
	Arena       *alloc;
	size_t tensor_count; // This can never overflow.
	TensorError  error;
	const char  *file;
	const char  *func;
	int          line;
	// NOTE: Here I can also count the number of created tensors to ease allocation.
	// This would allow me to create an hash table big enough to not have many
	// collisions and completely avoid relocations.
} Context;

static bool
Context_invariant(const Context *ctx) {
	if (!ctx) return false;
	if (!Arena_invariant(ctx->alloc)) return false;
	// If no error occurred that we shall have no "location" information.
	// Otherwise we shall have "location" information. Of course negative line
	// numbers are never admitted.
	if (ctx->error == TensorError_OK) {
		return (!ctx->file) & (!ctx->func) & (ctx->line == 0);
	} else {
		return (!!ctx->file) & (!!ctx->func) & (ctx->line > 0);
	}
}

static void
Context_set_error(
		Context ctx[static 1],
		TensorError error,
		const char file[static 1],
		const char func[static 1],
		int line) {
	ASSERT(Context_invariant(ctx));
	ctx->error = error;
	ctx->file = file;
	ctx->func = func;
	ctx->line = line;
	ASSERT(Context_invariant(ctx));
}

#define Context_FROM_ARENA(alloc_) (Context){.alloc = (alloc_)}

#define Context_FMT(ctx) \
	"%s:%s:%d: %s", (ctx)->file, (ctx)->func, (ctx)->line, TensorError_str[(ctx)->error]

static bool
Tensor_same_shape(const Tensor *a, const Tensor *b) {
	if (!a & !b) return true;
	if (!a | !b) return false;
	unsigned res = 1;
	for (unsigned i = 0; i < MAX_DIMS; i++) {
		res &= a->dim_size[0] == b->dim_size[0];
	}
	return res;
}

static size_t
Tensor_element_number(const Tensor *t) {
	ASSERT(Tensor_invariant(t));
	if (!t) return 0;
	size_t res = 1;
	for (unsigned i = 0; i < t->dim_len; i++) {
		res *= t->dim_size[i]; // TODO: what assertion should I put here to avoid check for overflow.
	}
	return res;
}

static size_t
Tensor_storage_required(const Tensor t[static 1]) {
	ASSERT(Tensor_invariant(t));
	// NOTE: How do I make sure that my numbers are never too big? Maybe by
	// dividing SIZE_MAX/MAX_DIMS? and checking that assigned dimensions sizes
	// are less than equal that. Yeah I think this is OK. But this does not take
	// into account the starting Type_sizeof.
	size_t mem_required = Type_sizeof[t->type];
	for (unsigned i = 0; i < t->dim_len; i++) {
		ASSERT(t->dim_size[i] <= UINT_MAX/MAX_DIMS);
		mem_required *= t->dim_size[i];
	}
	return mem_required;
}

static bool
Tensor_assign_value(Tensor *t, double value) {
	ASSERT(Tensor_invariant(t));
	if (!t) return false;
	if (t->operation != Operation_nop) return false;

	size_t elements = Tensor_element_number(t);

	switch (t->type) {
	case TYPE_FLOAT32:
		for (size_t i = 0; i < elements; i++)
			((float *) t->data)[i] = value;
		break;
	case TYPE_INT16:
		ASSERT(false);
		break;
	}

	return true;
}

static Tensor *
Tensor_new(
		Context ctx[static 1],
		Type type,
		unsigned dim_len,
		const unsigned dim_size[static dim_len],
		const char file[static 1],
		const char func[static 1],
		int line) {
	ASSERT(Context_invariant(ctx));
	ASSERT((0 < dim_len) & (dim_len <= MAX_DIMS));

	if (ctx->error) {
		return NULL;
	}

	size_t mem_required = Type_sizeof[type] * dim_size[0];
	for (unsigned i = 1; i < dim_len; i++) {
		mem_required *= dim_size[i];
	}

	Tensor *res = Arena_alloc(ctx->alloc, sizeof *res + mem_required);
	void *data = res + 1;
	if (!res) {
		Context_set_error(ctx, TensorError_OOM, file, func, line);
		return NULL;
	}

	*res = (Tensor) {
		.type = type,
		.backend = BACKEND_CPU,
		.dim_len = dim_len,
		.dim_size = {}, // Assigned later.
		.data = data,
		.grad = NULL,
		.src = {},
		.operation = Operation_nop,
	};

	for (unsigned i = 0; i < dim_len; i++) {
		res->dim_size[i] = dim_size[i];
	}

	ctx->tensor_count++;

	ASSERT(Tensor_invariant(res));
	return res;
}

#define Tensor_NEW(ctx, type, dim_len, dim_size) \
	Tensor_new((ctx), (type), (dim_len), (dim_size), __FILE__, __func__, __LINE__)

static Tensor *
Tensor_element_wise_operation(
	Context ctx[static 1],
	Operation operation,
	Tensor *a,
	Tensor *b,
	const char file[static 1],
	const char func[static 1],
	int line
	) {
	ASSERT(Context_invariant(ctx));
	ASSERT(Tensor_invariant(a));
	ASSERT(Tensor_invariant(b));

	if (ctx->error) {
		return NULL;
	}
	if (!a | !b) {
		ASSERT(!ctx->error);
		Context_set_error(ctx, TensorError_NULL_WITHOUT_ERROR, file, func, line);
		return NULL;
	}
	if (!Tensor_same_shape(a, b)) {
		Context_set_error(ctx, TensorError_SHAPE_MISMATCH, file, func, line);
		return NULL;
	}
	if (a->type != b->type) {
		Context_set_error(ctx, TensorError_TYPE_MISMATCH, file, func, line);
		return NULL;
	}

	size_t mem_required = Tensor_storage_required(a);

	Tensor *res = Arena_alloc(ctx->alloc, sizeof *res + mem_required);
	void *data = res + 1;
	if (!res) {
		Context_set_error(ctx, TensorError_OOM, file, func, line);
		return NULL;
	}

	*res = (Tensor) {
		.type = a->type,
		.backend = BACKEND_CPU,
		.dim_len = a->dim_len,
		.dim_size = {}, // later
		.data = data,
		.grad = NULL,
		.src = {a, b},
		.operation = operation,
	};

	for (unsigned i = 0; i < a->dim_len; i++) {
		res->dim_size[i] = a->dim_size[i];
	}

	ctx->tensor_count++;

	ASSERT(Tensor_invariant(res));
	return res;
}

static Tensor *
Tensor_add(
		Context ctx[static 1],
		Tensor *a,
		Tensor *b,
		const char file[static 1],
		const char func[static 1],
		int line
	) {
	return Tensor_element_wise_operation(ctx, OPERATION_ADD, a, b, file, func, line);
}

#define Tensor_ADD(ctx, a, b) \
	Tensor_add((ctx), (a), (b), __FILE__, __func__, __LINE__)

static Tensor *
Tensor_mul(
		Context ctx[static 1],
		Tensor *a,
		Tensor *b,
		const char file[static 1],
		const char func[static 1],
		int line
		) {
	return Tensor_element_wise_operation(ctx, OPERATION_MUL, a, b, file, func, line);
}

#define Tensor_MUL(ctx, a, b) \
	Tensor_mul((ctx), (a), (b), __FILE__, __func__, __LINE__)

static Tensor *
Tensor_sub(
		Context ctx[static 1],
		Tensor *a,
		Tensor *b,
		const char file[static 1],
		const char func[static 1],
		int line
		) {
	return Tensor_element_wise_operation(ctx, OPERATION_SUB, a, b, file, func, line);
}

#define Tensor_SUB(ctx, a, b) \
	Tensor_sub((ctx), (a), (b), __FILE__, __func__, __LINE__)

static Tensor *
Tensor_div(
		Context ctx[static 1],
		Tensor *a,
		Tensor *b,
		const char file[static 1],
		const char func[static 1],
		int line
		) {
	return Tensor_element_wise_operation(ctx, OPERATION_DIV, a, b, file, func, line);
}

#define Tensor_DIV(ctx, a, b) \
	Tensor_div((ctx), (a), (b), __FILE__, __func__, __LINE__)

// NOTE: looking at numpy (and similar libraries) might be a good way to learn
// about broadcasting. Also look at fortran and array languages (including R).
// Looking at the ndarray structure can also be a good thing.
// https://stackoverflow.com/questions/39626233/how-did-numpy-implement-multi-dimensional-broadcasting
// https://numpy.org/doc/stable/user/basics.broadcasting.html
// https://mathematica.stackexchange.com/questions/99171/how-to-implement-the-general-array-broadcasting-method-from-numpy
// https://stackoverflow.com/questions/26948776/where-did-the-term-broadcasting-come-from
// https://en.wikipedia.org/wiki/NumPy
// Also Julia is really cool https://www.youtube.com/watch?v=jS9eouMJf_Y
// NOTE: It would be a good exercise handle broadcasting like
// https://pytorch.org/docs/stable/generated/torch.matmul.html does. Right now
// we are more like https://pytorch.org/docs/stable/generated/torch.mm.html
static Tensor *
Tensor_matmul(
		Context ctx[static 1],
		Tensor *a,
		Tensor *b,
		const char file[static 1],
		const char func[static 1],
		int line) {
	ASSERT(Context_invariant(ctx));
	ASSERT(Tensor_invariant(a));
	ASSERT(Tensor_invariant(b));

	if (ctx->error) {
		return NULL;
	}
	if (!a | !b) {
		ASSERT(!ctx->error);
		Context_set_error(ctx, TensorError_NULL_WITHOUT_ERROR, file, func, line);
		return NULL;
	}
	if ((a->dim_len != 2) | (b->dim_len != 2)) {
		Context_set_error(ctx, TensorError_SMALL_DIM, file, func, line);
		return NULL;
	}
	if (a->dim_size[1] != b->dim_size[0]) {
		Context_set_error(ctx, TensorError_SHAPE_MISMATCH, file, func, line);
		return NULL;
	}
	if (a->type != b->type) {
		Context_set_error(ctx, TensorError_TYPE_MISMATCH, file, func, line);
		return NULL;
	}

	size_t mem_required = Type_sizeof[a->type] * a->dim_size[0] * b->dim_size[1];

	Tensor *res = Arena_alloc(ctx->alloc, sizeof *res + mem_required);
	void *data = res + 1;
	if (!res) {
		Context_set_error(ctx, TensorError_OOM, file, func, line);
		return NULL;
	}

	*res = (Tensor) {
		.type = a->type,
		.backend = BACKEND_CPU,
		.dim_len = 2,
		.dim_size = {a->dim_size[0], b->dim_size[1]},
		.data = data,
		.grad = NULL,
		.src = {a, b},
		.operation = OPERATION_MATMUL,
	};

	ctx->tensor_count++;

	ASSERT(Tensor_invariant(res));
	return res;
}

#define Tensor_MATMUL(ctx, a, b) \
	Tensor_matmul((ctx), (a), (b), __FILE__, __func__, __LINE__)

static Tensor *
Tensor_relu(
		Context ctx[static 1],
		Tensor *t,
		const char file[static 1],
		const char func[static 1],
		int line) {
	ASSERT(Context_invariant(ctx));
	ASSERT(Tensor_invariant(t));

	if (ctx->error) {
		return NULL;
	}
	if (!t) {
		ASSERT(!ctx->error);
		Context_set_error(ctx, TensorError_NULL_WITHOUT_ERROR, file, func, line);
		return NULL;
	}

	size_t mem_required = Tensor_storage_required(t);

	Tensor *res = Arena_alloc(ctx->alloc, sizeof *res + mem_required);
	void *data = res + 1;
	if (!res) {
		Context_set_error(ctx, TensorError_OOM, file, func, line);
		return NULL;
	}

	*res = (Tensor) {
		.type = t->type,
		.backend = BACKEND_CPU,
		.dim_len = t->dim_len,
		.dim_size = {}, // later
		.data = data,
		.grad = NULL,
		.src = {t},
		.operation = OPERATION_RELU,
	};

	for (unsigned i = 0; i < t->dim_len; i++) {
		res->dim_size[i] = t->dim_size[i];
	}

	ctx->tensor_count++;

	ASSERT(Tensor_invariant(res));
	return res;
}

#define Tensor_RELU(ctx, a) \
	Tensor_relu((ctx), (a), __FILE__, __func__, __LINE__)

static Tensor *
Tensor_pow(
		Context ctx[static 1],
		Tensor *t,
		float exponent,
		const char file[static 1],
		const char func[static 1],
		int line) {
	ASSERT(Context_invariant(ctx));
	ASSERT(Tensor_invariant(t));

	if (ctx->error) {
		return NULL;
	}
	if (!t) {
		ASSERT(!ctx->error);
		Context_set_error(ctx, TensorError_NULL_WITHOUT_ERROR, file, func, line);
		return NULL;
	}

	size_t mem_required = Tensor_storage_required(t);

	Tensor *res = Arena_alloc(ctx->alloc, sizeof *res + mem_required);
	void *data = res + 1;
	if (!res) {
		Context_set_error(ctx, TensorError_OOM, file, func, line);
		return NULL;
	}

	*res = (Tensor) {
		.type = t->type,
		.backend = BACKEND_CPU,
		.dim_len = t->dim_len,
		.dim_size = {}, // later
		.data = data,
		.grad = NULL,
		.src = {t},
		.exponent = exponent,
		.operation = OPERATION_POW,
	};

	for (unsigned i = 0; i < t->dim_len; i++) {
		res->dim_size[i] = t->dim_size[i];
	}

	ctx->tensor_count++;

	ASSERT(Tensor_invariant(res));
	return res;
}

// This holds the reverse of the topological order i.e. when the source vertex
// is the last vertex and the first vertex is a sink. This is what you need for
// the forward calculation.
typedef struct ExecutionOrder {
	const Tensor **tensors;
	size_t count;
} ExecutionOrder;

// If a pointer to a Tensor is not null then it points to a node in a graph with
// no nulls in it. This is due to the way in which the funcitons that construct
// the DAG are structured. I should assert this in some way. This means that
// when I'm doing the ordering for execution I only need to check for the first
// pointer being null, then when I start following the childs there will be no
// nulls. Therefore in the order if tensors point to a list it contains only non
// null pointers.

static bool
ExecutionOrder_invariant(ExecutionOrder order) {
	if (!order.tensors) return order.count == 0;
	// If the pointer to tensors is not null then we must have at least one.
	if (!order.count) return false;
	// The first operation must always be a leaf.
	if (order.tensors[0]->operation != Operation_nop) return false;

	// We have to check that edge only go backward because we are testing for a
	// reversed topological order.
	for (size_t i = order.count-1; i --> 0;) {
		if (order.tensors[i] == Operation_nop) continue;
		for (size_t j = i+1; j < order.count-1; j++) {
			for (unsigned k = 0; k < MAX_SRC && order.tensors[i]->src[k]; k++)
				 if (order.tensors[i]->src[k] == order.tensors[j])
					 return false;
		}
	}
	// The last operation should always be a non leaf (unless we have only one
	// operation).
	if (order.count > 1 && order.tensors[order.count-1]->operation == Operation_nop)
		return false;

	return true;
}

// This function is used just for testing.
static void
Tensor_plan_execution_recursive_internal(
		Tensor root[static 1],
		HashSet visited[static 1],
		ExecutionOrder order[static 1]
	) {
	ASSERT(Tensor_invariant(root));
	ASSERT(HashSet_invariant(visited));

	if (HashSet_search(visited, (uintptr_t)root)) {
		return;
	}
	bool res = HashSet_insert(visited, (uintptr_t)root);
	ASSERT(res);

	for (unsigned i = 0; i < MAX_SRC && root->src[i]; i++) {
		Tensor_plan_execution_recursive_internal(root->src[i], visited, order);
	}

	// printf("0x%" PRIXPTR "\n", (uintptr_t)root);
	order->tensors[order->count++] = root;
}

// This function is used just for testing.
static ExecutionOrder
Tensor_plan_execution_recursive(Tensor *root) {
	if (!root) return (ExecutionOrder){};
	
	HashSet *visited = &(HashSet){
		.table = (uintptr_t [16]){0},
		.length = 16,
	};
	ExecutionOrder *order = &(ExecutionOrder){
		.tensors = (const Tensor *[16]){0},
		.count = 0,
	};
	
	Tensor_plan_execution_recursive_internal(root, visited, order);
	ASSERT(ExecutionOrder_invariant(*order));
	
	return *order;
}

#include <math.h>

// https://www.codeproject.com/Articles/418776/How-to-replace-recursive-functions-using-stack-and
static ExecutionOrder
Tensor_plan_execution(
		Context ctx[static 1],
		Tensor *root,
		const char file[static 1],
		const char func[static 1],
		int line) {
	ASSERT(Context_invariant(ctx));
	ASSERT(Tensor_invariant(root));
	// TODO: we can assert that root (and all other tensors for that matter) are
	// from the context's area with the following test:
	//     arena->data <= tensor < arena->data + arena->used

	if (ctx->error) {
		return (ExecutionOrder){};
	}
	if (!root) {
		ASSERT(!ctx->error);
		Context_set_error(ctx, TensorError_NULL_WITHOUT_ERROR, file, func, line);
		return (ExecutionOrder){};
	}

	// Note that even if we don't start from the source vertex of the DAG and
	// last allocated tensor, ctx->tensor_count is going to make us allocate
	// more than what is strictly needed but this is fine.
	ExecutionOrder order = {
		.tensors = Arena_alloc(ctx->alloc, sizeof (Tensor *) * ctx->tensor_count),
		.count = 0,
	};
	if (!order.tensors) {
		Context_set_error(ctx, TensorError_OOM, file, func, line);
		return (ExecutionOrder){};
	}

	typedef struct Snapshot {
		Tensor *root;
		unsigned i;
		enum {BEFORE_CALL, AFTER_CALL} stage;
	} Snapshot;

	// TODO: make a better implementation for temporary memory.
	size_t arena_used_restore_point = ctx->alloc->used;

	size_t top = 0;
	Snapshot *stack = Arena_alloc(ctx->alloc, sizeof *stack * ctx->tensor_count);
	if (!stack) {
		Context_set_error(ctx, TensorError_OOM, file, func, line);
		return (ExecutionOrder){};
	}

	// We allocate 70% more than needed to keep the fill factor (i.e.
	// occupied/total) a.k.a. alpha always below 0.7. Furthermore we round up
	// our allocation to the nearest power of 2.
	double fill_factor_overhead = 1.7;
	// https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2
	// TODO: double check this math.
	size_t set_size = (size_t)1 << (size_t)log2(ctx->tensor_count * fill_factor_overhead + 1);
	HashSet *visited = &(HashSet){
		.table = Arena_alloc(ctx->alloc, sizeof (uintptr_t) * set_size),
		.length = set_size,
	};
	if (!visited->table) {
		Context_set_error(ctx, TensorError_OOM, file, func, line);
		ctx->alloc->used = arena_used_restore_point;
		return (ExecutionOrder){};
	}

	stack[top++] = (Snapshot){.root = root, .i = 0, .stage = 0,};

	while (top) {
		Snapshot current = stack[--top];
		switch (current.stage) {
		case BEFORE_CALL:
			if (HashSet_search(visited, (uintptr_t)current.root)) {
				continue;
			}
			bool res = HashSet_insert(visited, (uintptr_t)current.root);
			ASSERT(res);
			current.i = 0;
			if (current.i < MAX_SRC && current.root->src[current.i]) {
				current.stage = AFTER_CALL;
				stack[top++] = current;
				stack[top++] = (Snapshot){
					.root = current.root->src[current.i],
					.i = 0,
					.stage = BEFORE_CALL,
				};
				continue;
			}

			// printf("0x%" PRIXPTR "\n", (uintptr_t)current.root);
			order.tensors[order.count++] = current.root;
			break;
		case AFTER_CALL:
			current.i++;
			if (current.i < MAX_SRC && current.root->src[current.i]) {
				stack[top++] = current;
				stack[top++] = (Snapshot){
					.root = current.root->src[current.i],
					.i = 0,
					.stage = BEFORE_CALL,
				};
				continue;
			}

			// printf("0x%" PRIXPTR "\n", (uintptr_t)current.root);
			order.tensors[order.count++] = current.root;
			break;
		}
	}

	ctx->alloc->used = arena_used_restore_point;
	ASSERT(order.tensors[order.count-1] == root);
	ASSERT(ExecutionOrder_invariant(order));
	return order;
}

#define Tensor_PLAN_EXECUTION(ctx, root) \
	Tensor_plan_execution((ctx), (root), __FILE__, __func__, __LINE__)

static void
Tensor_compute_graph(ExecutionOrder order) {
	for (size_t i = 0; i < order.count; i++) {
		const Tensor *t = order.tensors[i];
		ASSERT(Tensor_invariant(t));
		switch (t->operation) {
		case Operation_nop:
			continue; // t should be a leaf of the graph.
		case OPERATION_ADD:
			switch (t->type) {
			case TYPE_FLOAT32:
				ASSERT(Tensor_same_shape(t, t->src[0]));
				ASSERT(Tensor_same_shape(t->src[0], t->src[1]));
				size_t elements = Tensor_element_number(t);
				for (size_t i = 0; i < elements; i++) {
					float *lhs = ((float *) t->src[0]->data) + i;
					float *rhs = ((float *) t->src[1]->data) + i;
					((float *) t->data)[i] = *lhs + *rhs;
				}
				break;
			case TYPE_INT16:
				ASSERT(false);
				break;
			}
			break;
		case OPERATION_MUL:
			switch (t->type) {
			case TYPE_FLOAT32:
				ASSERT(Tensor_same_shape(t, t->src[0]));
				ASSERT(Tensor_same_shape(t->src[0], t->src[1]));
				size_t elements = Tensor_element_number(t);
				for (size_t i = 0; i < elements; i++) {
					float *lhs = ((float *) t->src[0]->data) + i;
					float *rhs = ((float *) t->src[1]->data) + i;
					((float *) t->data)[i] = *lhs * *rhs;
				}
				break;
			case TYPE_INT16:
				ASSERT(false);
				break;
			}
			break;
		case OPERATION_SUB:
			switch (t->type) {
			case TYPE_FLOAT32:
				ASSERT(Tensor_same_shape(t, t->src[0]));
				ASSERT(Tensor_same_shape(t->src[0], t->src[1]));
				size_t elements = Tensor_element_number(t);
				for (size_t i = 0; i < elements; i++) {
					float *lhs = ((float *) t->src[0]->data) + i;
					float *rhs = ((float *) t->src[1]->data) + i;
					((float *) t->data)[i] = *lhs - *rhs;
				}
				break;
			case TYPE_INT16:
				ASSERT(false);
				break;
			}
			break;
		case OPERATION_DIV:
			switch (t->type) {
			case TYPE_FLOAT32:
				ASSERT(Tensor_same_shape(t, t->src[0]));
				ASSERT(Tensor_same_shape(t->src[0], t->src[1]));
				size_t elements = Tensor_element_number(t);
				for (size_t i = 0; i < elements; i++) {
					float *lhs = ((float *) t->src[0]->data) + i;
					float *rhs = ((float *) t->src[1]->data) + i;
					((float *) t->data)[i] = *lhs / *rhs;
				}
				break;
			case TYPE_INT16:
				ASSERT(false);
				break;
			}
			break;
		case OPERATION_MATMUL:
			switch (t->type) {
			case TYPE_FLOAT32: {
				// TODO: assert shapes are correct.
				unsigned lhs_rows = t->src[0]->dim_size[0];
				unsigned lhs_cols = t->src[0]->dim_size[1];
				unsigned rhs_rows = t->src[1]->dim_size[0];
				unsigned rhs_cols = t->src[1]->dim_size[1];
				float (*lhs)[lhs_rows][lhs_cols] = t->src[0]->data;
				float (*rhs)[rhs_rows][rhs_cols] = t->src[1]->data;
				float (*res)[lhs_rows][rhs_cols] = t->data;
				for (size_t row_lhs = 0; row_lhs < lhs_rows; row_lhs++)
				for (size_t col_rhs = 0; col_rhs < rhs_cols; col_rhs++) {
					(*res)[row_lhs][col_rhs] = 0;
					for (size_t i = 0; i < lhs_cols; i++)
						(*res)[row_lhs][col_rhs] += (*lhs)[row_lhs][i] * (*rhs)[i][col_rhs];
				}
				} break;
			case TYPE_INT16:
				ASSERT(false);
				break;
			}
			break;
		case OPERATION_POW:
			switch (t->type) {
			case TYPE_FLOAT32:
				ASSERT(Tensor_same_shape(t, t->src[0]));
				size_t elements = Tensor_element_number(t);
				for (size_t i = 0; i < elements; i++) {
					((float *) t->data)[i] = pow(((float *) t->src[0]->data)[i], t->exponent);
				}
				break;
			case TYPE_INT16:
				ASSERT(false);
				break;
			}
			break;
		case OPERATION_RELU:
			switch (t->type) {
			case TYPE_FLOAT32:
				ASSERT(Tensor_same_shape(t, t->src[0]));
				size_t elements = Tensor_element_number(t);
				for (size_t i = 0; i < elements; i++) {
					float n = ((float *) t->src[0]->data)[i];
					((float *) t->data)[i] = MAX(0, n);
				}
				break;
			case TYPE_INT16:
				ASSERT(false);
				break;
			}
			break;
		}
	}
}

#include <string.h>

int main(void) {
	static unsigned char mem[1 << 16]; // 65 Kibibyte.
	Context *ctx = &Context_FROM_ARENA(&Arena_FROM_ARRAY(mem));

	Tensor *a = Tensor_NEW(ctx, TYPE_FLOAT32, 2, ((unsigned[]){2,2}));
	Tensor_assign_value(a, 1);
	Tensor *b = Tensor_NEW(ctx, TYPE_FLOAT32, 2, ((unsigned[]){2,2}));
	Tensor_assign_value(b, -1);
	Tensor *c = Tensor_ADD(ctx, a, b);
	Tensor *d = Tensor_MUL(ctx, a, b);
	Tensor *e = Tensor_SUB(ctx, a, b);
	Tensor *f = Tensor_DIV(ctx, a, b);
	// A diamond.
	Tensor *g = Tensor_RELU(ctx, a);
	Tensor *h = Tensor_RELU(ctx, a);
	Tensor *i = Tensor_ADD(ctx, g, h);

	{
		ExecutionOrder order_rec = Tensor_plan_execution_recursive(i);
		ExecutionOrder order_iter = Tensor_PLAN_EXECUTION(ctx, i);
		ASSERT(order_rec.count == order_iter.count);
		ASSERT(memcmp(order_rec.tensors, order_iter.tensors, order_iter.count) == 0);
	}

	{
		ExecutionOrder order = Tensor_PLAN_EXECUTION(ctx, i);
		Tensor_compute_graph(order);
		for (int ii = 0; ii < 2*2; ii++) {
			printf("%f\n", ((float *) i->data)[ii]);
		}
		printf("\n");
		order = Tensor_PLAN_EXECUTION(ctx, c);
		Tensor_compute_graph(order);
		for (int ii = 0; ii < 2*2; ii++) {
			printf("%f\n", ((float *) c->data)[ii]);
		}
	}
	
	{
		static unsigned char mem[1 << 16]; // 65 Kibibyte.
		Context *ctx = &Context_FROM_ARENA(&Arena_FROM_ARRAY(mem));

		Tensor *A = Tensor_NEW(ctx, TYPE_FLOAT32, 2, ((unsigned[]){4,3}));
		memcpy(A->data, (float [4][3]){
			{1,2,3},
			{4,5,6},
			{7,8,9},
			{3,2,1},
		}, sizeof (float) * 4 * 3);
		Tensor *x = Tensor_NEW(ctx, TYPE_FLOAT32, 2, ((unsigned[]){3,1}));
		memcpy(x->data, (float [3][1]){
			{1},
			{2},
			{3},
		}, sizeof (float) * 3 * 1);
		Tensor *res = Tensor_MATMUL(ctx, A, x);
		ExecutionOrder order = Tensor_PLAN_EXECUTION(ctx, res);
		Tensor_compute_graph(order);
		printf("\n");
		for (int i = 0; i < 4*1; i++) {
			printf("%f\n", ((float *) res->data)[i]);
		}
	}

	// Obiettivi:
	//   4. calcolare il gradiente del grafo.
	//   5. addestrare un semplice MLP.

	if (ctx->error != TensorError_OK) {
		printf(Context_FMT(ctx));
	}
	return ctx->error;
}

// Linear layer illustration.
//     ([ * * * ]         [ * ])
//     ([ * * * ] [ * ]   [ * ])
// ReLU([ * * * ] [ * ] + [ * ])
//     ([ * * * ] [ * ]   [ * ])
//     ([ * * * ]         [ * ])
//         5x3     3x1     4x1
