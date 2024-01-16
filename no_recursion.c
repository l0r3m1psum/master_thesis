#include <stddef.h>

struct BinaryTree {
	int data;
	struct BinaryTree *left;
	struct BinaryTree *right;
};

typedef struct BinaryTree BinaryTree;

BinaryTree *tree = &(BinaryTree){
	.data = 0,
	.left  = &(BinaryTree){
		.data = 1,
		.left = NULL,
		.right = &(BinaryTree){
			.data = 2,
		}
	},
	.right = &(BinaryTree){
		.data = 3,
		.left = &(BinaryTree){
			.data = 4,
		}, .right = &(BinaryTree){
			.data = 5,
		}
	}
};

#include <stdio.h>
#include <inttypes.h>
#include <assert.h>

void pre_order_recursive_dfs(BinaryTree *node) {
	if (node == NULL) return;
	printf("0x%" PRIXPTR " %d\n", (uintptr_t)node, node->data);
	pre_order_recursive_dfs(node->left);
	pre_order_recursive_dfs(node->right);
}

void post_order_recursive_dfs(BinaryTree *node) {
	if (node == NULL) return;
	post_order_recursive_dfs(node->left);
	post_order_recursive_dfs(node->right);
	printf("0x%" PRIXPTR " %d\n", (uintptr_t)node, node->data);
}

// https://www.geeksforgeeks.org/iterative-preorder-traversal/
void pre_order_iterative_dfs(BinaryTree *root) {
	if (root == NULL) return;

	BinaryTree *stack[16] = {root};
	size_t top = 1;

	while (top) {
		BinaryTree *node = stack[--top];
		printf("0x%" PRIXPTR " %d\n", (uintptr_t)node, node->data);

		if (node->right) stack[top++] = node->right;
		if (node->left) stack[top++] = node->left;
	}
}

// https://www.geeksforgeeks.org/iterative-postorder-traversal/
// https://www.geeksforgeeks.org/iterative-postorder-traversal-using-stack/
// https://stackoverflow.com/questions/1294701/post-order-traversal-of-binary-tree-without-recursion/16092333#16092333
void post_order_iterative_dfs(BinaryTree *root) {
	if (root == NULL)
		return;

	BinaryTree *stack[16] = {};
	size_t top = 0;

	do {
		while (root) {
			if (root->right)
				stack[top++] = root->right;
			stack[top++] = root;
			root = root->left;
		}

		assert(top);
		root = stack[--top];

		// if (top && (root->right) && (stack[top-1] == root->right)) {
		if (root->right && (top ? stack[top-1] : NULL) == root->right) {
			(void) stack[--top];
			stack[top++] = root;
			root = root->right;
		} else {
			printf("0x%" PRIXPTR " %d\n", (uintptr_t)root, root->data);
			root = NULL;
		}
	} while (top);
}

// https://stackoverflow.com/a/47966326
void knuth_post_order_dfs(BinaryTree *p) {
	BinaryTree   *stack[40] = {};
	BinaryTree  **sp = stack;
	BinaryTree   *last_visited = NULL;

	// We go to the left until we can.
	for (; p != NULL; p = p->left)
		*sp++ = p;

	// Until we do not point at the base of the stack.
	while (sp != stack) {
		p = sp[-1]; // Peek.
		if (p->right == NULL || p->right == last_visited) {
			printf("0x%" PRIXPTR " %d\n", (uintptr_t)p, p->data);
			last_visited = p;
			sp--; // Pop.
		} else {
			for (p = p->right; p != NULL; p = p->left)
				*sp++ = p;
		}
	}
}

void topologicalSortUtil(
		struct Graph* graph,
		int v,
		bool visited[],
		struct Stack** stack
	) {
	visited[v] = true;

	struct List* current = graph->adj[v].next;
	while (current != NULL) {
		int adjacentVertex = current->data;
		if (!visited[adjacentVertex]) {
			topologicalSortUtil(graph, adjacentVertex, visited, stack);
		}
		current = current->next;
	}

	struct Stack* newNode = createStackNode(v);
	newNode->next = *stack;
	*stack = newNode;
}

// The function to do Topological Sort. It uses recursive
// topologicalSortUtil
void topologicalSort(struct Graph* graph) {
	struct Stack* stack = NULL;
	bool* visited = (bool*)malloc(graph->V * sizeof(bool));
	for (int i = 0; i < graph->V; ++i) {
		visited[i] = false;
	}

	for (int i = 0; i < graph->V; ++i) {
		if (!visited[i]) {
			topologicalSortUtil(graph, i, visited, &stack);
		}
	}

	// Free allocated memory
	free(visited);
	free(graph->adj);
	free(graph);
}

int main(void) {

	BinaryTree *inner_dag = &(BinaryTree) {
		.data = 4,
		.left = &(BinaryTree) {.data = 5},
		.right = &(BinaryTree) {.data = 6}
	};
	BinaryTree *dag = &(BinaryTree){
		.data = 1,
		.left = &(BinaryTree){
			.data = 2,
			.right = inner_dag,
		},
		.right = &(BinaryTree){
			.data = 3,
			.left = inner_dag,
		}
	};

	printf("pre-order recursive DFS\n");
	pre_order_recursive_dfs(tree);
	printf("pre-order iterative DFS\n");
	pre_order_iterative_dfs(tree);
	printf("post-order recursive DFS\n");
	post_order_recursive_dfs(tree);
	printf("post-order iterative DFS\n");
	post_order_iterative_dfs(tree);
	// We treat a DAG as a compressed tree.
	printf("pre-order recursive DFS on DAG\n");
	pre_order_recursive_dfs(dag);
	printf("pre-order iterative DFS on DAG\n");
	pre_order_iterative_dfs(dag);
	printf("post-order recursive DFS on DAG\n");
	post_order_recursive_dfs(dag);
	printf("post-order iterative DFS on DAG\n");
	post_order_iterative_dfs(dag);
	printf("Knuth\n");
	knuth_post_order_dfs(dag);

#if 0
	BinaryTree *cycle = &(BinaryTree){
		.data = 1,
		.left = cycle,
	};
	printf("Cycle\n");
	fflush(stdout);
	post_order_recursive_dfs(cycle);
#endif

	return 0;
}
