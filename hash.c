// https://benhoyt.com/writings/hash-table-in-c/
// https://stackoverflow.com/questions/11677201/near-perfect-or-perfect-hash-of-memory-addresses-in-c
// https://www.gnu.org/software/gperf/manual/gperf.html
// https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
// https://github.com/avsej/hashset.c/blob/master/hashset.c
// https://github.com/mtimjones/HashSet he also does stuff with ML
// https://itecnote.com/tecnote/c-the-fastest-hash-function-for-pointers/
// https://opensource.apple.com/source/xnu/xnu-6153.81.5/libkern/os/hash.h.auto.html
// https://www.sultanik.com/blog/HashingPointers
// https://github.com/facebook/folly/blob/main/folly/hash/Hash.h
// https://www.youtube.com/watch?v=2Ti5yvumFTU Jacob Sorber
// https://www.youtube.com/watch?v=KI_V91UdL1I Jacob Sorber
// https://stackoverflow.com/a/58381061
// https://en.wikipedia.org/wiki/Hash_function
// https://stackoverflow.com/a/57556517 (wow really cool visualization)
// https://thenumb.at/Hashtables/
// https://jameshfisher.com/2017/12/10/what-is-open-addressing/
// https://dl.acm.org/doi/pdf/10.1145/1734714.1734729

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <stdio.h>

struct HashTable {
	uintptr_t *table;
	size_t length;
};

static bool HashTable_invariant(struct HashTable *ht) {
	if (!ht) return false;
	if (!ht->table) return false;
	size_t m = ht->length;
	// Testing if it is a power of 2.
	// http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
	return m && !(m & (m - 1));
}

// TODO: implement expansion logic based on filling factor.

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

static size_t HashTable_hash(struct HashTable ht[static 1], uintptr_t key, size_t i) {
	assert(HashTable_invariant(ht));
	assert(key);
	size_t res = (hash(key) + i) & (ht->length - 1);
	assert(res < ht->length);
	return res;
}

// Here we could return a pointer to where we have inserted the uintptr_t
static bool HashTable_insert(struct HashTable ht[static 1], uintptr_t key) {
	// Cormen style.
	assert(HashTable_invariant(ht));
	assert(key);
	size_t i = 0;
	do {
		size_t j = HashTable_hash(ht, key, i);
		if (ht->table[j] == key) {
			// We don't have to insert the key since it is already in the table.
			return true;
		}
		if (!ht->table[j]) {
			ht->table[j] = key;
			return true;
		}
		printf("collision!\n");
	} while(++i < ht->length);
	return false; // The table was full we should reallocate it.
}
static bool HashTable_insert2(struct HashTable ht[static 1], uintptr_t key) {
	// benhoyt.com style
	assert(HashTable_invariant(ht));
	assert(key);
	size_t index = hash(key) & (ht->length - 1);
	size_t start = index;
	while (ht->table[index]) {
		if (ht->table[index] == key) {
			// We don't have to insert the key since it is already in the table.
			return true;
		}
		printf("collision!\n");
		index = (index + 1) & (ht->length - 1);
		if (index == start) return false; // We have wrapped around and we haven't found it.
	}
	// We have found an empty spot.
	ht->table[index] = key;
	return true;
}

// Here we could return a pointer to the uintptr_t found or a null pointer.
static bool HashTable_search(struct HashTable ht[static 1], uintptr_t key) {
	// Cormen et al. style.
	assert(HashTable_invariant(ht));
	assert(key);
	size_t i = 0, j = 0;
	do {
		j = HashTable_hash(ht, key, i);
		if (ht->table[j] == key) {
			return true;
		}
		i += 1;
		printf("collision!\n");
	} while (ht->table[j] && (i < ht->length));
	return false;
}
static bool HashTable_search2(struct HashTable ht[static 1], uintptr_t key) {
	// benhoyt.com style
	assert(HashTable_invariant(ht));
	assert(key);
	size_t index = hash(key) & (ht->length - 1);
	size_t start = index;
	while (ht->table[index]) {
		if (ht->table[index] == key) {
			return true;
		}
		index = (index + 1) & (ht->length - 1);
		printf("collision!\n");
		if (index == start) return false; // We have wrapped around and we haven't found it.
	}
	return false;
}

static void HashTable_reset(struct HashTable ht[static 1]) {
	assert(HashTable_invariant(ht));
	for (size_t i = 0; i < ht->length; i++) {
		ht->table[i] = 0;
	}
}

static int HashTable_print(struct HashTable ht[static 1]) {
	assert(HashTable_invariant(ht));
	int tot = 0;
	for (size_t i = 0; i < ht->length; i++) {
		int res = printf("%zu: 0x%" PRIXPTR "\n", i, ht->table[i]);
		if (res < 0) {
			return res;
		}
		// Lol what about overflow here? Does the real printf saturate?
		tot += res;
	}
	return tot;
}

#include <math.h>

int main(void) {
	struct HashTable *ht = &(struct HashTable){
		.table = (uintptr_t [8]){},
		.length = 8,
	};

	assert(HashTable_insert(ht, 1));
	assert(HashTable_insert(ht, 2));
	assert(HashTable_insert(ht, 3));
	assert(HashTable_insert(ht, 4));
	assert(HashTable_insert(ht, 5));
	assert(HashTable_insert(ht, 6));
	assert(HashTable_insert(ht, 7));
	assert(HashTable_insert(ht, 8));
	assert(HashTable_insert(ht, 8)); // Inserting the an element already present.
	assert(HashTable_search(ht, 1));
	assert(HashTable_search(ht, 2));
	assert(HashTable_search(ht, 3));
	assert(HashTable_search(ht, 4));
	assert(HashTable_search(ht, 5));
	assert(HashTable_search(ht, 6));
	assert(HashTable_search(ht, 7));
	assert(HashTable_search(ht, 8));
	assert(!HashTable_search(ht, 9));
	HashTable_print(ht);
	HashTable_reset(ht);
	assert(HashTable_insert(ht, 0xaaaa));
	assert(HashTable_insert(ht, 0xbbbb));
	assert(HashTable_insert(ht, 0xcccc));
	assert(HashTable_insert(ht, 0xdddd));
	assert(HashTable_insert(ht, 0xcafe));
	assert(HashTable_insert(ht, 0xbabe));
	assert(HashTable_insert(ht, 0xdead));
	assert(HashTable_insert(ht, 0xbeef));
	assert(HashTable_insert(ht, 0xbeef)); // Inserting the an element already present.
	assert(HashTable_search(ht, 0xaaaa));
	assert(HashTable_search(ht, 0xbbbb));
	assert(HashTable_search(ht, 0xcccc));
	assert(HashTable_search(ht, 0xdddd));
	assert(HashTable_search(ht, 0xcafe));
	assert(HashTable_search(ht, 0xbabe));
	assert(HashTable_search(ht, 0xdead));
	assert(HashTable_search(ht, 0xbeef));
	assert(!HashTable_search(ht, 0xeeee));
	HashTable_print(ht);
	HashTable_reset(ht);
	assert(HashTable_insert2(ht, 1));
	assert(HashTable_insert2(ht, 2));
	assert(HashTable_insert2(ht, 3));
	assert(HashTable_insert2(ht, 4));
	assert(HashTable_insert2(ht, 5));
	assert(HashTable_insert2(ht, 6));
	assert(HashTable_insert2(ht, 7));
	assert(HashTable_insert2(ht, 8));
	assert(HashTable_insert2(ht, 8)); // Inserting the an element already present.
	assert(HashTable_search2(ht, 1));
	assert(HashTable_search2(ht, 2));
	assert(HashTable_search2(ht, 3));
	assert(HashTable_search2(ht, 4));
	assert(HashTable_search2(ht, 5));
	assert(HashTable_search2(ht, 6));
	assert(HashTable_search2(ht, 7));
	assert(HashTable_search2(ht, 8));
	assert(!HashTable_search2(ht, 9));
	HashTable_print(ht);
	HashTable_reset(ht);
	assert(HashTable_insert2(ht, 0xaaaa));
	assert(HashTable_insert2(ht, 0xbbbb));
	assert(HashTable_insert2(ht, 0xcccc));
	assert(HashTable_insert2(ht, 0xdddd));
	assert(HashTable_insert2(ht, 0xcafe));
	assert(HashTable_insert2(ht, 0xbabe));
	assert(HashTable_insert2(ht, 0xdead));
	assert(HashTable_insert2(ht, 0xbeef));
	assert(HashTable_insert2(ht, 0xbeef)); // Inserting the an element already present.
	assert(HashTable_search2(ht, 0xaaaa));
	assert(HashTable_search2(ht, 0xbbbb));
	assert(HashTable_search2(ht, 0xcccc));
	assert(HashTable_search2(ht, 0xdddd));
	assert(HashTable_search2(ht, 0xcafe));
	assert(HashTable_search2(ht, 0xbabe));
	assert(HashTable_search2(ht, 0xdead));
	assert(HashTable_search2(ht, 0xbeef));
	assert(!HashTable_search2(ht, 0xeeee));
	HashTable_print(ht);
	HashTable_reset(ht);
	assert(HashTable_insert(ht, (uintptr_t)0x10FB91010));
	assert(HashTable_insert(ht, (uintptr_t)0x10FB91088));
	assert(HashTable_insert(ht, (uintptr_t)0x10FB91100));
	assert(HashTable_insert(ht, (uintptr_t)0x10FB91178));
	assert(HashTable_insert(ht, (uintptr_t)0x10FB911F0));
	assert(HashTable_insert(ht, (uintptr_t)0x10FB91268));
	assert(HashTable_insert(ht, (uintptr_t)0x10FB912E0));
	// To make search perform better we should leave some empty spots.
	assert(!HashTable_search(ht, (uintptr_t) 123));
	HashTable_print(ht);
	HashTable_reset(ht);
	assert(HashTable_insert2(ht, (uintptr_t)0x10FB91010));
	assert(HashTable_insert2(ht, (uintptr_t)0x10FB91088));
	assert(HashTable_insert2(ht, (uintptr_t)0x10FB91100));
	assert(HashTable_insert2(ht, (uintptr_t)0x10FB91178));
	assert(HashTable_insert2(ht, (uintptr_t)0x10FB911F0));
	assert(HashTable_insert2(ht, (uintptr_t)0x10FB91268));
	assert(HashTable_insert2(ht, (uintptr_t)0x10FB912E0));
	assert(!HashTable_search2(ht, (uintptr_t) 123));
	HashTable_print(ht);
	HashTable_reset(ht);
	return 0;
}
