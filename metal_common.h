#ifndef metal_common_h
#define metal_common_h

// From Hacker's Delight
// https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
#define CEIL_DIV(n, d) ( ( (n) + (d) + ((d)>0?-1:+1) ) / (d) )

#define COARSE_FACTOR 4

#endif /* metal_common_h */
