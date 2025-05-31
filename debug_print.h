#ifndef DEBUG_PRINT_H
#define DEBUG_PRINT_H

#include <stdio.h>

// #define ENABLE_DEBUG_PRINT 1

#ifdef ENABLE_DEBUG_PRINT
#define DEBUG_PRINT() do { \
    FILE* f = fopen("/tmp/xdnn.log", "a"); \
    if (f) { fprintf(f, "[DEBUG] %s\n", __FUNCTION__); fclose(f); } \
} while(0)

#define DEBUG_PRINT_PARAMS(fmt, ...) do { \
    FILE* f = fopen("/tmp/xdnn.log", "a"); \
    if (f) { fprintf(f, "[DEBUG] %s: " fmt "\n", __FUNCTION__, __VA_ARGS__); fclose(f); } \
} while(0)

#else
#define DEBUG_PRINT() ((void)0)
#define DEBUG_PRINT_PARAMS(fmt, ...) ((void)0)
#endif

#endif // DEBUG_PRINT_H
