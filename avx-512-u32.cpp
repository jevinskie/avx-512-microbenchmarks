#include <benchmark/benchmark.h>
#include <gflags/gflags.h>

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#undef NDEBUG
#include <cassert>
#define NDEBUG

#define ALIGNMENT 4096

typedef uint32_t vec_elem_t;

constexpr size_t vec_elem_size_bytes = sizeof(vec_elem_t);

typedef vec_elem_t vN_elem_t __attribute__((vector_size(512 / 8)));
typedef vN_elem_t *aligned_vN_elem_ptr __attribute__((align_value(ALIGNMENT)));
typedef vN_elem_t *const const_aligned_vN_elem_ptr __attribute__((align_value(ALIGNMENT)));

typedef vec_elem_t *aligned_elem_ptr __attribute__((align_value(ALIGNMENT)));
typedef vec_elem_t *const const_aligned_elem_ptr __attribute__((align_value(ALIGNMENT)));

static_assert(ALIGNMENT % sizeof(vN_elem_t) == 0);

constexpr uint32_t vec_type_num_elem = sizeof(vN_elem_t) / sizeof(vec_elem_t);
// constexpr size_t vec_size_bytes      = 1024 * 1024 * 64;
constexpr size_t vec_size_bytes = 1024 * 12;
static_assert(vec_size_bytes % ALIGNMENT == 0);
constexpr uint32_t vec_num_elem_max = UINT32_MAX / 2;
constexpr uint32_t vec_num_elem     = vec_size_bytes / sizeof(vec_elem_t);
static_assert(vec_num_elem < vec_num_elem_max);

volatile vec_elem_t first_sum;
volatile vec_elem_t last_sum;

DEFINE_int32(rand_seed, -1, "Random seed (use < 0 for non-deterministic)");
static std::random_device rand_dev;
static std::mt19937_64 rand_gen(FLAGS_rand_seed < 0 ? rand_dev() : FLAGS_rand_seed);

static void sum(const const_aligned_elem_ptr __restrict__ a,
                const const_aligned_elem_ptr __restrict__ b, aligned_elem_ptr __restrict__ o,
                uint32_t n) {
    if ((n + 1) * vec_type_num_elem >= vec_num_elem_max) {
        __builtin_unreachable();
    }
    if ((n * vec_type_num_elem * vec_elem_size_bytes) % ALIGNMENT) {
        __builtin_unreachable();
    }
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < vec_type_num_elem; ++j) {
            const auto off = i + j;
            o[off]         = a[off] + b[off];
        }
    }
}

static void sum_vec(const const_aligned_vN_elem_ptr __restrict__ a,
                    const const_aligned_vN_elem_ptr __restrict__ b,
                    aligned_vN_elem_ptr __restrict__ o, uint32_t n) {
    if ((n + 1) * vec_type_num_elem >= vec_num_elem_max) {
        __builtin_unreachable();
    }
    if ((n * vec_type_num_elem * vec_elem_size_bytes) % ALIGNMENT) {
        __builtin_unreachable();
    }
    for (uint32_t i = 0; i < n / vec_type_num_elem; ++i) {
        o[i] = a[i] + b[i];
    }
}

static vN_elem_t sum_single_vec(const vN_elem_t a, const vN_elem_t b) {
    return a + b;
}

static void sum_vec_helper(const const_aligned_vN_elem_ptr __restrict__ a,
                           const const_aligned_vN_elem_ptr __restrict__ b,
                           aligned_vN_elem_ptr __restrict__ o, uint32_t n) {
    if ((n + 1) * vec_type_num_elem >= vec_num_elem_max) {
        __builtin_unreachable();
    }
    if ((n * vec_type_num_elem * vec_elem_size_bytes) % ALIGNMENT) {
        __builtin_unreachable();
    }
    for (uint32_t i = 0; i < n / vec_type_num_elem; ++i) {
        o[i] = sum_single_vec(a[i], b[i]);
    }
}

static uint8_t *alloc_vec(size_t sz) {
    assert(sz % ALIGNMENT == 0);
    uint8_t *res         = nullptr;
    const auto alloc_res = posix_memalign((void **)&res, ALIGNMENT, sz);
    if (alloc_res) {
        const auto cerrno = errno;
        fprintf(stderr, "posix_memalign of %zu bytes returned errno %d aka '%s'\n", vec_size_bytes,
                cerrno, strerror(cerrno));
        assert(false && "alloc_vec failed");
    }
    return res;
}

static void fill_rand(uint8_t *buf, size_t sz) {
    assert(sz % sizeof(uint64_t) == 0);
    assert((uintptr_t)buf % sizeof(uint64_t) == 0);
    auto buf64 = (uint64_t *)buf;
    std::uniform_int_distribution<uint64_t> dist(0);
    for (size_t i = 0; i < sz / sizeof(uint64_t); ++i) {
        buf64[i] = dist(rand_gen);
    }
}

static uint8_t *get_rand_buf(size_t sz) {
    auto buf = alloc_vec(sz);
    fill_rand(buf, sz);
    return buf;
}

static uint8_t *get_zero_buf(size_t sz) {
    auto buf = alloc_vec(sz);
    memset(buf, 0, sz);
    return buf;
}

static void BM_sum(benchmark::State &state) {
    auto a = (const const_aligned_elem_ptr)get_rand_buf(vec_size_bytes);
    auto b = (const const_aligned_elem_ptr)get_rand_buf(vec_size_bytes);
    auto o = (const aligned_elem_ptr)get_zero_buf(vec_size_bytes);
    for (auto _ : state) {
        sum(a, b, o, vec_num_elem);
        first_sum = o[0];
        last_sum  = o[vec_num_elem - 1];
        benchmark::DoNotOptimize(first_sum);
        benchmark::DoNotOptimize(last_sum);
    }
    free(a);
    free(b);
    free(o);
}

BENCHMARK(BM_sum);

static void BM_sum_vec(benchmark::State &state) {
    auto a = (const const_aligned_vN_elem_ptr)get_rand_buf(vec_size_bytes);
    auto b = (const const_aligned_vN_elem_ptr)get_rand_buf(vec_size_bytes);
    auto o = (const aligned_vN_elem_ptr)get_zero_buf(vec_size_bytes);
    for (auto _ : state) {
        sum_vec(a, b, o, vec_num_elem);
        first_sum = o[0][0];
        last_sum  = o[(vec_num_elem / vec_type_num_elem) - 1][0];
        benchmark::DoNotOptimize(first_sum);
        benchmark::DoNotOptimize(last_sum);
    }
    free(a);
    free(b);
    free(o);
}

BENCHMARK(BM_sum_vec);

static void BM_sum_vec_helper(benchmark::State &state) {
    auto a = (const const_aligned_vN_elem_ptr)get_rand_buf(vec_size_bytes);
    auto b = (const const_aligned_vN_elem_ptr)get_rand_buf(vec_size_bytes);
    auto o = (const aligned_vN_elem_ptr)get_zero_buf(vec_size_bytes);
    for (auto _ : state) {
        sum_vec_helper(a, b, o, vec_num_elem);
        first_sum = o[0][0];
        last_sum  = o[(vec_num_elem / vec_type_num_elem) - 1][0];
        benchmark::DoNotOptimize(first_sum);
        benchmark::DoNotOptimize(last_sum);
    }
    free(a);
    free(b);
    free(o);
}

BENCHMARK(BM_sum_vec_helper);

BENCHMARK_MAIN();
