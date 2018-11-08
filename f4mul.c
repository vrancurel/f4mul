/* Implement F_4 (2^2^4+1) butterfly operations on 16 bits */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <immintrin.h>

//#define DEBUG

#define N_OPS 500000

typedef int32_t t_vecd[16];
typedef uint16_t t_vecw[16];
typedef uint16_t t_bitmap;

__m256i m0x0, m0x1, m0xffff;

uint64_t mul_basic_cycles = 0;
uint64_t mul_basic_usec = 0;
int mul_basic_ops = 0;

uint64_t mul_avx2_cycles = 0;
uint64_t mul_avx2_usec = 0;
int mul_avx2_ops = 0;

int n_mul_except = 0;
int n_mul_normal = 0;

#define STATS_START(Ident)                      \
  struct timeval Ident##_t1 = tick();           \
  uint64_t Ident##_start = hw_timer();          \
  
#define STATS_END(Ident)                        \
  uint64_t Ident##_end = hw_timer();            \
  uint64_t Ident##_t2 = hrtime_usec(Ident##_t1);        \
  Ident##_cycles += Ident##_end - Ident##_start;        \
  Ident##_usec += Ident##_t2;                           \
  Ident##_ops+=16;                                      \
  
#define STATS_PRINT(Ident)                            \
  printf("%s: cycles=%ld usec=%ld ops=%d\n",          \
         #Ident, Ident##_cycles, Ident##_usec, Ident##_ops);


void vecd_dump(t_vecd vecd);
void vecw_dump(t_vecw vecw);
void m256_dump(__m256i m);
void bitmap_dump(uint16_t b);

static inline struct timeval tick()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv;
}

static inline uint64_t hrtime_usec(struct timeval begin)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return 1000000 * (tv.tv_sec - begin.tv_sec) + tv.tv_usec - begin.tv_usec;
}

static inline uint64_t hw_timer()
{
  uint64_t x, lo, hi;
  
  __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
  x = (hi << 32) | lo;
  return x;
}    

// doubleword operation
void vecd_mul_mod(t_vecd dst, t_vecd vecd1, t_vecd vecd2) {
  int i;
  
  for (i = 0;i < 16;i++) {
    uint32_t c = (uint32_t) vecd1[i] * (uint32_t) vecd2[i];
    dst[i] = (c % 65537) & 65535;
  }
}

// doubleword operation
void vecd_add_mod(t_vecd dst, t_vecd vecd1, t_vecd vecd2) {
  int i;
  
  for (i = 0;i < 16;i++) {
    uint32_t c = (uint32_t) vecd1[i] + (uint32_t) vecd2[i];
    dst[i] = (c % 65537) & 65535;
  }
}

// doubleword operation
void vecd_sub_mod(t_vecd dst, t_vecd vecd1, t_vecd vecd2) {
  int i;
  
  for (i = 0;i < 16;i++) {
    uint32_t c = (uint32_t) vecd1[i] - (uint32_t) vecd2[i];
    dst[i] = (c % 65537) & 65535;
  }
}

void m256_mul_mod(t_vecw dst,  t_bitmap *bdstp,
                  __m256i m1, t_bitmap b1,
                  __m256i m2, t_bitmap b2) {
  int i;
  t_bitmap bdst = 0;
  __m256i _hi, _lo, _cmp1, _cmp2, _sub, _add1, _dst;
  int _z1, _z2;
  t_vecw hi, lo, vecw1, vecw2;
  
  _lo = _mm256_mullo_epi16(m1, m2);
  _hi = _mm256_mulhi_epu16(m1, m2);

  //check if inputs contains zero
  _cmp1 = _mm256_cmpeq_epi16(m0x0, m1);
  _cmp2 = _mm256_cmpeq_epi16(m0x0, m2);
  _z1 = !_mm256_testz_si256(_cmp1, _cmp1);
  _z2 = !_mm256_testz_si256(_cmp2, _cmp2);

  if (b1 || b2 || _z1 || _z2) {
    n_mul_except++;
    _mm256_storeu_si256((__m256i *) hi, _hi);
    _mm256_storeu_si256((__m256i *) lo, _lo);
    _mm256_storeu_si256((__m256i *) vecw1, m1);
    _mm256_storeu_si256((__m256i *) vecw2, m2);
    
    for (i = 0;i < 16;i++) {
      if (b1 & (1 << i)) {
        bdst |= (1 << i);
        dst[i] = 0;
      } else if (b2 & (1 << i)) {
        bdst |= (1 << i);
        dst[i] = 0;
      } else if (vecw1[i] == 0)
        dst[i] = -vecw2[i] + 1;
      else if (vecw2[i] == 0)
        dst[i] = -vecw1[i] + 1;
      else if (lo[i] > hi[i])
        dst[i] = lo[i] - hi[i];
      else
        dst[i] = lo[i] - hi[i] + 1;
    }
  } else {
    n_mul_normal++;

    _sub = _mm256_sub_epi16(_lo, _hi);

    //cmpge(_hi, _lo)
    _cmp1 = _mm256_cmpeq_epi16(_mm256_max_epu16(_hi, _lo), _hi);

    _dst = _mm256_sub_epi16(_sub, _cmp1);
    _mm256_storeu_si256((__m256i *) dst, _dst);
  }

  if (bdstp)
    *bdstp = bdst;
}

// simulate SIMD add_epu16
void vecw_add(t_vecw dst, t_vecw vecw1, t_vecw vecw2) {
  int i;

  for (i = 0;i < 16;i++) {
    uint32_t x = ((uint32_t) vecw1[i] + vecw2[i]) & 0xffff;
    dst[i] = x;
  }
}

// simulate SIMD adds_epu16
void vecw_adds(t_vecw dst, t_vecw vecw1, t_vecw vecw2) {
  int i;

  for (i = 0;i < 16;i++) {
    uint32_t x = (uint32_t) vecw1[i] + vecw2[i];
    if (x > 65535)
      dst[i] = 0xffff;
    else
      dst[i] = x;
  }
}

/*
 * perform the addition modulo F_4 on 16 bits 
 */
void vecw_add_mod(t_vecw dst,  t_bitmap *bdstp,
                  t_vecw vecw1, t_bitmap b1,
                  t_vecw vecw2, t_bitmap b2) {
  int i;
  t_bitmap bdst = 0;
  t_vecw rol, sat;

  vecw_add(rol, vecw1, vecw2);
  vecw_adds(sat, vecw1, vecw2);
  
  for (i = 0;i < 16;i++) {
    if (b1 & (1 << i)) {
      dst[i] = vecw2[i];
    } else if (b2 & (1 << i)) {
      dst[i] = vecw1[i];
    } else if (vecw1[i] == 0) {
      dst[i] = vecw2[i] - 1;
    } else if (vecw2[i] == 0) {
      dst[i] = vecw1[i] - 1;
    } else if (sat[i] == 0xffff) {
      if (rol[i] == 0)
        dst[i] = rol[i];
      else if (rol[i] == 0xffff)
        dst[i] = rol[i];
      else
        dst[i] = rol[i] - 1;
    } else {
      dst[i] = sat[i];
    }
  }
  
  if (bdstp)
    *bdstp = bdst;
}

// simulate SIMD sub_epu16
void vecw_sub(t_vecw dst, t_vecw vecw1, t_vecw vecw2) {
  int i;

  for (i = 0;i < 16;i++) {
    uint32_t x = ((uint32_t) vecw1[i] - vecw2[i]) & 0xffff;
    dst[i] = x;
  }
}

// simulate SIMD subs_epu16
void vecw_subs(t_vecw dst, t_vecw vecw1, t_vecw vecw2) {
  int i;

  for (i = 0;i < 16;i++) {
    uint32_t x = (uint32_t) vecw1[i] - vecw2[i];
    if (x > 65535)
      dst[i] = 0xffff;
    else
      dst[i] = x;
  }
}

/*
 * perform the substraction modulo F_4 on 16 bits 
 */
void vecw_sub_mod(t_vecw dst,  t_bitmap *bdstp,
                  t_vecw vecw1, t_bitmap b1,
                  t_vecw vecw2, t_bitmap b2) {
  int i;
  t_bitmap bdst = 0;
  t_vecw rol, sat;

  vecw_sub(rol, vecw1, vecw2);
  vecw_subs(sat, vecw1, vecw2);
  
  for (i = 0;i < 16;i++) {
    if (b1 & (1 << i)) {
      dst[i] = 65535 - vecw2[i] + 3;
    } else if (b2 & (1 << i)) {
      dst[i] = vecw1[i];
    } else if (vecw1[i] == 0) {
      dst[i] = 65535 - vecw2[i] + 1;
    } else if (vecw2[i] == 0) {
      dst[i] = vecw1[i] + 2;
    } else if (sat[i] == 0xffff) {
      if (rol[i] == 0xffff)
        dst[i] = 0;
      else if (rol[i] == 0xffff)
        dst[i] = rol[i];
      else
        dst[i] = rol[i] + 2;
    } else {
      dst[i] = sat[i];
    }
  }
  
  if (bdstp)
    *bdstp = bdst;
}

// returns 1 if equal, else 0
int vecd_equal(t_vecd vecd1, t_vecd vecd2) {
  int i;

  for (i = 0;i < 16;i++) {
    if (vecd1[i] != vecd2[i])
      return 0;
  }

  return 1;
}

// generate random numbers in F_4
void vecd_rnd(t_vecd vecd) {
  int i;

  for (i = 0;i < 16;i++) {
    vecd[i] = rand() % 65537;
  }
}

// convert a F_4 doubleword into a word and a bitmap
void vecd_to_vecw(t_vecd vecd, t_vecw vecw, t_bitmap *bp) {
  int i;
  t_bitmap b = 0;
  
  for (i = 0;i < 16;i++) {
    if (vecd[i] == 0) {
      b |= (1 << i);
    } else if (vecd[i] == 65536) {
      vecw[i] = 0;
    } else if (vecd[i] < 65536) {
      vecw[i] = vecd[i];
    } else {
      assert(0);
    }
  }

  if (bp)
    *bp = b;
}

// convert a word and a bitmap into a F_4 doubleword
void vecw_to_vecd(t_vecw vecw, t_bitmap b, t_vecd vecd) {
  int i;
  
  for (i = 0;i < 16;i++) {
    if (b & (1 << i))
      vecd[i] = 0;
    else
      vecd[i] = vecw[i];
  }
}

// convert a F_4 doubleword into a word and a bitmap
void vecd_to_m256(t_vecd vecd, __m256i *mp, t_bitmap *bp) {
  int i;
  t_bitmap b = 0;
  t_vecw vecw;
  
  for (i = 0;i < 16;i++) {
    if (vecd[i] == 0) {
      b |= (1 << i);
    } else if (vecd[i] == 65536) {
      vecw[i] = 0;
    } else if (vecd[i] < 65536) {
      vecw[i] = vecd[i];
    } else {
      assert(0);
    }
  }

  if (mp)
    *mp = _mm256_loadu_si256((__m256i *) vecw);

  if (bp)
    *bp = b;
}

// convert a word and a bitmap into a F_4 doubleword
void m256_to_vecd(__m256i m, t_bitmap b, t_vecd vecd) {
  int i;
  
  for (i = 0;i < 16;i++) {
    if (b & (1 << i))
      vecd[i] = 0;
    else
      vecd[i] = m[i];
  }
}

// test multiplication
void test_mul() {
  t_vecd vecd1, vecd2, vdst, vrdst;
  __m256i m1, m2;
  t_vecw rdst;
  t_bitmap b1, b2, bdst;

  vecd_rnd(vecd1);
  vecd_rnd(vecd2);
#ifdef DEBUG
  vecd_dump(vecd1);
  printf("*\n");
  vecd_dump(vecd2);
#endif
  vecd_to_m256(vecd1, &m1, &b1);
  vecd_to_m256(vecd2, &m2, &b2);
  STATS_START(mul_basic);
  vecd_mul_mod(vdst, vecd1, vecd2);
  STATS_END(mul_basic);
  STATS_START(mul_avx2);
  m256_mul_mod(rdst, &bdst, m1, b1, m2, b2);
  STATS_END(mul_avx2);
#ifdef DEBUG
  printf("=\n");
  vecd_dump(vdst);
  vecw_dump(rdst);
#endif
  vecw_to_vecd(rdst, bdst, vrdst);
  if (!vecd_equal(vdst, vrdst)) {
    exit(1);
  }
}

// test addition
void test_add() {
  t_vecd vecd1, vecd2, vdst, vrdst;
  t_vecw vecw1, vecw2, rdst;
  t_bitmap b1, b2, bdst;

  vecd_rnd(vecd1);
  vecd_rnd(vecd2);
#ifdef DEBUG
  vecd_dump(vecd1);
  printf("+\n");
  vecd_dump(vecd2);
#endif
  vecd_to_vecw(vecd1, vecw1, &b1);
  vecd_to_vecw(vecd2, vecw2, &b2);
  vecd_add_mod(vdst, vecd1, vecd2);
  vecw_add_mod(rdst, &bdst, vecw1, b1, vecw2, b2);
#ifdef DEBUG
  printf("=\n");
  vecd_dump(vdst);
  vecw_dump(rdst);
#endif
  vecw_to_vecd(rdst, bdst, vrdst);
  if (!vecd_equal(vdst, vrdst)) {
    exit(1);
  }
}

// test substraction
void test_sub() {
  t_vecd vecd1, vecd2, vdst, vrdst;
  t_vecw vecw1, vecw2, rdst;
  t_bitmap b1, b2, bdst;

  vecd_rnd(vecd1);
  vecd_rnd(vecd2);
#ifdef DEBUG
  vecd_dump(vecd1);
  printf("-\n");
  vecd_dump(vecd2);
#endif
  vecd_to_vecw(vecd1, vecw1, &b1);
  vecd_to_vecw(vecd2, vecw2, &b2);
  vecd_sub_mod(vdst, vecd1, vecd2);
  vecw_sub_mod(rdst, &bdst, vecw1, b1, vecw2, b2);
#ifdef DEBUG
  printf("=\n");
  vecd_dump(vdst);
  vecw_dump(rdst);
#endif
  vecw_to_vecd(rdst, bdst, vrdst);
  if (!vecd_equal(vdst, vrdst)) {
    exit(1);
  }
}

// dump a vector of doublewords
void vecd_dump(t_vecd vecd) {
  int i;
  
  for (i = 0;i < 16;i++) {
    printf("%6d ", vecd[i]);
  }
  printf(" (D)\n");
}

// dump a vector of words
void vecw_dump(t_vecw vecw) {
  int i;
  
  for (i = 0;i < 16;i++) {
    printf("%6d ", vecw[i]);
  }
  printf(" (W)\n");
}

// dump a vector of words
void m256_dump(__m256i m) {
  int i;
  t_vecw vecw;
  
  _mm256_storeu_si256((__m256i *) vecw, m);
  
  for (i = 0;i < 16;i++) {
    printf("%6d ", vecw[i]);
  }
  printf(" (m256)\n");
}

// dump a bitmap
void bitmap_dump(uint16_t b)
{
  int i;
  
  for (i = 0;i < 16;i++) {
    if (b & (1 << i))
      printf("1");
    else
      printf("0");
  }
  printf("\n");
}

int main() {
  int i;

  m0x0 = _mm256_set1_epi16(0);
  m0x1 = _mm256_set1_epi16(1);
  m0xffff = _mm256_set1_epi16(0xffff);
  
  srand(0);
  for (i = 0;i < N_OPS;i++) {
    test_mul();
#ifdef DEBUG
    printf("............\n");
#endif
    test_add();
#ifdef DEBUG
    printf("............\n");
#endif
    test_sub();
#ifdef DEBUG
    printf("............\n");
#endif
  }
  STATS_PRINT(mul_basic);
  STATS_PRINT(mul_avx2);
  printf("n_mul_except=%d n_mul_normal=%d\n", n_mul_except, n_mul_normal);
}
