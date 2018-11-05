/* Implement F_4 (2^2^4+1) butterfly operations on 16 bits */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

typedef int32_t t_vecd[16];
typedef uint16_t t_vecw[16];
typedef uint16_t t_bitmap;

void vecd_dump(t_vecd vecd);
void vecw_dump(t_vecw vecw);
void bitmap_dump(uint16_t b);

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

// simulate SIMD mulhi_epu16
void vecw_mulhi(t_vecw dst, t_vecw vecw1, t_vecw vecw2) {
  int i;

  for (i = 0;i < 16;i++) {
    uint32_t x = ((uint32_t) vecw1[i] * vecw2[i]) >> 16;
    dst[i] = x;
  }
}

// simulate SIMD mullo_epi16
void vecw_mullo(t_vecw dst, t_vecw vecw1, t_vecw vecw2) {
  int i;

  for (i = 0;i < 16;i++) {
    uint32_t x = ((uint32_t) vecw1[i] * vecw2[i]) & 0xffff;
    dst[i] = x;
  }
}

/*
 * perform the multiplication modulo F_4 on 16 bits 
 *
 * we utilize the fact, that (N-1) % N = -1.
 * thus, (65536 * a) % 65537 == -a % 65537.
 * also, -a % 65537 == -a + 1 (mod 65536), when 0 is interpreted as 65536
*/
void vecw_mul_mod(t_vecw dst,  t_bitmap *bdstp,
                  t_vecw vecw1, t_bitmap b1,
                  t_vecw vecw2, t_bitmap b2) {
  int i;
  t_bitmap bdst = 0;
  t_vecw hi, lo;
  
  vecw_mullo(lo, vecw1, vecw2);
  vecw_mulhi(hi, vecw1, vecw2);

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

// test multiplication
void test_mul() {
  t_vecd vecd1, vecd2, vdst, vrdst;
  t_vecw vecw1, vecw2, rdst;
  t_bitmap b1, b2, bdst;

  vecd_rnd(vecd1);
  vecd_rnd(vecd2);
  vecd_dump(vecd1);
  printf("*\n");
  vecd_dump(vecd2);
  vecd_to_vecw(vecd1, vecw1, &b1);
  vecd_to_vecw(vecd2, vecw2, &b2);
  vecd_mul_mod(vdst, vecd1, vecd2);
  vecw_mul_mod(rdst, &bdst, vecw1, b1, vecw2, b2);
  printf("=\n");
  vecd_dump(vdst);
  vecw_dump(rdst);
  vecw_to_vecd(rdst, bdst, vrdst);
  if (!vecd_equal(vdst, vrdst)) {
    exit(1);
  }
  printf("............\n");
}

// test addition
void test_add() {
  t_vecd vecd1, vecd2, vdst, vrdst;
  t_vecw vecw1, vecw2, rdst;
  t_bitmap b1, b2, bdst;

  vecd_rnd(vecd1);
  vecd_rnd(vecd2);
  vecd_dump(vecd1);
  printf("+\n");
  vecd_dump(vecd2);
  vecd_to_vecw(vecd1, vecw1, &b1);
  vecd_to_vecw(vecd2, vecw2, &b2);
  vecd_add_mod(vdst, vecd1, vecd2);
  vecw_add_mod(rdst, &bdst, vecw1, b1, vecw2, b2);
  printf("=\n");
  vecd_dump(vdst);
  vecw_dump(rdst);
  vecw_to_vecd(rdst, bdst, vrdst);
  if (!vecd_equal(vdst, vrdst)) {
    exit(1);
  }
  printf("............\n");
}

// test substraction
void test_sub() {
  t_vecd vecd1, vecd2, vdst, vrdst;
  t_vecw vecw1, vecw2, rdst;
  t_bitmap b1, b2, bdst;

  vecd_rnd(vecd1);
  vecd_rnd(vecd2);
  vecd_dump(vecd1);
  printf("-\n");
  vecd_dump(vecd2);
  vecd_to_vecw(vecd1, vecw1, &b1);
  vecd_to_vecw(vecd2, vecw2, &b2);
  vecd_sub_mod(vdst, vecd1, vecd2);
  vecw_sub_mod(rdst, &bdst, vecw1, b1, vecw2, b2);
  printf("=\n");
  vecd_dump(vdst);
  vecw_dump(rdst);
  vecw_to_vecd(rdst, bdst, vrdst);
  if (!vecd_equal(vdst, vrdst)) {
    exit(1);
  }
  printf("............\n");
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
  srand(0);
  while (1) {
    test_mul();
    test_add();
    test_sub();
  }
}
