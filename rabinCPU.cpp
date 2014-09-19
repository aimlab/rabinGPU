 /*
  * Copyright (C) 2008 Tudor Marian (tudorm@cs.cornell.edu)
  */

#include "rabinCPU.h"

#ifndef __KERNEL__ /* these routines are already defined in bitops.h */
/* Highest bit set in a byte */
static const char bytemsb[0x100] = 
{ 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
  4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
  6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8,
};

/* Least set bit (ffs) */
static const char bytelsb[0x100] = 
{ 0, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1,
  2, 1, 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 6, 1, 2, 1, 3, 1,
  2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1,
  2, 1, 7, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5, 1, 2, 1, 3, 1,
  2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 6, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1,
  2, 1, 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 8, 1, 2, 1, 3, 1,
  2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1,
  2, 1, 6, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5, 1, 2, 1, 3, 1,
  2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 7, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1,
  2, 1, 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 6, 1, 2, 1, 3, 1,
  2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1,
  2, 1,
};

/* Find last set (most significant bit) */
static inline u_int fls32(u_int32_t v)
{
    if (v & 0xffff0000) {
        if (v & 0xff000000)
            return 24 + bytemsb[v >> 24];
        else
            return 16 + bytemsb[v >> 16];
    }
    if (v & 0x0000ff00)
        return 8 + bytemsb[v >> 8];
    else
        return bytemsb[v];
}

static inline char fls64(u_int64_t v)
{
    u_int32_t h;
    if ((h = v >> 32))
        return 32 + fls32(h);
    else
        return fls32((u_int32_t) v);
}


/*
 * For symmetry, a 64-bit find first set, "ffs," that finds the least
 * significant 1 bit in a word.
 */
static inline u_int ffs32(u_int32_t v)
{
    int vv;
    if (v & 0xffff) {
        if ((vv = v & 0xff))
            return bytelsb[vv];
        else
            return 8 + bytelsb[v >> 8 & 0xff];
    } else {
        if ((vv = v & 0xff0000))
            return 16 + bytelsb[vv >> 16];
        else if (v)
            return 24 + bytelsb[v >> 24 & 0xff];
        else
            return 0;
    }
}

static inline u_int ffs64(u_int64_t v)
{
    u_int32_t l;
    if ((l = v & 0xffffffff))
        return fls32(l);
    else if ((l = v >> 32))
        return 32 + fls32(l);
    else
        return 0;
}
#define ffs(v) (sizeof (v) > 4 ? ffs64 (v) : ffs32 (v))

#endif // ! __KERNEL__

#define INT64(n) n##LL
#define MSB64 INT64(0x8000000000000000)

static u_int64_t polymod(u_int64_t nh, u_int64_t nl, u_int64_t d)
{
    int i, k = fls64(d) - 1;
    d <<= 63 - k;

    if (nh) {
        if (nh & MSB64)
            nh ^= d;
        for (i = 62; i >= 0; i--)
            if (nh & ((u_int64_t) 1) << i) {
                nh ^= d >> (63 - i);
                nl ^= d << (i + 1);
            }
    }
    for (i = 63; i >= k; i--) {
        if (nl & INT64 (1) << i)
            nl ^= d >> (63 - i);
    }

    return nl;
}

static u_int64_t polygcd(u_int64_t x, u_int64_t y)
{
    for (;;) {
        if (!y)
            return x;
        x = polymod(0, x, y);
        if (!x)
            return y;
        y = polymod(0, y, x);
    }
}

static void polymult(u_int64_t *php, u_int64_t *plp, u_int64_t x, u_int64_t y)
{
    int i;
    u_int64_t ph = 0, pl = 0;

    if (x & 1)
        pl = y;
    for (i = 1; i < 64; i++)
        if (x & (INT64 (1) << i)) {
            ph ^= y >> (64 - i);
            pl ^= y << i;
        }
    if (php)
        *php = ph;
    if (plp)
        *plp = pl;
}

static u_int64_t polymmult(u_int64_t x, u_int64_t y, u_int64_t d)
{
    u_int64_t h, l;
    polymult(&h, &l, x, y);
    return polymod(h, l, d);
}

int polyirreducible(u_int64_t f)
{
    u_int64_t u = 2;
    int i, m = (fls64(f) - 1) >> 1;

    for (i = 0; i < m; i++) {
        u = polymmult(u, u, f);
        if (polygcd(f, u ^ 2) != 1)
            return 0;
    }
    return 1;
}

static void __rabinpoly_init(struct rabinpoly *rp, u_int64_t poly)
{
    int j, xshift;
    u_int64_t T1;

    if (!rp)
        return;

    xshift = fls64(poly) - 1;
    T1 = polymod(0, INT64 (1) << xshift, poly);

    rp->poly = poly;
    rp->shift = xshift - 8;

    for (j = 0; j < 256; j++)
        rp->T[j] = polymmult(j, T1, poly) | ((u_int64_t) j << xshift);
}

static inline u_int64_t append8(struct rabinpoly *rp, u_int64_t p, u_char m)
{
    return ((p << 8) | m) ^ rp->T[p >> rp->shift];
}

int rabinpoly_init(struct rabinpoly_window *w, u_int64_t poly, int size)
{
    int i;
    u_int64_t sizeshift = 1;

    memset(w, 0, sizeof(struct rabinpoly_window));
    w->size = size;
    if (!(w->buf = (u_char *) default_alloc(size * sizeof(u_char))))
        return ENOMEM;
    rabinpoly_reset(w);

    __rabinpoly_init(&w->rp, poly);

    for (i = 1; i < size; i++)
        sizeshift = append8(&w->rp, sizeshift, 0);

    for (i = 0; i < 256; i++)
        w->U[i] = polymmult(i, sizeshift, poly);

    return 0;
}

u_int64_t rabinpoly_slide8(struct rabinpoly_window *w, u_char m)
{
    u_char om;
    if (++w->bufpos >= w->size)
        w->bufpos = 0;

    om = w->buf[w->bufpos];
    w->buf[w->bufpos] = m;

    return w->fingerprint = append8(&w->rp, w->fingerprint ^ w->U[om], m);
}
