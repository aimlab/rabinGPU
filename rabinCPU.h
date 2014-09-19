/*
 * Copyright (C) 2009 Tudor Marian (tudorm@cs.cornell.edu)
 * 		- ported and modified (kernel ready) the original C++ implementation
 *
 * Copyright (C) 2004 Hyang-Ah Kim (hakim@cs.cmu.edu)
 * Copyright (C) 1999 David Mazieres (dm@uun.org)
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
 * USA
 *
 */

#ifndef RABINPOLY_H
#define RABINPOLY_H

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/highmem.h>
#include <linux/errno.h>
#define default_alloc(bytes)	kmalloc(bytes, GFP_KERNEL)
#define default_free(buf)		kfree(buf)
#else
#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#define default_alloc(bytes)	malloc(bytes)
#define default_free(buf)		free(buf)
#endif //__KERNEL

/* refer to the original rabin fingerprint paper */
#define FINGERPRINT_PT 0xbfe6b8a5bf378d83LL

/* this structure is in fact opaque */
struct rabinpoly {
    int shift;
    u_int64_t T[256]; /* lookup table */
    u_int64_t poly;
};

struct rabinpoly_window {
    int size, bufpos;
    u_int64_t fingerprint;
    u_int64_t U[256];
    u_char *buf;
    struct rabinpoly rp;
};

static inline int get_rabinpoly_window_size(struct rabinpoly_window *rw)
{
    return rw->size;
}

/* size is the size of the window */
int rabinpoly_init(struct rabinpoly_window *w, u_int64_t poly, int size);
/* main functon */
u_int64_t rabinpoly_slide8(struct rabinpoly_window *w, u_char m);

/* can be used to lightweight reset the rabinpoly_window */
static void rabinpoly_reset(struct rabinpoly_window *w)
{
    if (!w)
        return;

    w->bufpos = -1;
    w->fingerprint = 0;
    memset(w->buf, 0, w->size);
}

/* deallocates resources */
static inline void rabinpoly_deinit(struct rabinpoly_window *w)
{
    if (!w)
        return;
    default_free(w->buf);
    memset(w, 0, sizeof(struct rabinpoly_window));
}

#endif // RABINPOLY_H


