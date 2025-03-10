// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/macros.h
// Copyrigth 2024 Mozilla Foundation.
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

#define MIN(X, Y) ((Y) > (X) ? (X) : (Y))
#define MAX(X, Y) ((Y) < (X) ? (X) : (Y))
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define ROUNDUP(X, K) (((X) + (K) - 1) & -(K))
#define ARRAYLEN(A) ((sizeof(A) / sizeof(*(A))) / ((unsigned)!(sizeof(A) % sizeof(*(A)))))
