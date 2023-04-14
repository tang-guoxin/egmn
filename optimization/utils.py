# -*- coding: utf-8 -*-

import numpy as np
import random


def float2bin(num, max_flt=64):
    s = []
    for i in range(max_flt):
        num *= 2
        b_ = int(num)
        s.append(str(b_))
        num -= b_
    return ''.join(s)


def bin2float(bin_flt):
    res = 0
    for i in range(len(bin_flt)):
        res += int(bin_flt[i]) * (1/2) ** (i+1)
    return res


def dec2bin(num, max_int=10, max_flt=64):
    sym = 1 if num > 0 else 0
    num = abs(num)
    lnum, rnum = int(num), num - int(num)
    bin_int = np.binary_repr(lnum, max_int)
    bin_flt = float2bin(rnum, max_flt)
    return str(sym) + bin_int + '.' + bin_flt
  

def bin2dec(num):
    sym = 1 if num[0] == '1' else -1
    num = num[1:]
    s = num.split('.')
    ten = int(s[0], 2)
    flt = bin2float(s[1])
    return sym * (ten + flt)


def roulette(fval, perc):
    res = list()
    max_val = np.max(fval)
    fval = fval - max_val
    sz = fval.shape[0]
    m = int(sz * perc)
    s_ = np.sum(fval)
    for i in range(m):
        r = np.random.random()
        point = r * s_
        count = 0
        for j in range(sz):
            if j in res:
                continue
            count += fval[j]
            if count <= point:
                res.append(j)
                s_ -= fval[j]
                break
    return res


def mat2bin(x, max_int, max_flt):
    res = list()
    [m, n] = x.shape
    for i in range(m):
        tmp = list()
        for j in range(n):
            tmp.append(dec2bin(x[i, j], max_int, max_flt))
        res.append(tmp)
    return res


def bin2mat(bx):
    m, n = len(bx), len(bx[0])
    res = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            res[i, j] = bin2dec(bx[i][j])
    return res


# 单个交叉
def cross_str(str1, str2):
    lx = len(str1)
    ls = [i for i in range(lx)]
    idx = random.sample(ls, int(lx*0.5))
    str1_ = [s for s in str1]
    str2_ = [s for s in str2]
    for i in idx:
        str1_[i] = str2[i]
        str2_[i] = str1[i]
    return ''.join(str1_), ''.join(str2_)


def cross(evx2bin, num):
    m, n = len(evx2bin), len(evx2bin[0])
    res = list()
    for i in range(m, num):
        r1, r2 = random.sample(evx2bin, 2)
        newx = list()
        for j in range(n):
            x1, x2 = cross_str(r1[j], r2[j])
            newx.append(x1)
        res.append(newx)
    return res + evx2bin
    

def variation_str(str_):
    n = len(str_)
    str1 = [s for s in str_]
    idx = random.randint(0, n-1)
    while str1[idx] == '.':
        idx = random.randint(0, n-1)
    str1[idx] = '0' if str1[idx] == '1' else '1'
    return ''.join(str1)


def variation(crsx, perc):
    lc = len(crsx)
    vnum = int(lc * perc)
    idx = random.sample([i for i in range(lc)], vnum)
    for i in idx:
        newx = list()
        for x in crsx[i]:
            newx.append(variation_str(x))
        crsx[i] = newx
    return crsx


if __name__ == '__main__':
    print(cross_str('123456', 'abcdef'))








