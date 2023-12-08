



################ STANDARD ################

import os
import sys
import stat
import traceback
import gc
import io
import shutil
import subprocess
import struct
import sqlite3
import logging
import warnings
import pathlib
import importlib
import tempfile
import inspect
import math
import random
import re
import copy
import threading
import multiprocessing
import urllib
import base64
import zlib
import functools
import time
import contextlib
import typing
import argparse
from argparse import Namespace
from pprint import pprint
from collections import OrderedDict, defaultdict, Counter
from collections.abc import Iterable
from datetime import datetime

try:
    import requests
except:
    pass
try:
    import psutil
except:
    pass
try:
    from addict import Dict
except:
    Dict = dict
try:
    import pyunpack
    import zipfile
    import tarfile
except:
    pass


################ SCIENTIFIC ################

try:
    import numpy as np
except:
    pass
try:
    import matplotlib.pyplot as plt
except:
    pass
try:
    import pandas as pd
except:
    pass
try:
    import scipy
    import scipy.spatial
except:
    pass
try:
    import sklearn
    from sklearn import cluster as _
    from sklearn import decomposition as _
    from sklearn import metrics as _
    from sklearn import neighbors as _
except:
    pass
try:
    import PIL
    from PIL import Image
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except:
    pass
try:
    import cv2
except:
    pass

@contextlib.contextmanager
def np_seed(seed):
    _np_seed_lock.acquire()
    state = np.random.get_state()
    if seed!=None and not isinstance(seed, int):
        seed = zlib.adler32(bytes(seed, encoding='utf-8'))
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
        _np_seed_lock.release()
_np_seed_lock = threading.Lock()

def np_seed_all(seed):
    _np_seed_lock.acquire()
    if seed!=None and not isinstance(seed, int):
        seed = zlib.adler32(bytes(seed, encoding='utf-8'))
    np.random.seed(seed)
    _np_seed_lock.release()
    return

def np_choice_strat(length, size, shuffle=True):
    ans = np.concatenate([
        np.tile(np.arange(length), size//length),
        np.random.choice(length, size=size%length, replace=False),
    ])
    if shuffle: np.random.shuffle(ans)
    return ans


################ FILES ################

def mkfile(fn, parents=True, exist_ok=True):
    dn = '/'.join(fn.split('/')[:-1])
    mkdir(dn, parents=parents, exist_ok=exist_ok)
    return fn
def mkdir(dn, parents=True, exist_ok=True):
    pathlib.Path(dn).mkdir(parents=parents, exist_ok=exist_ok)
    return dn if (not dn[-1]=='/' or dn=='/') else dn[:-1]
def fnstrip(fn, return_more=False):
    dspl = fn.split('/')
    dn = '/'.join(dspl[:-1]) if len(dspl)>1 else '.'
    fn = dspl[-1]
    fspl = fn.split('.')
    if len(fspl)==1:
        bn = fspl[0]
        ext = ''
    else:
        bn = '.'.join(fspl[:-1])
        ext = fspl[-1]
    if return_more:
        return Namespace(
            dn=dn,
            fn=fn,
            path=f'{dn}/{fn}',
            bn_path=f'{dn}/{bn}',
            bn=bn,
            ext=ext,
        )
    else:
        return bn
def patmat(bn, pat, junk=False, strict=True):
    # './_data/thing/dtype/asdfg/123-4567890.png',
    # './_data/thing/dtype/{sub}/{a}-{b:int}.png',
    # 'asdfkjlj,123/stuff/238b9',
    # '{a},{b:float}/stuff/{c}',
    # '*,{b:float}.{d}/{}/{c}',
    if pat is None:
        return Dict({0: bn}) if junk else Dict()
    try:
        bn = bn + '?'
        pat = pat + '?'
        pat = pat.replace('*', '{}')
        t = pat.split('{')
        s = [(t[0], None), ]  # exact match
        for q in t[1:]:
            a,b = q.split('}')
            a = a.split(':')
            if len(a)==1:
                a,f = a[0], str
            elif len(a)==2:
                a,f = a[0], {
                    'int': int,
                    'float': float,
                    'str': str,
                }[a[1]]
            else:
                assert 0
            s.append((a, f))
            s.append((b, None))
        ans = Dict()
        i = 0
        mc = 0
        for si,(q,f) in enumerate(s):
            if f is None:
                i += len(q)
            else:
                n = bn[i:].index(s[si+1][0])
                v = f(bn[i:i+n])
                if q=='':
                    if junk:
                        ans[mc] = v
                        mc += 1
                else:
                    ans[q] = v
                i += n
        return ans
    except:
        if not strict:
            return None
        else:
            assert 0, f'"{bn[:-1]}" does not match "{pat[:-1]}"'
def walk(dn, pat, junk=False, strip=False):
    ans = []
    for p,ds,fs in os.walk(dn):
        for f in ds+fs:
            f = f'{p}/{f}'
            q = patmat(f[len(dn)+1:], pat, junk=junk, strict=False)
            if q:
                if strip: q['fnstrip'] = Dict(vars(fnstrip(f,1)))
                ans.append((f, q))
    return sorted(ans, key=lambda x: x[0])

# r: open for reading (default)
# w: open for writing, truncating the file first
# x: open for exclusive creation, failing if the file already exists
# a: open for writing, appending to the end of file if it exists
# b: binary mode
# t: text mode (default)
# +: open for updating (reading and writing)
def read(fn, mode='r', buffering=-1, encoding=None,
        errors=None, newline=None, closefd=True, opener=None):
    with open(fn, mode=mode, buffering=buffering, encoding=encoding,
        errors=errors, newline=newline, closefd=closefd, opener=opener) as handle:
        return handle.read()
def write(text, fn, mode='w', buffering=-1, encoding=None,
        errors=None, newline=None, closefd=True, opener=None):
    with open(fn, mode=mode, buffering=buffering, encoding=encoding,
        errors=errors, newline=newline, closefd=closefd, opener=opener) as handle:
        return handle.write(text)
def read_bns(fn, safe=False, sort=False, sort_key=None):
    ans = [
        line
        for line in read(fn).split('\n')
        if line!=''
    ]
    if sort:
        ans = sorted(ans, key=sort_key)
    if safe:
        return safe_bns(ans)
    else:
        return ans
def safe_bns(bns):
    return np.array(bns, dtype=np.string_)
def safe_bn(bn):
    return np.string_(bn)
def unsafe_bns(bns):
    return [unsafe_bn(q) for q in bns]
def unsafe_bn(bn, bns=None):
    if np.issubdtype(type(bn), np.integer):
        assert bns is not None
        return unsafe_bn(bns[bn])
    elif isinstance(bn, np.bytes_):
        return str(bn, encoding='utf-8')
    elif isinstance(bn, str):
        return bn
    else:
        assert 0, f'bn not understood'

import pickle
def pload(fn, mode='rb'):
    with open(fn, mode) as handle:
        return pickle.load(handle)
def pdump(obj, fn, mode='wb'):
    with open(fn, mode) as handle:
        return pickle.dump(obj, handle)

import json
def jread(fn, mode='r'):
    with open(fn, mode) as handle:
        return json.load(handle)
def jwrite(x, fn, mode='w', indent='\t', sort_keys=False):
    with open(fn, mode) as handle:
        return json.dump(x, handle, indent=indent, sort_keys=sort_keys)

try:
    import yaml
    def yread(fn, mode='r'):
        with open(fn, mode) as handle:
            return yaml.safe_load(handle)
    def ywrite(x, fn, mode='w', default_flow_style=False):
        if dict!=Dict:
            x = Dict(top=x).to_dict()['top']
        with open(fn, mode) as handle:
            yaml.dump(x, handle, default_flow_style=default_flow_style)
except:
    pass


################ MISC ################

try:
    hakase = f'{os.environ["PROJECT_DN"]}/_env/__hakase__.jpg'
    if not os.path.isfile(hakase):
        hakase = f'{os.environ["PROJECT_DN"]}/env/__hakase__.jpg'
    if not os.path.isfile(hakase):
        hakase = f'{os.environ["PROJECT_DN"]}/__env__/__hakase__.jpg'
    if not os.path.isfile(hakase):
        hakase = None
except:
    hakase = None

enum = enumerate

def isonow():
    return datetime.now().isoformat().replace(':','-').replace('.','-')

def mem(units='m'):
    return psutil.Process(os.getpid()).memory_info().rss / {
        'b': 1,
        'k': 1e3,
        'm': 1e6,
        'g': 1e9,
        't': 1e12,
    }[units[0].lower()]
def gpumem(units='m', query='used'):
    assert query in ['used', 'free', 'total']
    out = subprocess.check_output(
        f'nvidia-smi --query-gpu=memory.{query} --format=csv'.split()
    ).decode('ascii').split('\n')[:-1][1:]
    ans = [int(x.split()[0]) for i, x in enumerate(out)][0] * {
        'b': 1e6,
        'k': 1e3,
        'm': 1,
        'g': 1e-3,
        't': 1e-9,
    }[units[0].lower()]
    return ans

def chunk_cols(x, n):
    return [x[i:i+n] for i in range(0, len(x), n)]
def chunk_rows(x, n):
    return chunk_cols(x, int(math.ceil(len(x)/n)))

def idxs2list(idxs, n=None):
    if isinstance(idxs, list) or isinstance(idxs, tuple):
        return idxs
    elif isinstance(idxs, slice) or isinstance(idxs, range):
        a,b,c = idxs.start, idxs.stop, idxs.step
        if a is None: a = 0
        if b is None: b = n
        if c is None: c = 1
        if a<0: a = len(self)+a
        if b<0: b = len(self)+b
        return list(range(a,b,c))
    assert 0, 'data not understood'


################ AESTHETIC ################

try:
    from tqdm import tqdm as tqdm_
    from tqdm import trange as trange_
    from tqdm.auto import tqdm, trange
except:
    pass

class Table:
    def __init__(self,
            table,
            delimiter=' ',
            orientation='br',
            double_colon=True,
                ):
        self.delimiter = delimiter
        self.orientation = orientation
        self.t = Table.parse(
            table, delimiter, orientation, double_colon,
        )
        return

    # rendering
    def __str__(self):
        return self.render()
    def __repr__(self):
        return self.render()
    def render(self):
        # set up empty entry
        empty = ('', Table._spec(self.orientation, transpose=False))

        # calculate table size
        t = copy.deepcopy(self.t)
        totalrows = len(t)
        totalcols = [len(r) for r in t]
        assert min(totalcols)==max(totalcols)
        totalcols = totalcols[0]

        # string-ify
        for i in range(totalrows):
            for j in range(totalcols):
                x,s = t[i][j]
                sp = s[11]
                if sp: x = eval(f'f"{{{x}{sp}}}"')
                Table._put((str(x),s), t, (i,j), empty)

        # expand delimiters
        _repl = lambda s: \
            s[:2] + (1,0,0,0,0) + s[7:10] + (1,) + s[11:] \
            if s[2] else \
            s[:2] + (0,0,0,0,0) + s[7:10] + (1,) + s[11:]
        for i,row in enumerate(t):
            for j,(x,s_own) in enumerate(row):
                # expand delim_up(^)
                if s_own[3]:
                    u,v = i,j
                    while 0<=u:
                        _,s = t[u][v]
                        if (i,j)!=(u,v) and (s[2] and not s[10]): break
                        Table._put((x, _repl(s)), t, (u,v), empty)
                        u -= 1

                # expand delim_down(v)
                if s_own[4]:
                    u,v = i,j
                    while u<totalrows:
                        _,s = t[u][v]
                        if (i,j)!=(u,v) and (s[2] and not s[10]): break
                        Table._put((x, _repl(s)), t, (u,v), empty)
                        u += 1

                # expand delim_right(>)
                if s_own[5]:
                    u,v = i,j
                    while v<totalcols:
                        _,s = t[u][v]
                        if (i,j)!=(u,v) and (s[2] and not s[10]): break
                        Table._put((x, _repl(s)), t, (u,v), empty)
                        v += 1

                # expand delim_left(<)
                if s_own[6]:
                    u,v = i,j
                    while 0<=v:
                        _,s = t[u][v]
                        if (i,j)!=(u,v) and (s[2] and not s[10]): break
                        Table._put((x, _repl(s)), t, (u,v), empty)
                        v -= 1

        # justification calculation
        widths = [0,] * totalcols  # j
        heights = [0,] * totalrows # i
        for i,row in enumerate(t):
            for j,(x,s) in enumerate(row):
                # height caclulation
                heights[i] = max(heights[i], x.count('\n'))

                # width calculation; non-delim fillers no contribution
                if s[2] or not s[10]:
                    w = max(len(q) for q in x.split('\n'))
                    widths[j] = max(widths[j], w)
        # no newline ==> height=1
        heights = [h+1 for h in heights]

        # render table
        rend = []
        roff = 0
        for i,row in enumerate(t):
            for j,(x,s) in enumerate(row):
                w,h = widths[j], heights[i]

                # expand fillers and delimiters
                if s[2] or s[10]:
                    xs = x.split('\n')
                    xw0 = min(len(l) for l in xs)
                    xw1 = max(len(l) for l in xs)
                    xh = len(xs)
                    if (xw0==xw1==w) and (xh==h):
                        pass
                    elif xw0==xw1==w:
                        x = '\n'.join([xs[0],]*h)
                    elif xh==h:
                        x = '\n'.join([(l[0] if l else '')*w for l in xs])
                    else:
                        x = x[0] if x else ' '
                        x = '\n'.join([x*w,]*h)

                # justify horizontally
                x = [
                    l.rjust(w) if s[0] else l.ljust(w)
                    for l in x.split('\n')
                ]

                # justify vertically
                plus = [' '*w,]*(h-len(x))
                x = plus+x if not s[1] else x+plus

                # input to table
                for r,xline in enumerate(x):
                    Table._put(xline, rend, (roff+r,j), None)
            roff += h

        # return rendered string
        return '\n'.join([''.join(r) for r in rend])

    # parsing
    def _spec(s, transpose=False):
        if ':' in s:
            i = s.index(':')
            sp = s[i:]
            s = s[:i]
        else:
            sp = ''
            s = s.lower()
        return (
            int('r' in s),                                      #  0:: 0:left(l)   1:right(r)
            int('t' in s),                                      #  1:: 0:bottom(b) 1:top(t)
            int(any([i in s for i in ['.','<','>','^','v']])),  #  2:: delim_here(.)
            int('^' in s if not transpose else '<' in s),       #  3:: delim_up(^)
            int('v' in s if not transpose else '>' in s),       #  4:: delim_down(v)
            int('>' in s if not transpose else 'v' in s),       #  5:: delim_right(>)
            int('<' in s if not transpose else '^' in s),       #  6:: delim_left(<)
            int('+' in s),                                      #  7:: subtable(+)
            int('-' in s if not transpose else '|' in s),       #  8:: subtable_horiz(-)
            int('|' in s if not transpose else '-' in s),       #  9:: subtable_vert(|)
            int('_' in s),                                      # 10:: fill(_); if delim, overwrite; else fit
            sp,                                                 # 11:: special(:) f-string for numbers
        )
    def _put(obj, t, ij, empty):
        i,j = ij
        while i>=len(t):
            t.append([])
        while j>=len(t[i]):
            t[i].append(empty)
        t[i][j] = obj
        return
    def parse(
            table,
            delimiter=' ',
            orientation='br',
            double_colon=True,
                ):
        # disabling transpose
        transpose = False

        # set up empty entry
        empty = ('', Table._spec(orientation, transpose))

        # transpose
        t = []
        for i,row in enumerate(table):
            for j,item in enumerate(row):
                ij = (i,j) if not transpose else (j,i)
                if type(item)==tuple and len(item)==2 and type(item[1])==str:
                    item = (item[0], Table._spec(item[1], transpose))
                elif double_colon and type(item)==str and '::' in item:
                    x,s = item.split('::')
                    item = (x, Table._spec(s, transpose))
                else:
                    item = (item, Table._spec(orientation, transpose))
                Table._put(item, t, ij, empty)

        # normalization
        maxcol = 0
        maxrow = len(t)
        for i,row in enumerate(t):
            # take element number into account
            maxcol = max(maxcol, len([i for i in row if not i[1][2]]))

            # take subtables into account
            for j,(x,s) in enumerate(row):
                if s[7]:
                    r = len(x)
                    maxrow = max(maxrow, i+r)
                    c = max(len(q) for q in x)
                    maxcol = max(maxcol, j+c)
                elif s[8]:
                    c = len(x)
                    maxcol = max(maxcol, j+c)
                elif s[9]:
                    r = len(x)
                    maxrow = max(maxrow, i+r)
        totalcols = 2*maxcol + 1
        totalrows = maxrow
        t += [[]]*(totalrows-len(t))
        newt = []
        delim = (delimiter, Table._spec('._'+orientation, transpose))
        for i,row in enumerate(t):
            wasd = False
            tcount = 0
            for j in range(totalcols):
                item = t[i][tcount] if tcount<len(t[i]) else empty
                isd = item[1][2]
                if wasd and isd:
                    Table._put(empty, newt, (i,j), empty)
                    wasd = False
                elif wasd and not isd:
                    Table._put(item, newt, (i,j), empty)
                    tcount += 1
                    wasd = False
                elif not wasd and isd:
                    Table._put(item, newt, (i,j), empty)
                    tcount += 1
                    wasd = True
                elif not wasd and not isd:
                    Table._put(delim, newt, (i,j), empty)
                    wasd = True
        t = newt

        # normalization: add dummy last column for delimiter
        for row in t:
            row.append(empty)

        # expand subtables
        delim_cols = [i for i in range(totalcols) if i%2==0]
        while True:
            # find a table
            ij = None
            for i,row in enumerate(t):
                for j,item in enumerate(row):
                    st,s = item
                    if s[7]:
                        ij = i,j,7,st,s
                        break
                    elif s[8]:
                        ij = i,j,8,st,s
                        break
                    elif s[9]:
                        ij = i,j,9,st,s
                        break
                if ij is not None: break
            if ij is None: break

            # replace its specs
            i,j,k,st,s = ij
            s = list(s)
            s[7] = s[8] = s[9] = 0
            s = tuple(s)

            # expand it
            if k==7: # 2d table
                for x,row in enumerate(st):
                    for y,obj in enumerate(row):
                        a = i+x if not transpose else i+y
                        b = j+2*y if not transpose else j+2*x
                        Table._put((obj, s), t, (a,b), None)
            if k==8: # subtable_horiz
                for y,obj in enumerate(st):
                    Table._put((obj, s), t, (i,j+2*y), None)
            if k==9: # subtable_vert
                for x,obj in enumerate(st):
                    Table._put((obj, s), t, (i+x,j), None)

        # return, finally
        return t


if __name__=='__main__':
    import this









