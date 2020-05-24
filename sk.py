from __future__ import division, print_function

import argparse
import math
import random
import sys
from functools import reduce
import numpy as np
sys.dont_write_bytecode = True


class o():
    "Anonymous container"

    def __init__(i, **fields):
        i.override(fields)

    def override(i, d): i.__dict__.update(d); return i

    def __repr__(i):
        d = i.__dict__
        name = i.__class__.__name__
        return name + '{' + ' '.join([':%s %s' % (k, d[k])
                                      for k in i.show()]) + '}'

    def show(i):
        return [k for k in sorted(i.__dict__.keys())
                if not "_" in k]


The = o(cohen=0.3, small=3, epsilon=0.01,
            width=50, lo=0, hi=100, conf=0.01, b=1000, a12=0.56)


def main(file, cohen=0.3,small=3,epsilon=0.01,conf=0.01,width=50,a12=0.56,text=12,latex=False,higher=False,useA12=False,deltaV=0.147):
    The.cohen = 0.3
    The.small = small
    The.epsilon = epsilon
    The.conf = conf
    The.width = width + 0
    The.a12 = a12 + 0
    The.text = text + 0
    The.latex = latex
    The.higherBetter = higher
    The.useA12 = useA12
    The.deltaV = deltaV

    log = None
    higherBetter = The.higherBetter
    # print(higherBetter)
    all = {}
    now = []
    for line in file.readlines():
        # print('ha',line)
        string = True
        for word in line.split(' '):
            word = thing(word)
            if string and isinstance(word, str):
                # print('word',word)
                now = all[word.strip('\n')] = all.get(word, [])
                break
            else:
                string=False
                if isinstance(word,float):
                    # print('not', word)
                    now += [word]
    rdivDemo([[k] + v for k, v in all.items()], latex, higherBetter)

def _ab12():
    def a12slow(lst1, lst2):
        more = same = 0.0
        for x in sorted(lst1):
            for y in sorted(lst2):
                if x == y:
                    same += 1
                elif x > y:
                    more += 1
        return (more + 0.5 * same) / (len(lst1) * len(lst2))

    random.seed(1)
    l1 = [random.random() for x in range(5000)]
    more = [random.random() * 2 for x in range(5000)]
    l2 = [random.random() for x in range(5000)]
    less = [random.random() / 2.0 for x in range(5000)]
    for tag, one, two in [("1less", l1, more),
                          ("1more", more, less), ("same", l1, l2)]:
        t1 = msecs(lambda: a12(l1, less))
        t2 = msecs(lambda: a12slow(l1, less))
        print("\n", tag, "\n", t1, a12(one, two))
        print(t2, a12slow(one, two))


"""
Note that the test code \__ab12_ shows that our fast and slow method generate the same A12 score, but the
fast way does so thousands of times faster. The following tests show runtimes for lists of 5000 numbers:
    experimemt  msecs(fast)  a12(fast)  msecs(slow)  a12(slow)
    1less          13        0.257      9382           0.257
    1more          20        0.868      9869           0.868
    same           11        0,502      9937           0.502
## Significance Tests
### Standard Utils
Didn't we do this before?
"""

"""
Misc functions:
"""
rand = random.random
any = random.choice
seed = random.seed
exp = lambda n: math.e ** n
ln = lambda n: math.log(n, math.e)
g = lambda n: round(n, 2)


def median(lst, ordered=False):
    if not ordered:
        lst = sorted(lst)
    n = len(lst)
    p = n // 2
    if n % 2:
        return lst[p]
    q = p - 1
    q = max(0, min(q, n))
    return (lst[p] + lst[q]) / 2


def msecs(f):
    import time
    t1 = time.time()
    f()
    return (time.time() - t1) * 1000


def pairs(lst):
    "Return all pairs of items i,i+1 from a list."
    last = lst[0]
    for i in lst[1:]:
        yield last, i
        last = i


def xtile(lst, lo=The.lo, hi=The.hi, width=The.width,
          chops=[0.1, 0.3, 0.5, 0.7, 0.9],
          marks=["-", " ", " ", "-", " "],
          bar="|", star="*", show=" %3.0f"):
    """The function _xtile_ takes a list of (possibly)
    unsorted numbers and presents them as a horizontal
    xtile chart (in ascii format). The default is a
    contracted _quintile_ that shows the
    10,30,50,70,90 breaks in the data (but this can be
    changed- see the optional flags of the function).
    """

    def pos(p):
        return ordered[int(len(lst) * p)]

    def place(x):
        return int(width * float((x - lo)) / (hi - lo + 0.00001))

    def pretty(lst):
        return ', '.join([show % x for x in lst])

    ordered = sorted(lst)
    lo = min(lo, ordered[0])
    hi = max(hi, ordered[-1])
    what = [pos(p) for p in chops]
    where = [place(n) for n in what]
    out = [" "] * width
    for one, two in pairs(where):
        for i in range(one, two):
            out[i] = marks[0]
        marks = marks[1:]
    out[int(width / 2)] = bar
    out[place(pos(0.5))] = star
    return '(' + ''.join(out) + ")," + pretty(what)


def _tileX():
    import random
    random.seed(1)
    nums = [random.random() ** 2 for _ in range(100)]
    print(xtile(nums, lo=0, hi=1.0, width=25, show=" %5.2f"))


"""````
### Standard Accumulator for Numbers
Note the _lt_ method: this accumulator can be sorted by median values.
Warning: this accumulator keeps _all_ numbers. Might be better to use
a bounded cache.
"""


class Num:
    "An Accumulator for numbers"

    def __init__(i, name, inits=[]):
        i.n = i.m2 = i.mu = 0.0
        i.all = []
        i._median = None
        i.name = name
        i.rank = 0
        for x in inits: i.add(x)

    def s(i):
        return (i.m2 / (i.n - 1)) ** 0.5

    def add(i, x):
        i._median = None
        i.n += 1
        i.all += [x]
        delta = x - i.mu
        i.mu += delta * 1.0 / i.n
        i.m2 += delta * (x - i.mu)

    def __add__(i, j):
        return Num(i.name + j.name, i.all + j.all)

    def quartiles(i):
        def p(x): return float(g(xs[x]))

        i.median()
        xs = i.all
        n = int(len(xs) * 0.25)
        return p(n), p(2 * n), p(3 * n)

    def median(i):
        if not i._median:
            i.all = sorted(i.all)
            i._median = median(i.all)
        return i._median

    def __lt__(i, j):
        return i.median() < j.median()

    def spread(i):
        i.all = sorted(i.all)
        n1 = i.n * 0.25
        n2 = i.n * 0.75
        if len(i.all) <= 1:
            return 0
        if len(i.all) == 2:
            return i.all[1] - i.all[0]
        else:
            return i.all[int(n2)] - i.all[int(n1)]


"""
### Cliff's Delta
"""


def cliffsDelta(lst1, lst2, **dull):
    """Returns delta and true if there are more than 'dull' differences"""
    if not dull:
        dull = {'small': 0.147, 'medium': 0.33, 'large': 0.474}  # effect sizes from (Hess and Kromrey, 2004)
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j * repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j) * repeats
    d = (more - less) / (m * n)
    smallEffect = lookup_size(d, dull)
    return smallEffect


def lookup_size(delta, dull):
    """
    :type delta: float
    :type dull: dict, a dictionary of small, medium, large thresholds.
    """
    delta = abs(delta)
    if delta <= dull['small']:
        return False
    if dull['small'] < delta < dull['medium']:
        return True
    if dull['medium'] <= delta < dull['large']:
        return True
    if delta >= dull['large']:
        return True


def runs(lst):
    """Iterator, chunks repeated values"""
    for j, two in enumerate(lst):
        if j == 0:
            one, i = two, 0
        if one != two:
            yield j - i, one
            i = j
        one = two
    yield j - i + 1, two


"""
### The A12 Effect Size Test
As above
"""


def a12slow(lst1, lst2):
    "how often is x in lst1 more than y in lst2?"
    more = same = 0.0
    for x in lst1:
        for y in lst2:
            if x == y:
                same += 1
            elif x > y:
                more += 1
    x = (more + 0.5 * same) / (len(lst1) * len(lst2))
    return x


def a12(lst1, lst2):
    "how often is x in lst1 more than y in lst2?"

    def loop(t, t1, t2):
        while t1.j < t1.n and t2.j < t2.n:
            h1 = t1.l[t1.j]
            h2 = t2.l[t2.j]
            h3 = t2.l[t2.j + 1] if t2.j + 1 < t2.n else None
            if h1 > h2:
                t1.j += 1;
                t1.gt += t2.n - t2.j
            elif h1 == h2:
                if h3 and h1 > h3:
                    t1.gt += t2.n - t2.j - 1
                t1.j += 1;
                t1.eq += 1;
                t2.eq += 1
            else:
                t2, t1 = t1, t2
        return t.gt * 1.0, t.eq * 1.0

    # --------------------------
    lst1 = sorted(lst1, reverse=True)
    lst2 = sorted(lst2, reverse=True)
    n1 = len(lst1)
    n2 = len(lst2)
    t1 = o(l=lst1, j=0, eq=0, gt=0, n=n1)
    t2 = o(l=lst2, j=0, eq=0, gt=0, n=n2)
    gt, eq = loop(t1, t1, t2)
    return gt / (n1 * n2) + eq / 2 / (n1 * n2) >= The.a12


def _a12():
    def f1(): return a12slow(l1, l2)

    def f2(): return a12(l1, l2)

    for n in [100, 200, 400, 800, 1600, 3200, 6400]:
        l1 = [rand() for _ in range(n)]
        l2 = [rand() for _ in range(n)]
        t1 = msecs(f1)
        t2 = msecs(f2)
        print(n, g(f1()), g(f2()), int((t1 / t2)))


"""
n   a12(fast)       a12(slow)       tfast / tslow
--- --------------- -------------- --------------
100  0.53           0.53               4
200  0.48           0.48               6
400  0.49           0.49              28
800  0.5            0.5               26
1600 0.51           0.51              72
3200 0.49           0.49             109
6400 0.5            0.5              244
````
## Non-Parametric Hypothesis Testing
The following _bootstrap_ method was introduced in
1979 by Bradley Efron at Stanford University. It
was inspired by earlier work on the
jackknife.
Improved estimates of the variance were [developed later][efron01].
[efron01]: http://goo.gl/14n8Wf "Bradley Efron and R.J. Tibshirani. An Introduction to the Bootstrap (Chapman & Hall/CRC Monographs on Statistics & Applied Probability), 1993"
To check if two populations _(y0,z0)_
are different, many times sample with replacement
from both to generate _(y1,z1), (y2,z2), (y3,z3)_.. etc.
"""


def sampleWithReplacement(lst):
    "returns a list same size as list"

    def any(n): return random.uniform(0, n)

    def one(lst): return lst[int(any(len(lst)))]

    return [one(lst) for _ in lst]


"""
Then, for all those samples,
 check if some *testStatistic* in the original pair
hold for all the other pairs. If it does more than (say) 99%
of the time, then we are 99% confident in that the
populations are the same.
In such a _bootstrap_ hypothesis test, the *some property*
is the difference between the two populations, muted by the
joint standard deviation of the populations.
"""


def testStatistic(y, z):
    """Checks if two means are different, tempered
     by the sample size of 'y' and 'z'"""
    tmp1 = tmp2 = 0
    for y1 in y.all: tmp1 += (y1 - y.mu) ** 2
    for z1 in z.all: tmp2 += (z1 - z.mu) ** 2
    s1 = (float(tmp1) / (y.n - 1)) ** 0.5
    s2 = (float(tmp2) / (z.n - 1)) ** 0.5
    delta = z.mu - y.mu
    if s1 + s2:
        delta = delta / ((s1 / y.n + s2 / z.n) ** 0.5)
    return delta


"""
The rest is just details:
+ Efron advises
  to make the mean of the populations the same (see
  the _yhat,zhat_ stuff shown below).
+ The class _total_ is a just a quick and dirty accumulation class.
+ For more details see [the Efron text][efron01].
"""


def bootstrap(y0, z0, conf=The.conf, b=The.b):
    """The bootstrap hypothesis test from
       p220 to 223 of Efron's book 'An
      introduction to the boostrap."""

    class total():
        "quick and dirty data collector"

        def __init__(i, some=[]):
            i.sum = i.n = i.mu = 0;
            i.all = []
            for one in some: i.put(one)

        def put(i, x):
            i.all.append(x);
            i.sum += x;
            i.n += 1;
            i.mu = float(i.sum) / i.n

        def __add__(i1, i2): return total(i1.all + i2.all)

    y, z = total(y0), total(z0)
    x = y + z
    tobs = testStatistic(y, z)
    yhat = [y1 - y.mu + x.mu for y1 in y.all]
    zhat = [z1 - z.mu + x.mu for z1 in z.all]
    bigger = 0.0
    for i in range(b):
        if testStatistic(total(sampleWithReplacement(yhat)),
                         total(sampleWithReplacement(zhat))) > tobs:
            bigger += 1
    return bigger / b < conf


"""
#### Examples
"""


def _bootstraped():
    def worker(n=1000,
               mu1=10, sigma1=1,
               mu2=10.2, sigma2=1):
        def g(mu, sigma): return random.gauss(mu, sigma)

        x = [g(mu1, sigma1) for i in range(n)]
        y = [g(mu2, sigma2) for i in range(n)]
        return n, mu1, sigma1, mu2, sigma2, \
               'different' if bootstrap(x, y) else 'same'

    # very different means, same std
    print(worker(mu1=10, sigma1=10,
                 mu2=100, sigma2=10))
    # similar means and std
    print(worker(mu1=10.1, sigma1=1,
                 mu2=10.2, sigma2=1))
    # slightly different means, same std
    print(worker(mu1=10.1, sigma1=1,
                 mu2=10.8, sigma2=1))
    # different in mu eater by large std
    print(worker(mu1=10.1, sigma1=10,
                 mu2=10.8, sigma2=1))


"""
Output:
"""

# _bootstraped()

(1000, 10, 10, 100, 10, 'different')
(1000, 10.1, 1, 10.2, 1, 'same')
(1000, 10.1, 1, 10.8, 1, 'different')
(1000, 10.1, 10, 10.8, 1, 'same')

"""
Warning- the above took 8 seconds to generate since we used 1000 bootstraps.
As to how many bootstraps are enough, that depends on the data. There are
results saying 200 to 400 are enough but, since I am  suspicious man, I run it for 1000.
Which means the runtimes associated with bootstrapping is a significant issue.
To reduce that runtime, I avoid things like an all-pairs comparison of all treatments
(see below: Scott-knott).  Also, BEFORE I do the boostrap, I first run
the effect size test (and only go to bootstrapping in effect size passes:
"""


def different(l1, l2):
    # return bootstrap(l1,l2) and a12(l2,l1)
    # return a12(l2,l1) and bootstrap(l1,l2)
    if The.useA12:
        return a12(l2, l1) and bootstrap(l1, l2)
    else:
        return cliffsDelta(l1, l2) and bootstrap(l1, l2)


"""
## Saner Hypothesis Testing
The following code, which you should use verbatim does the following:
+ All treatments are clustered into _ranks_. In practice, dozens
  of treatments end up generating just a handful of ranks.
+ The numbers of calls to the hypothesis tests are minimized:
    + Treatments are sorted by their median value.
    + Treatments are divided into two groups such that the
      expected value of the mean values _after_ the split is minimized;
    + Hypothesis tests are called to test if the two groups are truly difference.
          + All hypothesis tests are non-parametric and include (1) effect size tests
            and (2) tests for statistically significant numbers;
          + Slow bootstraps are executed  if the faster _A12_ tests are passed;
In practice, this means that the hypothesis tests (with confidence of say, 95%)
are called on only a logarithmic number of times. So...
+ With this method, 16 treatments can be studied using less than _&sum;<sub>1,2,4,8,16</sub>log<sub>2</sub>i =15_ hypothesis tests  and confidence _0.99<sup>15</sup>=0.86_.
+ But if did this with the 120 all-pairs comparisons of the 16 treatments, we would have total confidence _0.99<sup>120</sup>=0.30.
For examples on using this code, see _rdivDemo_ (below).
"""


def scottknott(data, cohen=The.cohen, small=The.small, useA12=The.a12 > 0, epsilon=The.epsilon):
    """Recursively split data, maximizing delta of
    the expected value of the mean before and
    after the splits.
    Reject splits with under 3 items"""
    all = reduce(lambda x, y: x + y, data)
    same = lambda l, r: abs(l.median() - r.median()) <= all.s() * cohen
    if useA12:
        same = lambda l, r: not different(l.all, r.all)
    big = lambda n: n > small
    return rdiv(data, all, minMu, big, same, epsilon)


def rdiv(data,  # a list of class Nums
         all,  # all the data combined into one num
         div,  # function: find the best split
         big,  # function: rejects small splits
         same,  # function: rejects similar splits
         epsilon):  # small enough to split two parts
    """Looks for ways to split sorted data,
    Recurses into each split. Assigns a 'rank' number
    to all the leaf splits found in this way.
    """

    def recurse(parts, all, rank=0):
        "Split, then recurse on each part."
        cut, left, right = maybeIgnore(div(parts, all, big, epsilon),
                                       same, parts)
        if cut:
            # if cut, rank "right" higher than "left"
            rank = recurse(parts[:cut], left, rank)+1
            rank = recurse(parts[cut:], right, rank)
        else:
            # if no cut, then all get same rank
            for part in parts:
                part.rank = rank
        return rank

    recurse(sorted(data), all)
    return data


def maybeIgnore(cut_left_right, same, parts):
    cut, left, right = cut_left_right[0], cut_left_right[1], cut_left_right[2]
    if cut:
        if same(sum(parts[:cut], Num('upto')),
                sum(parts[cut:], Num('above'))):
            cut = left = right = None
    return cut, left, right


def minMu(parts, all, big, epsilon):
    """Find a cut in the parts that maximizes
    the expected value of the difference in
    the mean before and after the cut.
    Reject splits that are insignificantly
    different or that generate very small subsets.
    """
    cut, left, right = None, None, None
    before, mu = 0, all.mu
    for i, l, r in leftRight(parts, epsilon):
        if big(l.n) and big(r.n):
            n = all.n * 1.0
            now = l.n / n * (mu - l.mu) ** 2 + r.n / n * (mu - r.mu) ** 2
            if now > before:
                before, cut, left, right = now, i, l, r
    return cut, left, right


def leftRight(parts, epsilon=The.epsilon):
    """Iterator. For all items in 'parts',
    return everything to the left and everything
    from here to the end. For reasons of
    efficiency, take a first pass over the data
    to pre-compute and cache right-hand-sides
    """
    rights = {}
    n = j = len(parts) - 1
    while j > 0:
        rights[j] = parts[j]
        if j < n: rights[j] += rights[j + 1]
        j -= 1
    left = parts[0]
    for i, one in enumerate(parts):
        if i > 0:
            if parts[i]._median - parts[i - 1]._median > epsilon:
                yield i, left, rights[i]
            left += one


"""
## Putting it All Together
Driver for the demos:
"""


def rdivDemo(data, latex=True, higherBetter=False):
    def zzz(x):
        return int(100 * (x - lo) / (hi - lo + 0.00001))
    data = list(map(lambda lst: Num(lst[0], lst[1:]),
                    data))
    print("Start sorting...")
    ranks = []
    for each in data:
        each.median()
    for x in scottknott(data, useA12=True):
        ranks += [(x.rank, x.median(), x)]
    all = []

    for _, __, x in sorted(ranks): all += x.all
    all = sorted(all)
    lo, hi = all[0], all[-1]
    line = "----------------------------------------------------"
    last = None
    formatStr = '%%4s , %%%ss ,    %%s   , %%4s ' % The.text

    #  print((formatStr  % \
    #               ('rank', 'name', 'med', 'iqr')) + "\n"+ line)
    if latex:
        latexPrint(ranks, all, higherBetter)
    for _, __, x in sorted(ranks):
        q1, q2, q3 = x.quartiles()
        d=np.round(q3-q1,2)
        print((formatStr % \
               (x.rank + 1, x.name, np.round(q2,2), d)) + \
              xtile(x.all, lo=lo, hi=hi, width=30, show="%5.2f"))
        last = x.rank


def latexPrint(ranks, all, higherBetter):
    if higherBetter:
        print(ranks)
        existed_order = [i[0] for i in ranks]
        max_order = max(existed_order)
        tmp_ranks = list()
        for P in ranks:
            tmp_ranks.append((max_order - P[0], P[1], P[2]))
            P[2].rank = max_order - P[0]
        ranks = tmp_ranks
        print(ranks)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%Latex Table%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("""\\begin{figure}[!t]
{\small
{\small \\begin{tabular}{p{.4cm}lp{1cm}rrc}
\\arrayrulecolor{darkgray}
\\rowcolor[gray]{.9}  rank & treatment & median & IQR & \\\\""")
    lo, hi = all[0], all[-1]
    for _, __, x in sorted(ranks):
        q1, q2, q3 = x.quartiles()
        q1, q2, q3 = q1 * 100, q2 * 100, q3 * 100
        print("    %d &      %s &    %d &  %d & \quart{%d}{%d}{%d}{%d} \\\\" % (
            x.rank + 1, x.name.replace('_', '\_'), q2, q3 - q1, q1,  q2,q3 - q1, 100))
        last = x.rank
    print("""\end{tabular}}
}
\\caption{%%%Enter Caption%%%
}\\label{fig:my fig}
\\end{figure}""")
    print("%%%%%%%%%%%%%%%%%%%%%%%End Latex Table%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


def xtile_modified(lst, lo=The.lo, hi=The.hi, width=The.width,
                   chops=[0.1, 0.3, 0.5, 0.7, 0.9],
                   marks=["-", " ", " ", "-", " "],
                   bar="|", star="*", show=" %3.0f"):
    """The function _xtile_ takes a list of (possibly)
    unsorted numbers and presents them as a horizontal
    xtile chart (in ascii format). The default is a
    contracted _quintile_ that shows the
    10,30,50,70,90 breaks in the data (but this can be
    changed- see the optional flags of the function).
    """

    def pos(p):
        return ordered[int(len(lst) * p)]

    def place(x):
        return int(width * float((x - lo)) / (hi - lo + 0.00001))

    def pretty(lst):
        return ', '.join([show % x for x in lst])

    ordered = sorted(lst)
    lo = min(lo, ordered[0])
    hi = max(hi, ordered[-1])
    what = [pos(p) for p in chops]
    where = [place(n) for n in what]
    out = [" "] * width
    for one, two in pairs(where):
        for i in range(one, two):
            out[i] = marks[0]
        marks = marks[1:]
    out[int(width / 2)] = bar
    out[place(pos(0.5))] = star
    print(what)
    return what


####################################

def thing(x):
    "Numbers become numbers; every other x is a symbol."
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


#
# if args.demo:
#     _rdivs()
# else:
#     main(file)
