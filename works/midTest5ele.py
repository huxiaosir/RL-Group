def rm(a, b,ts):
   if (a, b) in ts:
       ts.remove((a, b))
   else:
       ts.remove((b, a))
   return ts
def get(num):
    res = []
    for i in range(5):
        for j in range(i+1,5):
            res.append((num[i],num[j]))
    return res
n = int(input())
for i in range(n):
   num = [int(i) for i in input().split()]
   ts = get(num)
   if max(max(num[0], num[1]), max(num[2], num[3])) == max(num[0], num[1]):
       a = min(num[2], num[3])
       b = max(num[2], num[3])
       c = min(num[0], num[1])
       d = max(num[0], num[1])
   else:
       a = min(num[0], num[1])
       b = max(num[0], num[1])
       c = min(num[2], num[3])
       d = max(num[2], num[3])

   ts.remove((num[0], num[1]))
   ts.remove((num[2], num[3]))
   ts.remove((max(num[0], num[1]), max(num[2], num[3])))
   e = num[-1]

   ts = rm(b, e, ts)
   if e < a or a < e < b:
       ts = rm(a, e, ts)
   elif b < e < max(b, d) or e > max(b, d):
       ts = rm(max(b, d), e, ts)

   if e > d:
       ts = rm(b, c, ts)
       if c < b:
           ts = rm(a, c, ts)
   elif b < e < d:
       ts = rm(b, c, ts)
       if c < b:
           ts = rm(a, c, ts)
       else:
           ts = rm(c, e, ts)
   elif a < e < b:
       ts = rm(c, e, ts)
       if c < e:
           ts = rm(a, c, ts)
       else:
           ts = rm(b, c, ts)
   elif e < a:
       ts = rm(a, c, ts)
       if c > a:
           ts = rm(b, c, ts)
       else:
           ts = rm(c, e, ts)
   sorted(ts, key=lambda x: (num.index(x[0]), num.index(x[1])))

   for i in ts:
       pre = min(num.index(i[0])+1, num.index(i[1])+1)
       last = max(num.index(i[0])+1, num.index(i[1])+1)
       print(pre, last)

   print()