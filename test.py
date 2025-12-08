ans=10000000
a,b,c=map(int,input().split())
li=[[] for _ in range(10**6+1)]
for i in range(1,10**3+1):
    for j in range(i, 10**6+1,i):
        li[j].append(i)
for C in range(1,10**6+1):
    for el in li[C]:
        ans = min(ans, abs(el-a)+abs(C//el-b)+abs(C-c))
        ans = min(ans, abs(C//el-a)+abs(el-b)+abs(C-c))
print(ans)