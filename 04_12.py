
#백준(ABC)
q,w,e = map(int,input().split())
a = [q, w, e]

r,t,y = map(str,input().split())
b = [r,t,y]

for i in range(len(a) - 1):
    for j in range(len(a) - 1):
        if(a[j] > a[j+1]):
            a[j],a[j+1] = a[j+1],a[j]

num = []

for i in range(len(b)):
    if(b[i] == 'A'):
        num.append(q)
    elif(b[i] == 'B'):
        num.append(w)
    else:
        num.append(e)


print("%d %d %d"%(num[0],num[1],num[2]))




