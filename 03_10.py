def bu(w):

    for i in range(len(w)):

        for j in range(len(w)- 1):

            if(w[j] > w[j+1]):
                w[j],w[j+1] = w[j+1],w[j]

    return w

def plus(w,p,c):

    for i in range(len(w)):

        p = p + w[i]
        c = c + p

    return c


q = int(input())

w = input()

w = w.split(' ')
for i in range(len(w)):
    w[i] = int(w[i])


p = 0
c = 0

w = bu(w)
c = plus(w,p,c)

print(c)