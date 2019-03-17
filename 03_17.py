"""
#백준 문제
def bubble(q, w):

    for i in range(q):
        for j in range(q - 1):

            if (w[j] > w[j + 1]):

                w[j], w[j + 1] = w[j + 1], w[j]

    return w

q = int(input())

w = input()
w = w.split(' ')

for i in range(len(w)):

    w[i] = int(w[i])


e = input()
e = e.split(' ')

for i in range(len(e)):

    e[i] = int(e[i])


#using function
w = bubble(q,w)
print(w)
e = bubble(q,e)
print(e)

r = 0
t = 0

for i in range(q):
    r = r + w[i] * e[q-i -1]
    t = t + r
    r = 0

print(t)
"""

#백준 문제2

q = int(input())

w = input()

w = w.split(' ')

for i in range(q):
    w[i] = int(w[i])

e = []
for i in range(len(w)):

    e.append(w[i])

print(e)

for i in range(q):
    for j in range(q-1):
        if(e[j] > e[j+1]):
            e[j],e[j+1] = e[j+1],e[j]

r = e[0]

print(r)


