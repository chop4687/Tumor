
######### homework1 ###########

temp1 = '((())))))'
temp2 = '((()))'
temp3 = '[[]]'

def check(string):
    check = 0
    for i in string:
        if i == '(':
            check += 1
        elif i == ')':
            check -= 1
        else:
            return print('error different input')
    if check == 0:
        return print('YES')
    else:
        return print('No')

check(temp1)
check(temp2)
check(temp3)
###############################

######## homework2 ##############
def pascal(n):
    a=[]
    for i in range(n):
        a.append([])
        a[i].append(1)
        for j in range(1,i):
            a[i].append(a[i-1][j-1]+a[i-1][j])
        if(n!=0):
            a[i].append(1)
    for k in range(n):
        print(" "*(n-k),end=" ")
        for j in range(0,k+1):
            print(a[k][j],end=" ")
        print()

pascal(5)
################################
