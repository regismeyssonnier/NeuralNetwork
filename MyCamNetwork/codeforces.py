import sys 
import math


def solve():
    
    n = int(input())

    input_string = input()  
    input_list = input_string.split()  
    a = [int(x) for x in input_list] 
    
    c=0
    nba = 0
    c2 = 0
    calc = False
    for i in a:
        if i == 1:
            c+=1
            c2+=1
        elif i == 2:
            calc = True
            e = c2 // 2
            o = c2 - e
            cc=0
            if e % 2 == 0:
                nba += e // 2
            if c % 2 == 0:
                nba += c2 // 2
            else:
                nba += math.ceil(float(c2) / 2.0)

            c2 = 0
         
    if a[len(a)-1] == 1:
        #if nba %2 == 1:
        #    nba -= 1
        nba += c2

    print(nba)


T = int(input())

while T > 0:

    solve()

    T -= 1