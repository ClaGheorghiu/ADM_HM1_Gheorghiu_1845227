#Say "Hello, World!" With Python
print("Hello, World!")








#Python If-Else
#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
if n%2!=0:
    print('Weird')
elif n%2==0 and n in range(2,6):
    print('Not Weird')
elif n%2==0 and n in range(6,21):
    print('Weird')
elif n%2==0 and n>20:
    print('Not Weird')

#Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a+b)
print(a-b)
print(a*b)

#Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a//b)
print(a/b)

#Loops
if __name__ == '__main__':
    n = int(input())
i=0
for i in range(0,n):
    print(i**2)









#Write a function
def is_leap(year):
    leap = False
    if year%4==0 and year%100!=0:
        return True
    elif year%4==0 and year%100==0 and year%400==0:
        return True
    return leap


#Print Function
if __name__ == '__main__':
    n = int(input())
for i in range (n):
    print(i+1,end='')
    

#List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
coordinates=[[i,j,k]
         for i in range(x+1)
         for j in range(y+1)
         for k in range(z+1)
         if (i+j+k)!= n]
print(coordinates)
 
    
    #list comprehension
    #newList = [ expression(element) for element in oldList if condition ] 

#Find the Runner-Up Score!  
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    arr2=set(arr)
    arr2.remove(max(arr2))
    print(max(arr2))



#Nested Lists
if __name__ == '__main__':
    students = {}
    mylist=list()   
    for _ in range(int(input())):
        name = input()
        score = float(input())
        mylist.append([name,score])
    oldmin=min(x[1] for x in mylist) #trovo min score
    mylist=[x for x in mylist if x[1]>oldmin] # sovrascrivo lista ora senza vecchio minimo
    newmin= min(x[1] for x in mylist) #trovo nuovo min score
    soluzione=[x[0] for x in mylist if x[1]==newmin]
    soluzione.sort()
    for i in soluzione:
        print(i)


#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    #dictionary initialized
    student_marks = {}
    for _ in range(n):
        
        #gets name and then the numbers in a dictionary
        name, *line = input().split()
        
        #takes line input that is a list of strings 
        #then maps the strings to a float 
        #takes the float and return a map object [map(func,*iterables)]
        # then cast map obj to list ("scores")
        scores = list(map(float, line))
        
        #here we define the dictionary 
        #for a given name map to the list 
        #ex 'alpha':[20,30,40]
        student_marks[name] = scores
        
        #loop ends when all n people iterated through
    query_name = input()
    
    #how many marks for student (ie 3 as indicated in constraints)
    marks_l=len(student_marks[query_name])
    
    #sum marks for student
    marks_s=sum(student_marks[query_name])
    
    #avg aprx to 2nd decimal place
    marks_avg=marks_s/marks_l
    
    print("{:.2f}".format(marks_avg))

#Lists
if __name__ == '__main__':
    N = int(input())
    
    lista=[]
    
    for _ in range(N):
        cmd, *num=input().split(" ")
        if cmd=="insert":
            i=int(num[0])
            e=int(num[1])
            lista.insert(i,e)
        if cmd=="print":
            print(lista)
        if cmd=="remove":
            lista.remove(int(num[0]))
        if cmd=="append":
            lista.append(int(num[0]))
        if cmd=="sort":
            lista.sort()
        if cmd=="pop":
            lista.pop()
        if cmd=="reverse":
            lista.reverse()
            

#Tuples 
if __name__ == '__main__':
    n = int(input())  #n el in the tuple 
    l=map(int,input().split()) #n space separated int 
    t=tuple(l)  #to tuple 
    print(hash(t))  #hash


#sWAP cASE
def swap_case(s):
    s1=''
    for i in s:
        if i.islower():
            s1 = s1 + i.upper()
        elif i.isupper():
            s1= s1 + i.lower()
        else:
            s1=s1+i
    return(s1)



#String Split and Join


def split_and_join(line):
    return '-'.join(line.split(' '))

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#

def print_full_name(first, last):
    print(f"Hello {first} {last}! You just delved into python.")


#Mutations
# slicing the string and joining it back
def mutate_string(string, position, character):
    return string[:position] + character + string[position + 1:]


#Find a string
def count_substring(string, sub_string):
    count=0
    index=string.find(sub_string)    
    while(index!=-1):
        count +=1
        index=string.find(sub_string,index+1,len(string))
    return  count
    

#String Validators
if __name__ == '__main__':
    s = input()
   
    print(any(i.isalnum() for i in s))
    print(any(i.isalpha() for i in s))
    print(any(i.isdigit() for i in s))
    print(any(i.islower() for i in s))
    print(any(i.isupper() for i in  s))

#Text Alignment
w=int(input()) #width -> This must be an odd number
h='H'

#Top cone
for i in range(w):
    print((h*i).rjust(w-1)+h+(h*i).ljust(w-1))
#Top pillars
for i in range(w+1):
    print((h*w).center(w*2)+(h*w).center(w*6))
#Middle Belt
for i in range((w+1)//2):
    print((h*w*5).center(w*6))
#Bottom pillars
for i in range(w+1):
    print((h*w).center(w*2)+(h*w).center(w*6))
#Bottom cone 
for i in range(w):
    print(((h*(w-i-1)).rjust(w)+h+(h*(w-i-1)).ljust(w)).rjust(w*6))

#Text Wrap


def wrap(string, max_width):
    s=textwrap.fill(string,max_width)
    return s


#Designer Door Mat
n,m=map(int,(input().split()))
# 5<n<101 & m=3n
c='.|.'
#first half
for i in range(1,n,2):
        print((c*i).center(m,'-'))
#half
print('WELCOME'.center(m,'-'))
#second half
for i in range(n-2,0,-2):
    print((c*i).center(m,'-'))
        

#String Formatting
def print_formatted(num):
    l=len(bin(num)[2:])  #space-padded to match the width of the binary value
    for i in range(1,num+1):  #[,)
        print(
            str(i).rjust(l),                  #dec
            str(oct(i)[2:]).rjust(l),         #oct
            str(hex(i)[2:]).upper().rjust(l), #hex
            str(bin(i)[2:]).rjust(l)          #bin
            )


#Alphabet Rangoli
import string
alpha=string.ascii_lowercase

def print_rangoli(size):
    for i in range(1,2*size):
        k=alpha[abs(size-i):size] 
        k=k[-1:0:-1]+ k #before reverse alpha then alpha
        print('-'.join(k).center(4*size-3,'-'))



#Capitalize!


# Complete the solve function below.
def solve(s):
    name = []
    full_name = s.split(" ")
    for i in range(0,len(full_name)):
        name.append(full_name[i].capitalize())
    return " ".join(name)
    

#The Minion Game
def minion_game(string):
    s =0              #s cons #k voc
    k =0
    vovl = ["A","E","I","O","U"]
    for i in range(len(string)):
        if string[i] in vovl:
            k += len(string)-i
        else:
            s += len(string)-i
    if s>k:
        print("Stuart {}".format(s))
    elif(s==k):
        print("Draw")
    else:
        print("Kevin {}".format(k))


#Merge the Tools!
def merge_the_tools(string, k):             #len(string) multiple of k
    for i in range(0,len(string),k):
        sub_string=[]
        for x in range(i,i+k):
            if string[x] not in sub_string:
                sub_string.append(string[x])
        sub_seq=''.join(sub_string)
        print(sub_seq)
        sub_string.clear()  #new sub_string for the next cicle
            

#Introduction to Sets
def average(array):
    return round(sum(set(array))/len(set(array)),3)


#Symmetric Difference
M=int(input())
a=set(map(int,input().split()))
N=int(input())
b= set(map(int,input().split()))

all_int=a.union(b)
inter_int=a.intersection(b)

sym_dif= (all_int-inter_int)
sym_dif=list(sym_dif)
sym_dif.sort()
for x in sym_dif:
    print (x)

#No Idea!
n, m = map(int, input().split())
array= list(map(int, input().split()))
happyness = 0

A, B =  set(map(int, input().split())),  set(map(int, input().split()))

for i in array:
    # it works due to the average time complexity for set is O(1)
    # whereas for lists and tuples it's always O(n)
    if i in A:
        happyness += 1
    if i in B:
        happyness -= 1
print(happyness)

#Set .add() 
N=int(input())
nazioni=set()
for i in range(N):
    nazioni.add(input())
print(len(nazioni))




#Set .discard(), .remove() & .pop()
n = int(input())
s = set(input().split())
n_cmnd = int(input())
for i in range (0, n_cmnd):
    cmnd = input().split()
    if cmnd[0] == 'remove':
        s.remove(cmnd[1])
    elif cmnd[0] == 'discard':
        s.discard(cmnd[1])
    elif cmnd[0] == 'pop':
        s = set(sorted(s, reverse = True))
        s.pop()
        s = set(sorted(s, reverse = True))
if  len(s) != 0:
    print(sum(map(int, s)))
else:
    print('0')

#Set .union() Operation
n = int(input()) #number of students who have subscribed to the English newspaper
a = set(map(int,input().split())) # roll numbers of those students
m = int(input()) # number of students who have subscribed to the French newspaper
b = set(map(int,input().split())) #roll numbers of those students
print(len(a.union(b)))

#Set .intersection() Operation
n = int(input()) #number of students who have subscribed to the English newspaper
a = set(map(int,input().split())) # roll numbers of those students
m = int(input()) # number of students who have subscribed to the French newspaper
b = set(map(int,input().split())) #roll numbers of those students
print(len(a.intersection(b)))

#Set .difference() Operation
n = int(input()) #number of students who have subscribed to the English newspaper
a = set(map(int,input().split())) # roll numbers of those students
m = int(input()) # number of students who have subscribed to the French newspaper
b = set(map(int,input().split())) #roll numbers of those students
print(len(a.difference(b)))

#Set .symmetric_difference() Operation
n = int(input()) #number of students who have subscribed to the English newspaper
a = set(map(int,input().split())) # roll numbers of those students
m = int(input()) # number of students who have subscribed to the French newspaper
b = set(map(int,input().split())) #roll numbers of those students
print(len(a.symmetric_difference(b)))

#Set Mutations
n_el_A=int(input())
A=set(map(int,input().split()))
n_oth_sets=int(input())
for i in range (n_oth_sets):
    cmnd=input().split()
    if cmnd[0]=='intersection_update':
        set_=set(map(int,input().split()))
        A.intersection_update(set_)
    elif cmnd[0]=='update':
        set_=set(map(int,input().split()))
        A.update(set_)
    elif cmnd[0]=='symmetric_difference_update':
        set_=set(map(int,input().split()))
        A.symmetric_difference_update(set_)
    elif cmnd[0]=='difference_update':
        set_=set(map(int,input().split()))
        A.difference_update(set_)
    else:
        assert False

print(sum(A))
    

#The Captain's Room 

k=int(input()) #size groups
el_rnl_un=map(int,input().split()) #unordered elements of the room number list
el_rnl=sorted(el_rnl_un) #ordered elements of the room number list
for i in range(len(el_rnl)):
    if(i != len(el_rnl)-1):
        if(el_rnl[i]!=el_rnl[i-1] and el_rnl[i]!=el_rnl[i+1]):
            print(el_rnl[i])
            break;
    else:
        print(el_rnl[i])

#Check Subset
n=int(input()) #n test cases

for i in range(n):
    n_A=int(input())
    A=set(map(int,input().split()))
    n_B=int(input())
    B=set(map(int,input().split()))
    print(A.issubset(B))

#Check Strict Superset
A=set(input().split())
n=int(input()) #n other sets

r=True
for i in range(n):
    B=set(input().split())
    if B.issubset(A)==False:
        r=False
    if len(B) >= len(A):
        r=False
print(r)
         

#collections.Counter()
from collections import Counter as C

sh_num=int(input())
sh_sz=C(map(int,input().split()))
n_cust=int(input())

earn=0
for i in range(n_cust):
    size,price= map(int,input().split())
    if sh_sz[size]:
        earn += price
        sh_sz[size] -=1
print(earn)


#DefaultDict Tutorial
from collections import defaultdict

n, m = map(int,input().split())
d = defaultdict(lambda:[])

for i in range(n):
    d[input()].append(str(i+1))

for i in range(m):
    x = input()
    if x in d:
        print(" ".join(d[x]))
    else:
        print(-1)

#Collections.namedtuple()
from collections import namedtuple

N=int(input())
S=namedtuple('Student',input().split())
print(sum([int(S(*input().split()).MARKS) for _ in range(N)])/N)
    

#Collections.OrderedDict()
n = int(input(''))  # number of items
dict_ = {}  #dictionary to store the item_name : total_price
for i in range(n):
    list_ = list(map(str,input().split()))
    #if len of list_ > 2 then first 2 values will be item_name and price
    if len(list_)>2:
        name = list_[0]+' '+list_[1]
        price = int(list_[-1])
    else:
        name = list_[0]
        price = int(list_[-1])
       
    if name in dict_:
        dict_[name] +=int(price)
    else:
        dict_[name] = int(price)

for name, price in dict_.items():
    print(name,price)

#Word Order
n=int(input())
xx={}
word_l=[]

for i in range(n):
    w=input()
    word_l.append(w)
    if w in xx:
        xx[w] += 1
    else:
        xx[w]=1

print(len(xx))
print(' '.join([str(xx[w])for w in xx]))

#Collections.deque()
from collections import deque
n=int(input()) #n.operations

d=deque()
for i in range(n):
    cmnd=input().split()
    if cmnd[0]=='append':
        d.append(cmnd[1])
    elif cmnd[0]=='pop':
        d.pop()
    elif cmnd[0]=='popleft':
        d.popleft()
    elif cmnd[0]=='popleft':
        d.popleft()
    elif cmnd[0]=='appendleft':
        d.appendleft(cmnd[1])
print(' '.join(d))

#Piling Up!
from collections import deque 

for i in range(int(input())):
    m=input()
    blks= deque([int(s) for s in input().split()])
    for b in sorted(blks, reverse=True):
        if b==blks[0]:
            blks.popleft()
        elif b==blks[-1]:
            blks.pop()
        else:
            print('No');
            break
    else:
        print('Yes')

#Company Logo
#!/bin/python3

import math
import os
import random
import re
import sys
import collections



if __name__ == '__main__':
    s = sorted(input().strip())
    s_co=collections.Counter(s).most_common()
    s_co=sorted(s_co,key=lambda x: (x[1]*-1,x[0]))
    for i in range(0,3):
        print(s_co[i][0], s_co[i][1])

#Calendar Module
import calendar as cal

m,d,y=map(int,input().split())
day=cal.weekday(year=y,month=m,day=d)
print(cal.day_name[day].upper())

#Time Delta
#!/bin/python3
import math
import os
import random
import re
import sys
from datetime import datetime

def time_delta(d1,d2):
    f= '%a %d %b %Y %H:%M:%S %z'
    d1 = datetime.strptime(d1, f) 
    d2 = datetime.strptime(d2, f) 
    diff = (d2-d1).total_seconds()  
    return abs(int(diff))

for i in range(int(input())):
    print(time_delta(input(),input()))

#Exceptions
for i in range(int(input())):
    try:
        a,b=map(int,input().split())
        print(a//b)
    except Exception as e:
        print('Error Code:',e)

#Zipped!
N,X=map(int,input().split()) #N=n stud, X=n marks
L=[]
for i in range(X):
    L +=[map(float,input().split())]
for i in zip(*L):
    print(round(sum(i)/X,1))

#Athlete Sort
#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    
    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    k = int(input())

    arr.sort(key=lambda x:x[k])
    
    for i in arr:
        print(*i,sep=' ')

#Map and Lambda Function
cube = lambda x: x**3 # complete the lambda function 

def fibonacci(n):
    # return a list of fibonacci numbers
    n1,n2=0,1
    a=[n1,n2]
    if n>1:
        for i in range(2,n):
            n3=n1+n2
            n1=n2
            n2=n3
            a.append(n3)
        return(a)
    elif n==1:
        return[n1]
    else:
        return[]        
            


#XML 1 - Find the Score


def get_attr_number(node):
    return len(node.attrib) + sum(get_attr_number(a) for a in node);


#XML2 - Find the Maximum Depth


maxdepth = 0
def depth(elem, level):
    global maxdepth
    if (level==maxdepth):
        maxdepth+=1
    for i in elem:
        depth(i,level+1)


#Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        f(['+91 '+ i[-10:-5]+' '+ i[-5:] for i in l])
    return fun


#Decorators 2 - Name Directory


def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key=lambda x: int(x[2]))) 
    return inner


#Detect Floating Point Number

#Number can start with +, - or . symbol.
#Number must contain at least  decimal value
#Number must have exactly one . symbol
#Number must not give any exceptions when converted using float(N)

import re
num, pat=int(input()), r'^[+-]?\d*[.]\d*$'

#r' ' raw string
#^ start of the string
#[]matches a single char from the inside, except a newline
#? matches the preceding char zero or one times
#\d matches a single decimal digit char (0-9)
#* matches the preceding char zero or more times 
#. matches a single char, except a newline. But when [.] a dot is matched
#$ matches end of the string 

[print (bool(re.match(pat,input()))) for i in range(num)]

#Re.split()
regex_pattern = r"[.,]+"	

#+ Matches the preceding character one or one times


#Group(), Groups() & Groupdict()
# Print the first occurrence of the repeating character. If there are no repeating characters, print -1

import re
m=re.search(r'([a-zA-Z0-9)])\1+',input().strip())
print(m.group(1) if m else -1)

#Re.findall() & Re.finditer()
import re
v="aeiou"
c="qwrtypsdfghjklzxcvbnm"
m= re.findall(r'(?<=[%s])([%s]{2,})[%s]'%(c,v,c),input(),flags=re.I)
print('\n'.join(m or ['-1']))

#Re.start() & Re.end()
import re
s, k = input(), input()
indexes = [(m.start(), m.end() + (len(k) -1)) for m in re.finditer(rf"(?={k})", s)]

if len(indexes) > 0:
    for index_pair in indexes:
        print(index_pair)
else:
    print((-1, -1))

#Regex Substitution
import re

for _ in range(int(input())):
    print(re.sub(r'(?<= )(&&|\|\|)(?= )', lambda x: 'and'
     if x.group() == '&&' else 'or', input()))

#Validating Roman Numerals
migl = 'M{0,3}'
cent = '(C[MD]|D?C{0,3})'
dec = '(X[CL]|L?X{0,3})'
digit = '(I[VX]|V?I{0,3})'
regex_pattern = r"%s%s%s%s$" % (migl, cent, dec, digit)    


#Arrays



def arrays(arr):
    a =numpy.flip(numpy.array(arr,float))
    return a
    

#Shape and Reshape
import numpy

print (numpy.reshape(numpy.array(list(map(int,input().split()))),(3,3)))


#Transpose and Flatten
import numpy 

rows,cols=list(map(int,input().split()))
matrix= numpy.array([list(map(int,input().split())) for i in range(rows)])
print (matrix.transpose())
print (matrix.flatten())
    



#Concatenate
import numpy
n,m,p=map(int,input().split())
arr1=numpy.array([input().split() for i in range(n)],int)
arr2=numpy.array([input().split() for i in range(m)],int)
print(numpy.concatenate((arr1,arr2),axis=0))




#Zeros and Ones
import numpy
N=list(map(int,input().split()))
print(numpy.zeros(N,int))
print(numpy.ones(N,int))



#Eye and Identity
import numpy
numpy.set_printoptions(legacy='1.13')

n,m=list(map(int,input().split()))
print (numpy.eye(n,m))


#Array Mathematics
import numpy

n,m=list(map(int,input().split()))
a=numpy.array([input().split() for i in range(n)],int)
b=numpy.array([input().split() for i in range(n)],int)
print (numpy.add(a,b))
print (numpy.subtract(a,b))
print (numpy.multiply(a,b))
print (a//b)
print (numpy.mod(a,b))
print (numpy.power(a,b))


#Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy='1.13')
a=numpy.array([input().split()],float)
print (numpy.floor(*a))
print (numpy.ceil(*a))
print (numpy.rint(*a))

#Sum and Prod
import numpy as np

n,m=map(int,input().split())
matrix = np.array([list(map(int, input().split())) for i in range(n)])

print (np.prod(np.sum(matrix,axis=0)))

#Min and Max
import numpy

n,m=map(int,input().split())
matrix=numpy.array([list(map(int,input().split())) for i in range(n)],int)

print (numpy.max((numpy.min(matrix,axis=1))))

#Mean, Var, and Std
import numpy

n,m=map(int,input().split())
matrix=numpy.array([list(map(int,input().split())) for i in range(n)],float)

print (numpy.mean(matrix,axis=1))
print (numpy.var(matrix,axis=0))
print (round(numpy.std(matrix,axis=None),11))

#Dot and Cross
import numpy

n=int(input())
a=numpy.array([input().split() for i in range(n)],int)
b=numpy.array([input().split() for i in range(n)],int)

print (numpy.dot(a,b))

#Inner and Outer
import numpy as np

a=np.array([list(map(int,input().split()))])
b=np.array([list(map(int,input().split()))])

print (int(np.inner(a,b)))
print (np.outer(a,b))

#Polynomials
import numpy

pol=[float(x) for x in input().split()]
x=float(input())
print(numpy.polyval(pol,x))



#Linear Algebra
import numpy

n=int(input())
a=numpy.array([input().split() for i in range(n)],float)
print (round(numpy.linalg.det(a),2))

#Validating phone numbers
import re

pat= r"^[789][0-9]{9}$"
for i in range(int(input())):
    if re.search(pat, input()):
     print ('YES')
    else:
     print ('NO')

#Validating and Parsing Email Addresses
import re
for i in range(int(input())):
    x,y=input().split()
    m=re.match(r'<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>',y)
    if m:
        print(x,y)

#Hex Color Code
import re

A=int(input())
ret=False
for i in range(A):
    s=input()
    if '{' in s:
        ret= True
    elif '}' in s:
        ret= False
    elif ret:
        for color in re.findall('#[0-9a-fA-F]{3,6}', s):
            print (color)


#Birthday Cake Candles
import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    count=0
    maxH=max(candles)
    for i in candles:
        if i==maxH:
            count +=1
    return count
    
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

#Number Line Jumps
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
     if (v1>v2) and (x2-x1)%(v2-v1)==0: 
        return 'YES'
     else:
        return 'NO'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()





#Viral Advertising
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    liked=2
    tot=5
    cum=2
    for i in range(1,n):
        tot=int(tot/2)*3
        liked=int(tot/2)
        cum=cum+liked
    return cum
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip()) #days

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


#Recursive Digit Sum
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def digsum(n):
    return str(sum(int(i) for i in list(n)))

def sup_digit(n):
    if len(n)<=1:
        return n
    else:
        return sup_digit(digsum(n))

def superDigit(n, k):
    a=digsum(n)
    return sup_digit(str(int(a)*k))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

