import io
import random
# Initial Settings
from openfhe import *
# import openfhe.PKESchemeFeature as Feature
import time
#import fabric
#from fabric import Connection
import pandas as pd
import sys
import numpy as np
import pickle
from datetime import datetime

def subtractBits(r, a, b, carry):
    t1 = cc.EvalBinGate(XOR,a, b)  # axorb
    r[0] = cc.EvalBinGate(XOR,t1, carry)  # axorbxorc
    acomp = cc.EvalNOT(a)  # a'
    abcomp = cc.EvalNOT(t1)  # axorb'
    t2 = cc.EvalBinGate(AND,acomp, b)
    t3 = cc.EvalBinGate(AND,abcomp, carry)
    r[1] = cc.EvalBinGate(OR,t2, t3)

    return r


def subtractNumbers(ctA, ctB, nBits):
    ctRes = [False for i in range(0, nBits)]
    bitResult = [False for i in range(2)]
    ctRes[0] = cc.EvalBinGate(XOR,ctA[0], ctB[0])
    t1 = cc.EvalNOT(ctA[0])
    carry = cc.EvalBinGate(AND,t1, ctB[0])
    for i in range(1, nBits):
        bitResult = subtractBits(bitResult, ctA[i], ctB[i], carry)
        ctRes[i] = bitResult[0]
        carry = bitResult[1]
    return ctRes
    
def fixSizeBoolList(decimal,size):
    x = [int(x) for x in bin(decimal)[2:]]
    x = list(map(bool, x))
    x = [False]*(size - len(x)) + x
    pow2 = []
    for i in range(size):
      pow2.append([x[i]])
    pow2.reverse()
    return pow2
    
def boolListToInt(bitlists):
    out = 0
    for bit in bitlists:
        out = (out << 1) | bit
    return out
    
def approaddBits(r, a, b, carry):
    '''# Xor(t1[0], a, carry[0])
    t1 = vm.gate_xor(a, b)
    # Xor(t2[0], b, carry[0])
    # Xor(r[0], a, t2[0])
    r[0] = vm.gate_xor(t1, carry)
    # And(t1[0], t1[0], t2[0])
    t2 = vm.gate_and(a, carry)
    t3 = vm.gate_and(b, carry)
    t4=vm.gate_and(a,b)
    t5= vm.gate_or(t2,t3)
    # Xor(r[1], carry[0], t1[0])po
    r[1] = vm.gate_or(t5, t4)'''
    #2to1
    '''r[0]= cc.EvalBinGate(OR,a, b)    
    r[1]=  cc.EvalNOT(r[0])'''
    #3to2
    r[1]= cc.EvalBinGate(OR,a, b)    #p0 + p1
    temp=cc.EvalBinGate(AND,a, b)
    r[0]=cc.EvalBinGate(OR,carry, temp)   #p0 p1 + p2
    #r[1]=cc.EvalBinGate(OR,w0, w1)
    #r[1] = cc.EvalBinGate(AND,w0, w1)'''

     
    #cba1
    '''temp= cc.EvalBinGate(AND,b,carry)
    r[1]=cc.EvalBinGate(OR,temp, a)
    r[0]=cc.EvalNOT(r[1])''''
    
    #cba4
    '''temp= cc.EvalBinGate(AND,a, b)
    temp1= cc.EvalBinGate(AND, b,carry)
    temp2= cc.EvalBinGate(AND,a, carry)
    temp3= cc.EvalBinGate(OR,temp, temp1)
    r[1]=cc.EvalBinGate(OR,temp3,temp2)
    r[0]=cc.EvalNOT(r[1])'''
    
    
    '''#cba2
    temp= vm.gate_and(a, b)
    r[1]=vm.gate_xor(temp, carry)
    r[0]=vm.gate_not(r[1])'''
    '''#cba3
    temp= vm.gate_and(a, b)
    r[1]=vm.gate_or(temp, carry)
    r[0]=vm.gate_not(r[1])'''
    '''#cba4
    temp= vm.gate_and(a, b)
    temp1= vm.gate_and( b,carry)
    temp2= vm.gate_and(a, carry)
    temp3= vm.gate_or(temp, temp1)
    r[1]=vm.gate_or(temp3,temp2)
    r[0]=vm.gate_not(r[1])'''
    '''# Xor(t1[0], a, carry[0])
    t1 = vm.gate_xor(a, b)
    # Xor(t2[0], b, carry[0])
    # Xor(r[0], a, t2[0])
    r[0] = vm.gate_xor(t1, carry)
    # And(t1[0], t1[0], t2[0])
    t2 = vm.gate_and(a, carry)
    t3 = vm.gate_and(b, carry)
    t4=vm.gate_and(a,b)
    t5= vm.gate_or(t2,t3)
    # Xor(r[1], carry[0], t1[0])
    r[1] = vm.gate_or(t5, t4)'''

    return r

def addBits(r, a, b, carry):
    '''# Xor(t1[0], a, carry[0])
    t1 = vm.gate_xor(a, b)
    # Xor(t2[0], b, carry[0])
    # Xor(r[0], a, t2[0])
    r[0] = vm.gate_xor(t1, carry)
    # And(t1[0], t1[0], t2[0])
    t2 = vm.gate_and(a, carry)
    t3 = vm.gate_and(b, carry)
    t4=vm.gate_and(a,b)
    t5= vm.gate_or(t2,t3)
    # Xor(r[1], carry[0], t1[0])po
    r[1] = vm.gate_or(t5, t4)'''
    '''r[1]= vm.gate_or(a, b)    #p0 + p1
    temp=vm.gate_and(a, b)
    r[0]=vm.gate_or(carry, temp)   #p0 p1 + p2
    #r[1]=vm.gate_or(w0, w1)
    #r[1] = vm.gate_and(w0, w1)'''


    '''#cba1
    temp= vm.gate_and(b,carry)
    r[1]=vm.gate_or(temp, a)
    r[0]=vm.gate_not(r[1])'''
    '''#cba2
    temp= vm.gate_and(a, b)
    r[1]=vm.gate_xor(temp, carry)
    r[0]=vm.gate_not(r[1])'''
    '''#cba3
    temp= vm.gate_and(a, b)
    r[1]=vm.gate_or(temp, carry)
    r[0]=vm.gate_not(r[1])
    #cba4
    temp= vm.gate_and(a, b)
    temp1= vm.gate_and( b,carry)
    temp2= vm.gate_and(a, carry)
    temp3= vm.gate_or(temp, temp1)
    r[1]=vm.gate_or(temp3,temp2)
    r[0]=vm.gate_not(r[1])'''
    # Xor(t1[0], a, carry[0])
    t1 = cc.EvalBinGate(XOR,a, b)
    # Xor(t2[0], b, carry[0])
    # Xor(r[0], a, t2[0])
    r[0] = cc.EvalBinGate(XOR,t1, carry)
    # And(t1[0], t1[0], t2[0])
    t2 = cc.EvalBinGate(AND,a, carry)
    t3 = cc.EvalBinGate(AND,b, carry)
    t4=cc.EvalBinGate(AND,a,b)
    t5= cc.EvalBinGate(OR,t2,t3)
    # Xor(r[1], carry[0], t1[0])
    r[1] = cc.EvalBinGate(OR,t5, t4)

    return r


def addNumbers(ctA, ctB, nBits):
    ctRes = [False for i in range(nBits)]
    bitResult = [False for i in range(2)]
    ctRes[0] = cc.EvalBinGate(XOR, ctA[0], ctB[0])
    carry = cc.EvalNOT(cc.EvalBinGate(NAND, ctA[0], ctB[0]))
    for i in range(1, nBits):
        if i > 4:
            bitResult = addBits(bitResult, ctA[i], ctB[i], carry)
            ctRes[i] = bitResult[0]
            carry = bitResult[1]
        else:
            bitResult = approaddBits(bitResult, ctA[i], ctB[i], carry)
            ctRes[i] = bitResult[0]
            carry = bitResult[1]

    return ctRes   


def mulNumbers(ctA, ctB, sk, input_bits, output_bits):
    result = [cc.Encrypt(sk, False) for _ in
              range(output_bits)]
    temp = [cc.Encrypt(sk, False) for _ in
              range(output_bits)]
    # [False for _ in range(output_bits)]
    # andRes = [False for _ in range(input_bits)]

    for i in range(input_bits):
        andResLeft = [cc.Encrypt(sk, False) for _ in
                      range(output_bits)]
        #temp=mux(temp,ctA,ctB[i],size)
        for j in range(input_bits):
            andResLeft[j + i] = cc.EvalBinGate(AND, ctA[j], ctB[i])
            # andResLeft[j + i] = nufhe.LweSampleArray.copy(andRes[j])
        result = addNumbers(andResLeft, result, output_bits)
        #result_bits = [cc.Decrypt(sk, result[i]) for i in range(size * 2)]
        #print(" nuFHE multiplication intermdiate number is : ",i,boolListToInt(result_bits))

    return result[:size]
    
def Convert_list(string):
    list1=[]
    list1[:0]=string
    print(list1)
    list1=[int(i)for i in list1 ]
    listb=[]
    for i in list1:
        if i==0:
            listb.append(False)
        else:
            listb.append(True)

    #print(listb)
    return listb

def twos_complement(n,nbits):
    a=f"{n & ((1 << nbits) - 1):0{nbits}b}"
    #print(type(a))
    a=Convert_list(a)
    a.reverse()
    return a
def listToString(s):
    # initialize an empty string
    list1=[int(i)for i in s ]
    listp=[]
    for i in list1:
        if i==False:
            listp.append('0')
        else:
            listp.append('1')

    #print(listp)
    str1 = ""
    # traverse in the string
    s=['delim'.join([str(elem) for elem in sublist]) for sublist in listp]
    #print(s)
    for ele in s:
        str1 += ele
    # return string
    #print(str1)
    return str1
    
def twos_comp_val(val,bits):
    """compute the 2's complement of int value val"""
    #val=listToString(val)


    if (val & (1 << bits - 1)) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val
    
def compare_bit(  a, b,  lsb_carry,  tmp):
  result= cc.Encrypt(sk, False)
  tmp=cc.EvalBinGate(XNOR,a, b)
  result=mux(tmp,lsb_carry, a)
  return result
  
def minimum(  a,  b,  nb_bits):
    tmps1= cc.Encrypt(sk, False)
    tmps2= cc.Encrypt(sk, True)
    #initialize the carry to 0
    #run the elementary comparator gate n times
    for i in range(nb_bits):
      tmps1= compare_bit(a[i],b[i],tmps1,tmps2)
    #tmps[0] is the result of the comparaison: 0 if a is larger, 1 if b is larger
    #select the max and copy it to the result
    return tmps1
    
def mux(c_s, true, false):
    temp1 = cc.EvalBinGate(NAND, c_s, true)
    temp1_and = cc.EvalNOT(temp1)
    # print("temp1:", )
    temp2 = cc.EvalBinGate(NAND, cc.EvalNOT(c_s), false)
    temp2_and = cc.EvalNOT(temp2)
    # print("temp2:",cc.Decrypt(sk, temp2))
    mux_result = cc.EvalBinGate(OR, temp1_and, temp2_and)
    return mux_result  
      
def predict(ctA,sk, output_bits):
    zero = [cc.Encrypt(sk, False) for _ in range(output_bits)]
    onen=  [cc.Encrypt(sk, True) for _ in range(output_bits)]
    onep=  [cc.Encrypt(sk, False) for _ in range(output_bits)]
    one= cc.Encrypt(sk, True)
    temp= cc.Encrypt(sk, True)
    temp=ctA[output_bits-1]
    comp_res= cc.Encrypt(sk, True)
    onep[0]=one
    ctRes = [cc.Encrypt(sk, False) for _ in range(output_bits)]
    # Copy(ctRes[i], bitResult[0]);
    #comp_res= minimum(ctA,zero,output_bits)
    #temp=
    #comp_res=
    for i in range(output_bits):
      ctRes[i] = mux(temp,onen[i],onep[i])
    # Copy(carry[0], bitResult[1])
    return ctRes

def make_neg(n,nbits):
    list1=[int(i) for i in range(0,len(n)) ]
    listp=[]
    
    one= cc.Encrypt(sk, True)
    onep=  [cc.Encrypt(sk, False) for _ in range(nbits)]
    onep[0]=one
    testone= [False for i in range(nbits)]
    testone=onep[:]
    
    n=addNumbers(n,testone,nbits)
    
    return n
    
if __name__ == '__main__':

    
   with open('/home/user/EdgeMLaas-main_approx/pc11.txt') as f:
    lines = []
    for line in f:
        lines.append([int(v) for v in line.split()])
    cc = BinFHEContext()
    cc.GenerateBinFHEContext(STD128)
    sk = cc.KeyGen()
    cc.BTKeyGen(sk)
    size=16
    bits = [[False] for i in range(size - 2)]
    zeros = [[False] for i in range(size)]
    test_size = int(input("Please enter a string:\n"))
    
    onep=  [cc.Encrypt(sk, False) for _ in range(size)]
    one= cc.Encrypt(sk, True)
    zero=cc.Encrypt(sk, False)
    onep[0]=one
    
   
   
    w_p=[-8, -6, 8, -1, -1,  5, 2,  1,  3,  3, -8,  1]
    PO2=[1,2,4,8,16,32,64,128,256,512,1024,2048]
    featuresize=11
    w=[]
    w_b=[]
    pc=[[False for i in range(size)]for i in range(12)]
    for i in range(12):  
        p_x0=twos_complement(PO2[i],size) 
        for j in range(size):
            pc[i][j]=cc.Encrypt(sk, p_x0[j])
            
    for i in range(featuresize):
        w.append([False for i in range(size)])
        w_b.append(twos_complement(w_p[i],size))
        for j in range(size):
            w[i][j] = cc.Encrypt(sk, w_b[i][j])
    bias=[False for i in range(size)]
    bias_b=twos_complement(w_p[11],size)
    print("start time",datetime.now())
    for i in range(size):
      bias[i]=cc.Encrypt(sk, bias_b[i])

    plain_predict=[]
    ciphertext1=[False for i in range(size)]
    ciphertext2=[False for i in range(size)]
    ciphertext3=[False for i in range(size)]
    ciphertext4=[False for i in range(size)]
    ciphertext5=[False for i in range(size)]
    ciphertext6=[False for i in range(size)]
    ciphertext7=[False for i in range(size)]
    ciphertext8=[False for i in range(size)]
    ciphertext9=[False for i in range(size)]
    ciphertext10=[False for i in range(size)]
    ciphertext11=[False for i in range(size)]
    
    for k in range(test_size):
        print("k values is ",k)
        
        temp=int(lines[k][0])
        b_x0=twos_complement(temp,size)
        temp=int(lines[k][1])
        b_x1=twos_complement(temp,size)
        temp=int(lines[k][2])
        b_x2=twos_complement(temp,size)
        temp=int(lines[k][3])
        b_x3=twos_complement(temp,size)
        temp=int(lines[k][4])
        b_x4=twos_complement(temp,size)
        temp=int(lines[k][5])
        b_x5=twos_complement(temp,size)
        temp=int(lines[k][6])
        b_x6=twos_complement(temp,size)
        temp=int(lines[k][7])
        b_x7=twos_complement(temp,size)
        temp=int(lines[k][8])
        b_x8=twos_complement(temp,size)
        temp=int(lines[k][9])
        b_x9=twos_complement(temp,size)
        temp=int(lines[k][10])
        b_x10=twos_complement(temp,size)
        temp=int(lines[k][11])
        for j in range(size):
            ciphertext1[j] = cc.Encrypt(sk, b_x0[j]) 
            ciphertext2[j] = cc.Encrypt(sk, b_x1[j])
            ciphertext3[j] = cc.Encrypt(sk, b_x2[j])
            ciphertext4[j] = cc.Encrypt(sk, b_x3[j])
            ciphertext5[j] = cc.Encrypt(sk, b_x4[j])
            ciphertext6[j] = cc.Encrypt(sk, b_x5[j])
            ciphertext7[j] = cc.Encrypt(sk, b_x6[j])
            ciphertext8[j] = cc.Encrypt(sk, b_x7[j])
            ciphertext9[j] = cc.Encrypt(sk, b_x8[j])
            ciphertext10[j] = cc.Encrypt(sk, b_x9[j])
            ciphertext11[j] = cc.Encrypt(sk, b_x10[j])
        
        start_time = time.time()
      
        temp1 = [False for i in range(size)]
        temp2 = [False for i in range(size)]
       
        presult_mul1 = mulNumbers(ciphertext1,w[0], sk, size, size * 2)
        temp1=presult_mul1[:]
        temp1.reverse()
        '''result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("mul1",twos_comp_val(int(pa,2),len(pa)))'''
        
        presult_mul2 = mulNumbers(ciphertext2, w[1], sk, size, size * 2)
        temp1=presult_mul2[:]
        temp1.reverse()
        '''result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("mul2",twos_comp_val(int(pa,2),len(pa)))'''
        presult_mul3 = mulNumbers(ciphertext3, w[2], sk, size, size * 2)
        '''temp1=presult_mul3[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("mul3",twos_comp_val(int(pa,2),len(pa)))'''
        
        presult_mul4 = mulNumbers(ciphertext4, w[3], sk, size, size * 2)
        '''temp1=presult_mul4[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("mul4",twos_comp_val(int(pa,2),len(pa)))'''
        
        presult_mul5 = mulNumbers(ciphertext5, w[4], sk, size, size * 2)
        
        '''temp1=presult_mul5[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("mul5",twos_comp_val(int(pa,2),len(pa)))'''
        
        presult_mul6 = mulNumbers(ciphertext6, w[5], sk, size, size * 2)
        '''temp1=presult_mul6[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("mul6",twos_comp_val(int(pa,2),len(pa)))'''
        
        presult_mul7 = mulNumbers(ciphertext7, w[6], sk, size, size * 2)
        '''temp1=presult_mul7[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("mul7",twos_comp_val(int(pa,2),len(pa)))'''
        
        presult_mul8 = mulNumbers(ciphertext8, w[7], sk, size, size * 2)
        '''temp1=presult_mul8[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("mul8",twos_comp_val(int(pa,2),len(pa)))'''
        
        presult_mul9 = mulNumbers(ciphertext9, w[8], sk, size, size * 2)
        '''temp1=presult_mul9[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("mul9",twos_comp_val(int(pa,2),len(pa)))'''
        
        presult_mul10 = mulNumbers(ciphertext10, w[9], sk, size, size * 2)
        '''temp1=presult_mul10[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("mul10",twos_comp_val(int(pa,2),len(pa)))'''
        
        presult_mul11 = mulNumbers(ciphertext11, w[10], sk, size, size * 2)
        '''temp1=presult_mul11[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("mul11",twos_comp_val(int(pa,2),len(pa)))'''


        presult_add1 = addNumbers(presult_mul1, presult_mul2,  size)
        '''temp1=presult_add1[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("firstadd",pa)
        print("add1",twos_comp_val(int(pa,2),len(pa)))'''
        presult_add2 = addNumbers(presult_mul3, presult_mul4,  size)
        '''temp1=presult_add2[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("secondadd",pa)
        print("add2",twos_comp_val(int(pa,2),len(pa)))'''
        presult_add3 = addNumbers(presult_mul5, presult_mul6,  size)
        '''temp1=presult_add3[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("add3",twos_comp_val(int(pa,2),len(pa)))'''
        presult_add4 = addNumbers(presult_mul7, presult_mul8,  size)
        '''temp1=presult_add4[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("add4",twos_comp_val(int(pa,2),len(pa)))'''
        presult_add5 = addNumbers(presult_mul9, presult_mul10,  size)
        '''temp1=presult_add5[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("add5",twos_comp_val(int(pa,2),len(pa)))'''
        #first step
        presult_add11 = addNumbers(presult_add1, presult_add2,  size)
        '''temp1=presult_add11[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("add11",twos_comp_val(int(pa,2),len(pa)))'''
        presult_add12 = addNumbers(presult_add3, presult_add4,  size)
        '''temp1=presult_add12[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("add12",twos_comp_val(int(pa,2),len(pa)))'''
        presult_add13 = addNumbers(presult_add5, presult_mul11,  size)
        '''temp1=presult_add13[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("add13",twos_comp_val(int(pa,2),len(pa)))'''
        #secong step
        presult_add21 = addNumbers(presult_add11, presult_add12,  size)
        '''temp1=presult_add21[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("add21",twos_comp_val(int(pa,2),len(pa)))'''
        presult_add22 = addNumbers(presult_add13, bias,  size)
        '''temp1=presult_add22[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("add22",twos_comp_val(int(pa,2),len(pa)))'''
        '''temp1=presult_add3[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("thirdadd",pa)
        print("add3",twos_comp_val(int(pa,2),len(pa)))'''


        wx = addNumbers(presult_add21,presult_add22,  size)
        '''temp1=wx[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("wx",twos_comp_val(int(pa,2),len(pa)))'''
        signbit=wx[len(wx)-1]
        #print("signbit is ",cc.Decrypt(sk, signbit))
        
        
        
         
        wxcpy=[False for i in range(size)]
        wxcpy=wx[:]
        for i in range(size):
            bits3=wx[i]
            wx[i]=cc.EvalNOT(bits3)
        wxtwos=make_neg(wx,size)
        '''temp1=wxtwos[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("wxtwos",twos_comp_val(int(pa,2),len(pa)))'''
        x=[False for i in range(size)]

        for i in range(size):
            x[i]=mux(signbit,wxtwos[i],wxcpy[i])
        '''temp1=x[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("x",twos_comp_val(int(pa,2),len(pa))) '''   
        m=addNumbers(x,onep,size)
        '''temp1=m[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("m values is",twos_comp_val(int(pa,2),len(pa)))
        temp1=wxcpy[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("wxcpy values is",twos_comp_val(int(pa,2),len(pa)))'''
        
        #Division code
        abs_sub = [[False for i in range(size)]for i in range(12)]
        rsft1 = [False for i in range(size)]
        rsft2 = [False for i in range(size)]
        N1 = [False for i in range(size)]
        Dc = [False for i in range(size)]
        Nc = [False for i in range(size)]
        abs_sub[0]=[cc.Encrypt(sk, False) for _ in range(size)]
        s1=cc.Encrypt(sk, False)
        
        Nc=wxcpy[:]
        '''temp1=Nc[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("Nc is",twos_comp_val(int(pa,2),len(pa)))'''
        Dc=m[:]
        '''temp1=Dc[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("Dc is",twos_comp_val(int(pa,2),len(pa)))'''
        for i in range(1,12):
            print("i=",i)
            sub=subtractNumbers(pc[i],Dc,size)
            signbit=sub[size-1]
            for j in range(size):
                temp2[j]=cc.EvalNOT(sub[j])
            subtwos=addNumbers( temp2,onep,size)
        
            for j in range(size):
                abs_sub[i][j]=mux(signbit,subtwos[j],sub[j])
        
            for j in range(size-1):
                rsft1[j] =Nc[j+1]  
            rsft1[size-1]=Nc[size-1]
            '''temp1=rsft1[:]
            temp1.reverse()
            result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
            pa=listToString(result_bits)   
            print("rsft1",twos_comp_val(int(pa,2),len(pa))) '''
            for j in range(size):
                Nc[j]=mux(signbit,rsft1[j],Nc[j])
           
            sub2=subtractNumbers(abs_sub[i],abs_sub[i-1],size)
            s2=sub2[size-1]
            s3=cc.EvalBinGate(AND,cc.EvalNOT(signbit),cc.EvalBinGate(AND,s1,s2))
            #s3=AND(NOT(signbit),AND(s1.s2))
        
          
            for j in range(size):
                Nc[j]=mux(s3,rsft1[j],Nc[j])
            
            #save for next iteration    
            s1=signbit
            '''temp1=Nc[:]
            temp1.reverse()
            result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
            pa=listToString(result_bits)   
            print("Nc",twos_comp_val(int(pa,2),len(pa)))  '''  
    
        temp1=Nc[:]
        '''temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)   
        print("Division result",twos_comp_val(int(pa,2),len(pa)))  '''          

        
        end_time=time.time()
        predict1=predict(Nc,sk,size)
        '''temp1=predict1[:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        
        print("nuFHE multiplication number is ",twos_comp_val(int(pa,2),len(pa)))'''
        
        
    print("start time is",start_time)
    print("end time",end_time)
    print("prediction time",end_time - start_time)
    print("endtime",datetime.now(),"start time",start_time)
    #result.reverse()
   
        
        
    
