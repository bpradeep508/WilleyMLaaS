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
    '''r[1]= cc.EvalBinGate(OR,a, b)    #p0 + p1
    temp=cc.EvalBinGate(AND,a, b)
    r[0]=cc.EvalBinGate(OR,carry, temp) '''  #p0 p1 + p2
    #r[1]=cc.EvalBinGate(OR,w0, w1)
    #r[1] = cc.EvalBinGate(AND,w0, w1)'''

     
    #cba1
    temp= cc.EvalBinGate(AND,b,carry)
    r[1]=cc.EvalBinGate(OR,temp, a)
    r[0]=cc.EvalNOT(r[1])
    
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
        if i > 0:
            bitResult = addBits(bitResult, ctA[i], ctB[i], carry)
            ctRes[i] = bitResult[0]
            carry = bitResult[1]
        else:
            bitResult = approaddBits(bitResult, ctA[i], ctB[i], carry)
            ctRes[i] = bitResult[0]
            carry = bitResult[1]

    return ctRes   


def mulNumbers(ctA, ctB, secret_key, input_bits, output_bits):
    result = [cc.Encrypt(sk, False) for _ in
              range(output_bits)]
    temp = [cc.Encrypt(sk, False) for _ in
              range(output_bits)]
    # [[vm.empty_ciphertext((1,))] for _ in range(output_bits)]
    # andRes = [[vm.empty_ciphertext((1,))] for _ in range(input_bits)]

    for i in range(input_bits):
        andResLeft = [cc.Encrypt(sk, False) for _ in
                      range(output_bits)]
        #temp=mux(temp,ctA,ctB[i],size)
        for j in range(input_bits):
            andResLeft[j + i] = cc.EvalBinGate(AND, ctA[j], ctB[i])
            # andResLeft[j + i] = nufhe.LweSampleArray.copy(andRes[j])
        result = addNumbers(andResLeft, result, output_bits)
        #result_bits = [ctx.decrypt(secret_key, result[i]) for i in range(size * 2)]
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
    # print("temp2:",ctx.decrypt(secret_key, temp2))
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
    
    w_p=[-4, 6, -1, -1, 3, 1, 1, 2, 2, -4, 1, 6]
    
    featuresize=12
    w=[]
    w_b=[]
    for i in range(featuresize-1):
        w.append([False for i in range(size)])
        w_b.append(twos_complement(w_p[i],size))
        for j in range(size):
            w[i][j] = cc.Encrypt(sk, w_b[i][j])
    bias=[False for i in range(size)]
    bias_b=twos_complement(w_p[len(w_p)-1],size)
    
    for i in range(size):
      bias[i]=cc.Encrypt(sk, bias_b[i])

    plain_predict=[]
    print("start time",datetime.now())
    for i in range(test_size):
        print(lines[i][0],lines[i][1],lines[i][2],lines[i][3])
        temp=int(lines[i][0])
        b_x0=twos_complement(temp,size)
        temp=int(lines[i][1])
        b_x1=twos_complement(temp,size)
        temp=int(lines[i][2])
        b_x2=twos_complement(temp,size)
        temp=int(lines[i][3])
        b_x3=twos_complement(temp,size)
        temp=int(lines[i][4])
        b_x4=twos_complement(temp,size)
        temp=int(lines[i][5])
        b_x5=twos_complement(temp,size)
        temp=int(lines[i][6])
        b_x6=twos_complement(temp,size)
        temp=int(lines[i][7])
        b_x7=twos_complement(temp,size)
        temp=int(lines[i][8])
        b_x8=twos_complement(temp,size)
        temp=int(lines[i][9])
        b_x9=twos_complement(temp,size)
        temp=int(lines[i][10])
        b_x10=twos_complement(temp,size)



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
    partial_predict = [False for i in range(size)]
       
    plain_predict=[] 
    
    
    
    # 6 Mul
    presult_mul1 = mulNumbers(ciphertext1,w[0], sk, size, size * 2)
    
    presult_mul2 = mulNumbers(ciphertext2, w[1], sk, size, size * 2)
    
    presult_mul3 = mulNumbers(ciphertext3, w[2], sk, size, size * 2)
    
    presult_mul4 = mulNumbers(ciphertext4, w[3], sk, size, size * 2)
    
    presult_mul5 = mulNumbers(ciphertext5, w[4], sk, size, size * 2)
    
    presult_mul6 = mulNumbers(ciphertext6, w[5], sk, size, size * 2) 
    
    presult_mul7 = mulNumbers(ciphertext7, w[6], sk, size, size * 2)
    
    
    presult_mul8 = mulNumbers(ciphertext8, w[7], sk, size, size * 2)
    
    
    presult_mul9 = mulNumbers(ciphertext9, w[8], sk, size, size * 2)
    
    
    presult_mul10 = mulNumbers(ciphertext10, w[9], sk, size, size * 2)
    
    
    presult_mul11 = mulNumbers(ciphertext11, w[10], sk, size, size * 2)
     
     
    # 5 Add
    presult_add1 = addNumbers(presult_mul1, presult_mul2,  size)
    
    presult_add2 = addNumbers(presult_mul3, presult_mul4,  size)
    
    presult_add3 = addNumbers(presult_mul5, presult_mul6,  size) 
    
    presult_add4 = addNumbers(presult_add1, presult_add2,  size)   
    
    presult_add5 = addNumbers(presult_add3, presult_add4,  size) 
    
    presult_add6 = addNumbers(presult_mul7, presult_mul8,  size)

    
    presult_add7 = addNumbers(presult_mul9, presult_mul10,  size) 
 
       
    presult_add8 = addNumbers(presult_mul11, bias,  size)
 
    
    presult_add9 = addNumbers(presult_add6, presult_add7,  size) 
 
    
    presult_add10 = addNumbers(presult_add8, presult_add9,  size)  
    
    partial_predict = addNumbers(presult_add5, presult_add10,  size) 
        
    predict1=[cc.Encrypt(sk, False) for _ in range(size)]
    
    result_bits = [cc.Decrypt(sk, partial_predict[i]) for i in range(size)]
    result_bits.reverse()
    pa=listToString(result_bits)
    print("partial predict",twos_comp_val(int(pa,2),len(pa)))
    
    predict1=predict(partial_predict,sk,size)
    predict1.reverse()
    result_bits = [cc.Decrypt(sk, predict1[i]) for i in range(size)]
    pa=listToString(result_bits)
    plain_predict.append(twos_comp_val(int(pa,2),len(pa)))
    print(" openfhe multiplication number is : ", plain_predict)    
    print("end time",datetime.now())
    end_time = time.time()
    print(f"time taken for prediction is {float(end_time - start_time)} seconds ")   
