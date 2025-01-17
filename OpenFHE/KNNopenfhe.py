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

def euclidean_distance(row1, row2, nBits):
    ones = [False for i in range(nBits)]
    distance = [cc.Encrypt(sk, 0) for _ in range(nBits)]
    c_zero = [False for i in range(nBits)]
    result = subtractNumbers(row1, row2, nBits)
    signbit = result[len(result) - 1]
    for i in range(nBits):
        temp = result[i]
        ones[i] = cc.EvalNOT(temp)
    neg2 = make_neg(ones, nBits)
    for i in range(0, nBits):
        distance[i] = mux(signbit, neg2[i], result[i])
    return distance
    
def subtractNumbers(ctA, ctB, nBits):
    ctRes = [False for i in range(0, nBits)]
    bitResult = [False for i in range(2)]
    ctRes[0] = cc.EvalBinGate(XOR, ctA[0], ctB[0])
    t1 = cc.EvalNOT(ctA[0])
    carry = cc.EvalNOT(cc.EvalBinGate(NAND,t1, ctB[0]))
    for i in range(1, nBits):
        bitResult = subtractBits(bitResult, ctA[i], ctB[i], carry)
        ctRes[i] = bitResult[0]
        carry = bitResult[1]
    return ctRes    
    
def subtractBits(r, a, b, carry):
    t1 = cc.EvalBinGate(XOR, a, b) # axorb
    r[0] = cc.EvalBinGate(XOR, t1, carry)  # axorbxorc
    acomp = cc.EvalNOT(a)  # a'
    abcomp = cc.EvalNOT(t1)  # axorb'
    t2 = cc.EvalNOT(cc.EvalBinGate(NAND, acomp, b))
    t3 = cc.EvalNOT(cc.EvalBinGate(NAND, abcomp, carry))
    r[1] = cc.EvalBinGate(OR, t2, t3)

    return r    
    
def make_neg(n, nbits):
    list1 = [int(i) for i in range(0, len(n))]
    listp = []
    # print(n)
    one = cc.Encrypt(sk, True)
    onep = [cc.Encrypt(sk, False) for _ in range(nbits)]
    onep[0] = one
    testone = [False for i in range(nbits)]
    testone = onep[:]
    n = addNumbers(n, testone, nbits)
    return n 
    
def mux(c_s, true, false):
    temp1 = cc.EvalBinGate(NAND, c_s, true)
    temp1_and = cc.EvalNOT(temp1)
    # print("temp1:", )
    temp2 = cc.EvalBinGate(NAND, cc.EvalNOT(c_s), false)
    temp2_and = cc.EvalNOT(temp2)
    # print("temp2:",ctx.decrypt(secret_key, temp2))
    mux_result = cc.EvalBinGate(OR, temp1_and, temp2_and)
    return mux_result     
    
def addBits(r, a, b, carry):
    t1 = cc.EvalBinGate(XOR, a, b)
    r[0] = cc.EvalBinGate(XOR, t1, carry)
    t2 = cc.EvalNOT(cc.EvalBinGate(NAND, a, carry))
    t3 = cc.EvalNOT(cc.EvalBinGate(NAND, b, carry))
    t4 = cc.EvalNOT(cc.EvalBinGate(NAND, a, b))
    t5 = cc.EvalBinGate(OR, t2, t3)
    r[1] = cc.EvalBinGate(OR, t5, t4)

    return r

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
    r[0]=cc.EvalBinGate(OR,carry, temp)'''  #p0 p1 + p2
    #r[1]=cc.EvalBinGate(OR,w0, w1)
    #r[1] = cc.EvalBinGate(AND,w0, w1)'''

     
    #cba1
    '''temp= cc.EvalBinGate(AND,b,carry)
    r[1]=cc.EvalBinGate(OR,temp, a)
    r[0]=cc.EvalNOT(r[1])'''
    
    #cba4
    temp= cc.EvalBinGate(AND,a, b)
    temp1= cc.EvalBinGate(AND, b,carry)
    temp2= cc.EvalBinGate(AND,a, carry)
    temp3= cc.EvalBinGate(OR,temp, temp1)
    r[1]=cc.EvalBinGate(OR,temp3,temp2)
    r[0]=cc.EvalNOT(r[1])
    
    
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
def addNumbers(ctA, ctB, nBits):
    ctRes = [False for i in range(nBits)]
    bitResult = [False for i in range(2)]
    ctRes[0] = cc.EvalBinGate(XOR, ctA[0], ctB[0])
    carry = cc.EvalNOT(cc.EvalBinGate(NAND, ctA[0], ctB[0]))
    for i in range(1, nBits):
        if i >4:
            bitResult = addBits(bitResult, ctA[i], ctB[i], carry)
            ctRes[i] = bitResult[0]
            carry = bitResult[1]
        else:
            bitResult = approaddBits(bitResult, ctA[i], ctB[i], carry)
            ctRes[i] = bitResult[0]
            carry = bitResult[1]

    return ctRes   
      
def predict(ctA, sk, output_bits):
    zero = [cc.Encrypt(sk, 0) for _ in range(output_bits)]
    onen = [cc.Encrypt(sk, 1) for _ in range(output_bits)]
    onep = [cc.Encrypt(sk, 0) for _ in range(output_bits)]
    one = cc.Encrypt(sk, 1)
    temp = cc.Encrypt(sk, 1)
    temp = ctA[output_bits - 1]
    comp_res = cc.Encrypt(sk, 1)
    onep[0] = one
    ctRes = [cc.Encrypt(sk, 0) for _ in range(output_bits)]
    for i in range(output_bits):
        ctRes[i] = mux(temp, onen[i], onep[i])
    # Copy(carry[0], bitResult[1])
    return ctRes      
    



def Convert_list(string):
    list1 = []
    list1[:0] = string
    print(list1)
    list1 = [int(i) for i in list1]
    listb = []
    for i in list1:
        if i == 0:
            listb.append(False)
        else:
            listb.append(True)
    return listb


def twos_complement(n, nbits):
    a = f"{n & ((1 << nbits) - 1):0{nbits}b}"
    a = Convert_list(a)
    a.reverse()
    return a


def listToString(s):
    # initialize an empty string
    list1 = [int(i) for i in s]
    listp = []
    for i in list1:
        if i == False:
            listp.append('0')
        else:
            listp.append('1')
    # print(listp)
    str1 = ""
    # traverse in the string
    s = ['delim'.join([str(elem) for elem in sublist]) for sublist in listp]
    # print(s)
    for ele in s:
        str1 += ele
    # return string
    # print(str1)
    return str1


def twos_comp_val(val, bits):
    """compute the 2's complement of int value val"""
    # val=listToString(val)
    if (val & (1 << bits - 1)) != 0:  # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)  # compute negative value
    return val




if __name__ == '__main__':
        


    cc = BinFHEContext()
    cc.GenerateBinFHEContext(STD128)
    sk = cc.KeyGen()
    cc.BTKeyGen(sk)
    size = 16
    file_obj = io.BytesIO()
    to_file = True
    #host = 'pi@10.171.3.196'
    #password = 'raspbeery'
    #connection = Connection(host="pi@10.171.3.196", connect_kwargs={"password": "raspbeery"}, inline_ssh_env=True)
    
    with open('/home/user/EdgeMLaas-main_approx/pc11_train.txt') as f:
        lines = []   
        for line in f:
            lines.append([int(v) for v in line.split()])
    with open('/home/user/EdgeMLaas-main_approx/pc11_test.txt') as f:
        lines1 = []
        for line in f:
            lines1.append([int(v) for v in line.split()])
    bits = [[False] for i in range(size - 2)]
    zeros = [[False] for i in range(size)]
    train_size = int(input("Please enter train a string:\n"))
    test_size = int(input("Please enter test data a string:\n"))
    neigh = int(input("Please enter no of neighbours:\n"))
    half_n = int(neigh//2)
    ciphertext1 = [[False for j in range(size)]for i in range(train_size)]
    ciphertext2 = [[False for j in range(size)]for i in range(train_size)]
    ciphertext3 = [[False for j in range(size)]for i in range(train_size)]
    ciphertext4 = [[False for j in range(size)]for i in range(train_size)]
    ciphertext5 = [[False for j in range(size)]for i in range(train_size)]
    ciphertext6 = [[False for j in range(size)]for i in range(train_size)]
    ciphertext7 = [[False for j in range(size)]for i in range(train_size)]
    ciphertext8 = [[False for j in range(size)]for i in range(train_size)]
    ciphertext9 = [[False for j in range(size)]for i in range(train_size)]
    ciphertext10 = [[False for j in range(size)]for i in range(train_size)]
    
    
    ciphertext11 = [[False for j in range(size)]for i in range(train_size)]
    
    ciphertextT1 = [[False for j in range(size)]for i in range(test_size)]
    ciphertextT2 = [[False for j in range(size)]for i in range(test_size)]
    ciphertextT3 = [[False for j in range(size)]for i in range(test_size)]
    ciphertextT4 = [[False for j in range(size)]for i in range(test_size)]
    ciphertextT5 = [[False for j in range(size)]for i in range(test_size)]
    ciphertextT6 = [[False for j in range(size)]for i in range(test_size)]
    ciphertextT7 = [[False for j in range(size)]for i in range(test_size)]
    ciphertextT8 = [[False for j in range(size)]for i in range(test_size)]
    ciphertextT9 = [[False for j in range(size)]for i in range(test_size)]
    ciphertextT10 = [[False for j in range(size)]for i in range(test_size)]
    
    thr = [False for j in range(size)]
    b_n = twos_complement(half_n, size)
    for j in range(size):
            thr[j] = cc.Encrypt(sk, b_n[j])
    print(ciphertext1)

    for i in range(train_size):
        print("i value is ", i)
        
        temp = int(lines[i][0])
        b_x0 = twos_complement(temp, size)
        print("b_x0",b_x0)
        
        temp = int(lines[i][1])
        b_x1 = twos_complement(temp, size)
        
        temp = int(lines[i][2])
        b_x2 = twos_complement(temp, size)
        
        temp = int(lines[i][3])
        b_x3 = twos_complement(temp, size)
        temp = int(lines[i][4])
        b_x4 = twos_complement(temp, size)
        
        temp = int(lines[i][5])
        b_x5 = twos_complement(temp, size)
        
        temp = int(lines[i][6])
        b_x6 = twos_complement(temp, size)
        temp = int(lines[i][7])
        b_x7 = twos_complement(temp, size)
        
        temp = int(lines[i][8])
        b_x8 = twos_complement(temp, size)
        
        temp = int(lines[i][9])
        b_x9 = twos_complement(temp, size)
        temp = int(lines[i][10])
        b_l = twos_complement(temp, size)

        
        for j in range(size):
            
            #print("ptx",b_x0[j])
            ciphertext1[i][j] = cc.Encrypt(sk, b_x0[j])
            ciphertext2[i][j] = cc.Encrypt(sk, b_x1[j])
            ciphertext3[i][j] = cc.Encrypt(sk, b_x2[j])
            ciphertext4[i][j] = cc.Encrypt(sk, b_x3[j])
            ciphertext5[i][j] = cc.Encrypt(sk, b_x4[j])
            ciphertext6[i][j] = cc.Encrypt(sk, b_x5[j])
            ciphertext7[i][j] = cc.Encrypt(sk, b_x6[j])
            ciphertext8[i][j] = cc.Encrypt(sk, b_x7[j])
            ciphertext9[i][j] = cc.Encrypt(sk, b_x8[j])
            ciphertext10[i][j] = cc.Encrypt(sk, b_x9[j])
           # ciphertext10[i][j] = cc.Encrypt(sk, b_x10[j])
            ciphertext11[i][j] = cc.Encrypt(sk, b_l[j])
            
    for i in range(test_size):
        print("i value is ", i)
       # print(lines1[i][0],lines1[i][1],lines1[i][2],lines1[i][3])
        # print(lines1[j][0],lines1[j][1],lines1[j][2],lines1[j][3])
        temp1 = int(lines1[i][0])
        b_y0 = twos_complement(temp1, size)
        temp1 = int(lines1[i][1])
        b_y1 = twos_complement(temp1, size)
        temp1 = int(lines1[i][2])
        b_y2 = twos_complement(temp1, size)
        temp1 = int(lines1[i][3])
        b_y3 = twos_complement(temp1, size)
        temp1 = int(lines1[i][4])
        b_y4 = twos_complement(temp1, size)
        temp1 = int(lines1[i][5])
        b_y5 = twos_complement(temp1, size)
        temp1 = int(lines1[i][6])
        b_y6 = twos_complement(temp1, size)
        temp1 = int(lines1[i][7])
        b_y7 = twos_complement(temp1, size)
        temp1 = int(lines1[i][8])
        b_y8 = twos_complement(temp1, size)
        temp1 = int(lines1[i][9])
        b_y9 = twos_complement(temp1, size)
        

        for j in range(size):
            ciphertextT1[i][j] = cc.Encrypt(sk, b_y0[j])
            ciphertextT2[i][j] = cc.Encrypt(sk, b_y1[j])
            ciphertextT3[i][j] = cc.Encrypt(sk, b_y2[j])
            ciphertextT4[i][j] = cc.Encrypt(sk, b_y3[j])
            ciphertextT5[i][j] = cc.Encrypt(sk, b_y4[j])
            ciphertextT6[i][j] = cc.Encrypt(sk, b_y5[j])
            ciphertextT7[i][j] = cc.Encrypt(sk, b_y6[j])
            ciphertextT8[i][j] = cc.Encrypt(sk, b_y7[j])
            ciphertextT9[i][j] = cc.Encrypt(sk, b_y8[j])
            ciphertextT10[i][j] = cc.Encrypt(sk, b_y9[j])
            
                
    print("Encryption done")  
    temp1 = [False for i in range(size)]    
    #print("temp1",temp1)

#Encryption done
    start_time = time.time()    
    print("start time",datetime.now())     
    temp1 = [False for i in range(size)]
    temp2 = [False for i in range(size)]
    dist=[]
    for i in range(train_size):
        dist.append([[False for i in range(size)] for j in range(2)])
    
# Distance computation        
    for j in range(test_size): 
        print('j=',j)
        for i in range(train_size):
            print('i=',i)
        
            presult_mul1 = euclidean_distance(ciphertext1[i],ciphertextT1[j], size)
           
            presult_mul2 = euclidean_distance(ciphertext2[i],ciphertextT2[j], size)
            
        
            presult_mul3 = euclidean_distance(ciphertext3[i],ciphertextT3[j], size)
           
        
            presult_mul4 = euclidean_distance(ciphertext4[i],ciphertextT4[j], size)
            presult_mul5 = euclidean_distance(ciphertext5[i],ciphertextT5[j], size)
            presult_mul6 = euclidean_distance(ciphertext6[i],ciphertextT6[j], size)
            presult_mul7 = euclidean_distance(ciphertext7[i],ciphertextT7[j], size)
            presult_mul8 = euclidean_distance(ciphertext8[i],ciphertextT8[j], size)
            presult_mul9 = euclidean_distance(ciphertext9[i],ciphertextT9[j], size)
            presult_mul10 = euclidean_distance(ciphertext10[i],ciphertextT10[j], size)
            
        
                
    
    
    
            presult_add1 = addNumbers(presult_mul1, presult_mul2,  size)
         
            
            presult_add2 = addNumbers(presult_mul3, presult_mul4,  size)
            presult_add3 = addNumbers(presult_mul5, presult_mul6,  size)
            presult_add4 = addNumbers(presult_mul7, presult_mul8,  size)
            presult_add5 = addNumbers(presult_mul9, presult_mul10,  size)
            
            
            add1= addNumbers(presult_add1, presult_add2,  size)
            add2= addNumbers(presult_add3, presult_add4,  size)
            add3= addNumbers(add1, add2,  size)
            
            
            
            dist[i][0] = addNumbers(add3, presult_add5,  size)
            """temp1=dist[i][0][:]
            temp1.reverse()
            result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
            pa=listToString(result_bits)
            print("distance dist[i][0] is for train data i =",i,"is",twos_comp_val(int(pa,2),len(pa))) """
    
            
            #print(type(temp))
            #x = twos_complement(deci_x,size)
            dist[i][1]=ciphertext5[i][:]
            """temp1=dist[i][1][:]
            temp1.reverse()
            result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
            pa=listToString(result_bits)
            print("Label of Train data for i =", i, "is",twos_comp_val(int(pa,2),len(pa)))"""
    
    var1 = [[False for i in range(size)]for j in range(2)]
    var2 = [[False for i in range(size)]for j in range(2)]
    onep=  [cc.Encrypt(sk, 0) for _ in range(size)]
    onen=  [cc.Encrypt(sk, 0) for _ in range(size)]
    temp_dist=  [cc.Encrypt(sk, 0) for _ in range(size)]
    temp_label=  [cc.Encrypt(sk, 0) for _ in range(size)]
    one= cc.Encrypt(sk, 1)
    zero= cc.Encrypt(sk, 0)
    for k in range(size):
      onen[k]=one
    onep[0]=one
    onep[0]=one
    signbit= cc.Encrypt(sk, 0)
    signbitnot= cc.Encrypt(sk, 0)
    for i in range( train_size-1):
            # Last i elements are already in place
            for j in range(train_size-i-1):
                #if (arr[j] > arr[j+1]):
                #swap(&arr[j], &arr[j+1]);
                var1[0] = dist[j][0][:]
                var1[1] = dist[j][1][:]
                var2[0] = dist[j+1][0][:]
                var2[1] = dist[j+1][1][:]
                temp=subtractNumbers(var1[0],var2[0], size)
                signbit=temp[size-1]
                signbitnot=cc.EvalNOT(signbit)
                
                for k in range(size):
                    dist[j][0][k]=mux(signbit,var1[0][k],var2[0][k])
                    dist[j+1][0][k]=mux(signbitnot,var1[0][k],var2[0][k])
                    dist[j][1][k]=mux(signbit,var1[1][k],var2[1][k])
                    dist[j+1][1][k]=mux(signbitnot,var1[1][k],var2[1][k])
    
    """for i in range( train_size):
        temp1=dist[i][0][:]
        temp1.reverse()
        result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("sorted distance",dist[i][0],"is",twos_comp_val(int(pa,2),len(pa)),"with label",dist[i][1])       """
             
    pos=[cc.Encrypt(sk, 0) for _ in range(size)]
    neg=[cc.Encrypt(sk, 0) for _ in range(size)]
    predict1=[cc.Encrypt(sk, 0) for _ in range(size)]
    posc=[cc.Encrypt(sk, 0) for _ in range(size)]
    negc=[cc.Encrypt(sk, 0) for _ in range(size)]
    plain_predict=[]
    
    
    sum1=[cc.Encrypt(sk, 0) for _ in range(size)]
    
    
    for i in range(neigh):
        print(i)
        signbitl=dist[i][1][size-1]
        wx=[cc.Encrypt(sk, 0) for _ in range(size)]
        wx[0]=signbitl
        sum1=addNumbers(sum1,wx,size) 
    dsign=subtractNumbers(sum1,thr,size)
    dsignbit=dsign[size-1]
    for i in range(size):
        predict1[i]=mux(dsignbit,onep[i],onen[i])  
        
    
    
    
    
    
    """for i in range(neigh):
        print(i)
        signbitl=dist[i][1][size-1]
        signbitc=onep[size-1]
        dbit=cc.EvalNOT(cc.EvalBinGate(XOR, signbitl,signbitc))
        posc=pos[:]
        negc=neg[:]
        pos=make_neg(pos,size)
        neg=make_neg(neg,size)
        for j in range(size):
            pos[j]=mux(dbit,posc[j],pos[j])
            neg[j]=mux(dbit,neg[j],negc[j])
    temp=subtractNumbers(pos,neg,size)  
    signbit=temp[size-1]
    for i in range(size):
        predict1[i]=mux(signbit,onen[i],onep[i])"""
        
        
    predict=predict(predict1,sk,size)    
    predict.reverse()   
    result_bits = [cc.Decrypt(sk, predict[i]) for i in range(size)]
    pa=listToString(result_bits)    
    print("predicted label is",twos_comp_val(int(pa,2),len(pa)))  
    print("end time",datetime.now())    
    end_time = time.time()      
    print(f"time taken to execute the program is {float(end_time -start_time)} seconds ")        
             
    """print("ciphertext1",ciphertext1)
    print("ciphertext1[0]",ciphertext1[0])
    print("ciphertext1[0][0]",ciphertext1[0][0])
        
    temp1 = [False for i in range(size)]    
    print("temp1",temp1)
    temp1=ciphertextT1[0][:]
    temp1.reverse()
    result_bits = [cc.Decrypt(sk, temp1[i]) for i in range(size)]
    pa=listToString(result_bits)
    print("decryption of ciphertext[0]",twos_comp_val(int(pa,2),len(pa)))"""
   

  
 
   

