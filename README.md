# WilleyMLaaS


**OpenFHE Installtion (4GB RAM requried)**</br>
To setup evnvironemnt follow the instruction in https://github.com/openfheorg/openfhe-python?tab=readme-ov-file#system-level-installation</br>
 1)Each fie includes approximation and with out approxiation.</br>
 2)Chagne the file path according to your system path </br>
 3) To experminet the apprxoimate adders uncomment corresponing lines in "def approaddBits(r, a, b, carry):" and change lines in adder "def addNumbers(ctA, ctB, nBits):" function "if condtion to i>4"</br>

 **NuFHE Installtion (GPU requried)**</br>
 To setup evnvironemnt fllow the instruction in  https://nufhe.readthedocs.io/en/latest/</br>
1)Each fie includes approximation and with out approxiation.</br>
 2)Chagne the file path according to your system path</br>
 3) To experminet the apprxoimate adders uncomment corresponing lines in "def approaddBits(r, a, b, carry):" and change lines in adder "def addNumbers(ctA, ctB, nBits):" function "if condtion to i>4"</br>


 Note: if condtion i>0 "**Exact computing** " and i>4 "**apprximte computing**"
