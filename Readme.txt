Autor: Samuel Lipovetsky
Usage:
 python TP2.py <arquivodeentrada.txt> <saidateste>

Output:

saidateste.jpg - representação da Q-table
saidateste.gif - Representação dos passos tomados pelo agente

exemplo de <arquivodeentrada.txt> :

500 0.1 0.9 -0.06
5
0 0 0 0 7
0 0 0 0 -1
10 -1 0 0 0
0 0 0 0 4
-1 0 0 0 0

500= Número de passos
0.1 = taxa de aprendizado
0.9 = Fator de desconto
-0.06 = Recompensa padrão para os estados reprensentado no grid comoo 0

5= tamanho do grid

Valores do grid:
10 = o agente
-1 = obstaculo
4  = recompensa -1 , derrota
7  = recomepensa +1 , vitória
0  = recomepensa padrão