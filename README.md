# MarioRL
Projeto de reinforcement learning, cujo objetivo é treinar um agente para jogar super mario bros utilizando tecnicas de RL.
Nesse projeto foram abordados duas técnicas distintas, o Q-learning tabular e o o DDQL(double deep Q-learning)

# Q-learning tabular
Essa técnica consiste em armazenar em uma tabela bidimensional Q(s,a) uma médida do quão positivo é para o agente realizar a ação "a" estando no estado "s".

Para calcular esses valores da tabela, utiliza-se uma técnica denomida programação dinâmica. Inicialmente os valores dessa tabela são inicializados como zero, em cada uma das interações,com uma determinada probabilidade, escolhe-se uma ação aleatoria (etapa de exploração), do contrário escolhe a melhor ação no momento (etapa de exploitação). Por fim, o valor da tabela é atualizado segundo a formula:   


<p align="center">
  <img src="images/formula.png" />
</p>


Após cada interação, o ambiente retorna uma imagem de dimensões (84X84X3), se fosse utilizado essa imagem como um estado, seriam necessários uma quantidade enorme de memória para representar todos os estados, o que tornaria impossível utilizar o método tabular. Para solucioanr esse problema, foi utilizado a implementação do seguinte github https://github.com/yumouwei/super-mario-bros-reinforcement-learning/tree/main para poder converter a imagem para uma com mais informações.


<p align="center">
  <img src="images/mario_simplificado.png"/>
</p>

Utilizando as classes SMB, SBMGRIB e SMBWRAPPER, foi possível converter uma imagem de (84,84,3) em uma de tamanho (16,16) como mostrado na figura acima, de modo que: 

- -1: representa o inimigo
- 0: espaço vazio
- 1: tiles
- 2: Mario

Assim foi possível atraves de uma iteração em cada ciclo da imagem identificar a posição do mário dos inimigos e com isso converter a imagem em um vetor de bits que será o estado que representará a imagem

Foi utilizado um vetor com 13 bits, com os seguintes significados

- bit para verificar se o mario está no chão
- um bit para identificar se ele pode pular
- três bits para identificar se existem inimigos perto, a media distância ou longe
- um bit para ver se ele tomou um hit de um inimigo
- três bits para identificar se existe um obstáculo a frente
- três bits para identificar se existe um obstáculo próximo
- Um bit para identificar se ele matou um inimigo

Com relação a recompensa (R), foram utilizadas os seguinte critérios:

- v = x2 - x1: bonificar se ele andou para frente
- c = c0 - c1: penalisar pela demora de tempo
- d = 0 se vivo e d = -15 se morto
- R = v + c + d

  Além disso foi criado uma função para penalisar o mário caso ele pule em momentos dencessários, como quando não há obstáculos, precipicios ou inimigos.
