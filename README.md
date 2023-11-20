# MarioRL
Projeto de reinforcement learning, cujo objetivo é treinar um agente para jogar super mario bros utilizando tecnicas de RL.
Nesse projeto foram abordados duas técnicas distintas, o Q-learning tabular e o o DDQL(double deep Q-learning)

# Q-learning tabular
Essa técnica consiste em armazenar em uma tabela bidimensional Q(s,a) uma médida do quão positivo é para o agente realizar a ação "a" estando no estado "s". Para calcular esses valores da tabela, utiliza-se uma técnica denomida programação dinâmica. Inicialmente os valores dessa tabela são inicializados como zero, em seguida com probabilidade &epsilon, escolhe-se uma ação aleatoria (etapa de exploração), do contrário escolhe a melhor ação no momento (etapa de exploitação). Por fim, o valor da tabela é atualizado segundo a formula:   
