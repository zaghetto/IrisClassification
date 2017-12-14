#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv2/opencv_modules.hpp>
#include <opencv/highgui.h>
#include <opencv/ml.h>


//////////////////////////
#include <LibRNA.h>
//////////////////////////

using namespace std;
using namespace cv;

int main(int argc, char *argv[]){

	LibRNA lib;

	/////////////////////////////////////////////////////////////
	//                     CONFIGURAÇÃO                        //
	/////////////////////////////////////////////////////////////

	// Exemplo: 
	// Entrada=> [x1; x2; x3; x4] double
	// Saída=> [y1; y2; y3]; int
	// yn deve -1 em todas as posições exceto na posição que identifica a classe em questão, que deve ser 1.
	// Os valores podem ser acrescentados como valores de cabeçalho no arquivo de dados.
	// Cada classe deve ter o mesmo número de elementos no arquivo de dados. E todos os elementos de 
	// cada classe devem estar contíguos. O número de elementos por classe deve ser par.
	// Considere o exemplo abaixo, onde temos 3 classes com 3 elementos por classe
	// 0.10; 0.30; 0.50; 0.40; 1;-1;-1
	// 0.10; 0.25; 0.51; 0.39; 1;-1;-1
	// 0.15; 0.30; 0.52; 0.40; 1;-1;-1
	// 0.51; 0.62; 0.55; 0.46;-1; 1;-1
	// 0.55; 0.52; 0.64; 0.47;-1; 1;-1
	// 0.57; 0.49; 0.60; 0.50;-1; 1;-1
	// 0.90; 0.89; 0.78; 0.90;-1;-1; 1
	// 0.91; 0.88; 0.79; 0.89;-1;-1; 1
	// 0.95; 0.79; 0.83; 0.94;-1;-1; 1
	
	// Nome do arquivo de dados
	char nomeArquivoDados[] = "dados\\opencvInput_tanh.txt";

	// Número de features no vetor de entradas. Precisa ser fornecido pelo usuário.
	int numEntradas = 6;

	// Número de saídas binárias. Precisa ser conhecido pelo usuário.
	int numClasses = 10;

	// Número de elementos por classe. Precisa ser conhecido pelo usuário.
	int numElemClasse = 60;

	// Número mínimo e máximo de neurônios na camada escondida. Serão realizados treinamentos com
	// minNeuronio até maxNeuronio com intercalos de passoNeuronio
	int minNeuronio = 10;
	int maxNeuronio = 60;

	// Variação do número de neurônios entre passos.
	int passoNeuronio = 5;

	// Quantidade de treinamentos para cada quantidade de neurônios da camada escondidade
	// Para cada quantidade de neurônios serão realizados quantTreinamento treinamentos
	int quantTreinamento = 5;

	////////////////////////////////////////////////////////////////
	// Carrega os dados para treinamento e teste
	////////////////////////////////////////////////////////////////

	// Vetor com todas as entradas e saidas da rede
	vector<struct neuralInputVector> dataVector;

	if (argc == 1)
		lib.carregaDados(nomeArquivoDados, numEntradas, numClasses, numElemClasse, dataVector);
	else
		lib.carregaDados(argv[1], numEntradas, numClasses, numElemClasse, dataVector);

	///////////////////////////////////////////////////////////
	// Organiza os dados
	///////////////////////////////////////////////////////////
	
	// Matrizes que guardam entradas e saídas separadamente	
	// Entradas
	Mat input = Mat(dataVector.size(), numEntradas, CV_32FC1);

	// Classe esperada
	Mat output = Mat(dataVector.size(), numClasses, CV_32FC1);
	
	// Separa dataVector em input e output
	lib.separaDados(dataVector, numEntradas, numClasses, input, output);

	// Normmaliza input em inputNorm. Em Coef são armazenados os maiores valores para cada feature
	vector<float> coef;
	// Vetores de entradas normalizadas
	Mat inputNorm = Mat(dataVector.size(), numEntradas, CV_32FC1);

	lib.normalizaEntrada(input, inputNorm, coef);
	
	///////////////////////////////////////////////////////////
	// Prepara os conjuntos de treinamento e teste
	///////////////////////////////////////////////////////////

	// Entradas treino
	Mat vetorInputTreino = Mat(dataVector.size() / 2, numEntradas, CV_32FC1);

	// Saídas treino
	Mat vetorOutputTreino = Mat(dataVector.size() / 2, numClasses, CV_32FC1);

	// Entreda teste
	Mat vetorInputTeste = Mat(dataVector.size() / 2, numEntradas, CV_32FC1);

	// Saída teste
	Mat vetorOutputTeste = Mat(dataVector.size() / 2, numClasses, CV_32FC1);
	cout << endl;
	cout << "Criando conjuntos de treinamento e teste...";
	// Separa dados em Treinamento e Teste
	lib.criaConjuntos(
		inputNorm,
		output,
		numClasses,  //  Número de classes
		numElemClasse, // Elementos por classe
		vetorInputTreino,
		vetorOutputTreino,
		vetorInputTeste,
		vetorOutputTeste);
	cout << "Ok" << endl;

	////////////////////////////////////////////////////////////////
	// Configura a rede, cria de treina a rede
	////////////////////////////////////////////////////////////////

	// Desempenho parcial
	float percentDesempenho = 0;

	// Melhor desempenho final
	float melhorDesempenho = 0;

	// Matriz contendo as saídas preditas pela rede
	Mat vetorOutputPred = Mat(dataVector.size() / 2, 3, CV_32FC1);

	// Estrutura que contem a matriz de confução.
	Mat matConfusao     = Mat::zeros(numClasses, numClasses, CV_32FC1);

	// Rede neural a ser treinada
	CvANN_MLP nnetwork;

	// Rede neural de teste (vamos carregar os pessos salvos no arquivo mlp.xml)
	CvANN_MLP nnetwork_load;

	// Prâmetros de treinamento, não mudam entre treinamentos.
	// Terminar o treinamento após 1000 iterações ou se houver uma mudança muito pequena no valor dos pesos da rede (<0.00001).
	// O Algoritmo utilizado é o BACKPROPAGATION. É possivel ajustar a a taxa da aprendizagem e o momentum. Recomendado (0.1 e 0.1).
	CvANN_MLP_TrainParams params( cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.00001), CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);	

	// Matriz que contem a quantidade de nerônios em cada camada 
	Mat layersConfig = Mat(3, 1, CV_32S);
	
	// Realiza treinamentos variado-se a quantidade de neurônios na camada escondida
	for (int nNeuronios = minNeuronio; nNeuronios <= maxNeuronio; nNeuronios += passoNeuronio){
		
		//Camada de entrada <= quantidade de entradas
		layersConfig.at<int>(0, 0) = vetorInputTreino.cols;

		// Camada escondida <= Varia entre treinamentos
		layersConfig.at<int>(1, 0) = nNeuronios;
		
		// Camada de saída <= Quantidade de saidas
		layersConfig.at<int>(2, 0) = vetorOutputTreino.cols;

		cout << "======================================" << endl;
		cout << "Numero de Neuronio: " << nNeuronios << endl;
		cout << "======================================" << endl;
		
		// Cria a rede utilizando a configração de camadas acima							
		nnetwork.create(layersConfig, CvANN_MLP::SIGMOID_SYM);
		
		for (int numTreinamento = 1; numTreinamento <= quantTreinamento; numTreinamento++){
			cout << "Treinando ";
			cout << numTreinamento << ": ";
			nnetwork.train(vetorInputTreino, vetorOutputTreino, Mat(), Mat(), params);
						
			////////////////////////////////////////////////////////////////
			// Verifica o desempenho da rede
			////////////////////////////////////////////////////////////////			
			// Saída teste			
			nnetwork.predict(vetorInputTeste, vetorOutputPred);
									 
			// Computando a matriz de confusão									
			lib.calDesempenho(vetorOutputTeste, vetorOutputPred, matConfusao, percentDesempenho);
			cout << percentDesempenho << endl;
									
			// Atualiza o melhor desempenho e guarda a melhor rede
			if (percentDesempenho>melhorDesempenho)
			{
				melhorDesempenho = percentDesempenho;
				FileStorage fs("dados\\mlp.xml", FileStorage::WRITE);
				nnetwork.write(*fs,"nnetwork");
				fs.release();				
				cout << "MELHOR DESEMPENHO PARCIAL: ";
				cout << melhorDesempenho << endl;				
			}		
		}

	}
	cout << "======================================" << endl;

	//////////////////////////////////////////////////////////
	// TESTA A GRAVAÇÃO E A RECUPERAÇÃO DOS DADOS           // 
	//////////////////////////////////////////////////////////

	// CARREGA A MELHOR CONFIGURAÇÃO
	nnetwork_load.load("dados\\mlp.xml");
	
	// GRAVA OS VETORES DE TREINAMENTO E TESTE PARA USO FUTURO
	// IMPORTANTE POIS OS DADOS SÃO SORTEADOS A CADA TREINAMENTO
	string caminho("dados\\");
	lib.gravaConjuntos(vetorInputTreino, vetorOutputTreino, vetorInputTeste, vetorOutputTeste, caminho);

	// CARREGA OS CONJUNTOS DE TESTE E TREINAMENTO
	// Entradas treino
	Mat vetorInputTreino_loaded = Mat(dataVector.size() / 2, numEntradas, CV_32FC1);
	// Saídas treino
	Mat vetorOutputTreino_loaded = Mat(dataVector.size() / 2, numClasses, CV_32FC1);
	// Entreda teste
	Mat vetorInputTeste_loaded = Mat(dataVector.size() / 2, numEntradas, CV_32FC1);
	// Saída teste
	Mat vetorOutputTeste_loaded = Mat(dataVector.size() / 2, numClasses, CV_32FC1);

	lib.carregaConjuntos(
		vetorInputTreino_loaded, 
		vetorOutputTreino_loaded, 
		vetorInputTeste_loaded, 
		vetorOutputTeste_loaded, 
		caminho);
	
	// Saída teste			
	nnetwork_load.predict(vetorInputTeste_loaded, vetorOutputPred);
	// Computando a matriz de confusão						
	lib.calDesempenho(vetorOutputTeste_loaded, vetorOutputPred, matConfusao, percentDesempenho);
	cout << "MELHOR DESEMPENHO FINAL: " << percentDesempenho << endl;

	system("PAUSE");

	return 0;
}


