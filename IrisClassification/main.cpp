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
	//                     CONFIGURA��O                        //
	/////////////////////////////////////////////////////////////

	// Exemplo: 
	// Entrada=> [x1; x2; x3; x4] double
	// Sa�da=> [y1; y2; y3]; int
	// yn deve -1 em todas as posi��es exceto na posi��o que identifica a classe em quest�o, que deve ser 1.
	// Os valores podem ser acrescentados como valores de cabe�alho no arquivo de dados.
	// Cada classe deve ter o mesmo n�mero de elementos no arquivo de dados. E todos os elementos de 
	// cada classe devem estar cont�guos. O n�mero de elementos por classe deve ser par.
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

	// N�mero de features no vetor de entradas. Precisa ser fornecido pelo usu�rio.
	int numEntradas = 6;

	// N�mero de sa�das bin�rias. Precisa ser conhecido pelo usu�rio.
	int numClasses = 10;

	// N�mero de elementos por classe. Precisa ser conhecido pelo usu�rio.
	int numElemClasse = 60;

	// N�mero m�nimo e m�ximo de neur�nios na camada escondida. Ser�o realizados treinamentos com
	// minNeuronio at� maxNeuronio com intercalos de passoNeuronio
	int minNeuronio = 10;
	int maxNeuronio = 60;

	// Varia��o do n�mero de neur�nios entre passos.
	int passoNeuronio = 5;

	// Quantidade de treinamentos para cada quantidade de neur�nios da camada escondidade
	// Para cada quantidade de neur�nios ser�o realizados quantTreinamento treinamentos
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
	
	// Matrizes que guardam entradas e sa�das separadamente	
	// Entradas
	Mat input = Mat(dataVector.size(), numEntradas, CV_32FC1);

	// Classe esperada
	Mat output = Mat(dataVector.size(), numClasses, CV_32FC1);
	
	// Separa dataVector em input e output
	lib.separaDados(dataVector, numEntradas, numClasses, input, output);

	// Normmaliza input em inputNorm. Em Coef s�o armazenados os maiores valores para cada feature
	vector<float> coef;
	// Vetores de entradas normalizadas
	Mat inputNorm = Mat(dataVector.size(), numEntradas, CV_32FC1);

	lib.normalizaEntrada(input, inputNorm, coef);
	
	///////////////////////////////////////////////////////////
	// Prepara os conjuntos de treinamento e teste
	///////////////////////////////////////////////////////////

	// Entradas treino
	Mat vetorInputTreino = Mat(dataVector.size() / 2, numEntradas, CV_32FC1);

	// Sa�das treino
	Mat vetorOutputTreino = Mat(dataVector.size() / 2, numClasses, CV_32FC1);

	// Entreda teste
	Mat vetorInputTeste = Mat(dataVector.size() / 2, numEntradas, CV_32FC1);

	// Sa�da teste
	Mat vetorOutputTeste = Mat(dataVector.size() / 2, numClasses, CV_32FC1);
	cout << endl;
	cout << "Criando conjuntos de treinamento e teste...";
	// Separa dados em Treinamento e Teste
	lib.criaConjuntos(
		inputNorm,
		output,
		numClasses,  //  N�mero de classes
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

	// Matriz contendo as sa�das preditas pela rede
	Mat vetorOutputPred = Mat(dataVector.size() / 2, 3, CV_32FC1);

	// Estrutura que contem a matriz de confu��o.
	Mat matConfusao     = Mat::zeros(numClasses, numClasses, CV_32FC1);

	// Rede neural a ser treinada
	CvANN_MLP nnetwork;

	// Rede neural de teste (vamos carregar os pessos salvos no arquivo mlp.xml)
	CvANN_MLP nnetwork_load;

	// Pr�metros de treinamento, n�o mudam entre treinamentos.
	// Terminar o treinamento ap�s 1000 itera��es ou se houver uma mudan�a muito pequena no valor dos pesos da rede (<0.00001).
	// O Algoritmo utilizado � o BACKPROPAGATION. � possivel ajustar a a taxa da aprendizagem e o momentum. Recomendado (0.1 e 0.1).
	CvANN_MLP_TrainParams params( cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.00001), CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);	

	// Matriz que contem a quantidade de ner�nios em cada camada 
	Mat layersConfig = Mat(3, 1, CV_32S);
	
	// Realiza treinamentos variado-se a quantidade de neur�nios na camada escondida
	for (int nNeuronios = minNeuronio; nNeuronios <= maxNeuronio; nNeuronios += passoNeuronio){
		
		//Camada de entrada <= quantidade de entradas
		layersConfig.at<int>(0, 0) = vetorInputTreino.cols;

		// Camada escondida <= Varia entre treinamentos
		layersConfig.at<int>(1, 0) = nNeuronios;
		
		// Camada de sa�da <= Quantidade de saidas
		layersConfig.at<int>(2, 0) = vetorOutputTreino.cols;

		cout << "======================================" << endl;
		cout << "Numero de Neuronio: " << nNeuronios << endl;
		cout << "======================================" << endl;
		
		// Cria a rede utilizando a configra��o de camadas acima							
		nnetwork.create(layersConfig, CvANN_MLP::SIGMOID_SYM);
		
		for (int numTreinamento = 1; numTreinamento <= quantTreinamento; numTreinamento++){
			cout << "Treinando ";
			cout << numTreinamento << ": ";
			nnetwork.train(vetorInputTreino, vetorOutputTreino, Mat(), Mat(), params);
						
			////////////////////////////////////////////////////////////////
			// Verifica o desempenho da rede
			////////////////////////////////////////////////////////////////			
			// Sa�da teste			
			nnetwork.predict(vetorInputTeste, vetorOutputPred);
									 
			// Computando a matriz de confus�o									
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
	// TESTA A GRAVA��O E A RECUPERA��O DOS DADOS           // 
	//////////////////////////////////////////////////////////

	// CARREGA A MELHOR CONFIGURA��O
	nnetwork_load.load("dados\\mlp.xml");
	
	// GRAVA OS VETORES DE TREINAMENTO E TESTE PARA USO FUTURO
	// IMPORTANTE POIS OS DADOS S�O SORTEADOS A CADA TREINAMENTO
	string caminho("dados\\");
	lib.gravaConjuntos(vetorInputTreino, vetorOutputTreino, vetorInputTeste, vetorOutputTeste, caminho);

	// CARREGA OS CONJUNTOS DE TESTE E TREINAMENTO
	// Entradas treino
	Mat vetorInputTreino_loaded = Mat(dataVector.size() / 2, numEntradas, CV_32FC1);
	// Sa�das treino
	Mat vetorOutputTreino_loaded = Mat(dataVector.size() / 2, numClasses, CV_32FC1);
	// Entreda teste
	Mat vetorInputTeste_loaded = Mat(dataVector.size() / 2, numEntradas, CV_32FC1);
	// Sa�da teste
	Mat vetorOutputTeste_loaded = Mat(dataVector.size() / 2, numClasses, CV_32FC1);

	lib.carregaConjuntos(
		vetorInputTreino_loaded, 
		vetorOutputTreino_loaded, 
		vetorInputTeste_loaded, 
		vetorOutputTeste_loaded, 
		caminho);
	
	// Sa�da teste			
	nnetwork_load.predict(vetorInputTeste_loaded, vetorOutputPred);
	// Computando a matriz de confus�o						
	lib.calDesempenho(vetorOutputTeste_loaded, vetorOutputPred, matConfusao, percentDesempenho);
	cout << "MELHOR DESEMPENHO FINAL: " << percentDesempenho << endl;

	system("PAUSE");

	return 0;
}


