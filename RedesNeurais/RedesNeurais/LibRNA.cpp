#include "LibRNA.h"

LibRNA lib;

// Constructor
LibRNA::LibRNA(){

}

//Destructors
LibRNA::~LibRNA()
{

}

int LibRNA::zaghand(int a, int b){
	return a&&b;
};

int LibRNA::zaghor(int a, int b){
	return a||b;
};

int LibRNA::zaghxor(int a, int b){
	return zaghor(zaghand(!a, b), zaghand(a, !b));
};

void LibRNA::carregaDados(
	char *nomeArquivoDados,
	int numEntradas,
	int numClasses,
	int numElemClasse,
	vector<struct neuralInputVector> &vetorDados){

	// Uma única entrada
	struct neuralInputVector inputItem;

	cout << "Loading data ";

	// Ponteiro para o arquivo CSV com os dados
	FILE *data;

	// Abre o arquivo
	data = fopen(nomeArquivoDados, "r");

	// Contador de registros lidos
	unsigned int cont = 0;

	// Variável que lê o ";"
	char u;


	if (data == NULL)
	{
		printf("ERROR: Loading File");
	}
	else
	{ // São numClasses*numElemClasse registros, numElemClasse de cada classe	
		while (cont < numClasses*numElemClasse)
		{
			cout << "=";
			inputItem.entradas = (double *)malloc(numEntradas*sizeof(double));
			inputItem.nEntradas = numEntradas;
			inputItem.alvo = (int *)malloc(numClasses*sizeof(int));
			inputItem.nAlvos = numClasses;

			// Realiza a leitura das entradas armazenadas no arquivo CSV
			// São 4 entradas
			for (int i = 0; i < numEntradas; i++){
				fscanf(data, "%lf", &inputItem.entradas[i]);
				fscanf(data, "%c", &u);
				// cout << inputItem.entradas[i] << endl;
				// cout << u << endl;
			}
			// Realiza a leitura dos alvos armazenados no arquivo CSV
			// São três valores
			for (int i = 0; i < numClasses; i++){
				fscanf(data, "%d", &inputItem.alvo[i]);
				fscanf(data, "%c", &u);
				//cout << inputItem.alvo[i] << endl;
				//cout << u << endl;
			}
			// Insere o registro no vetor de registros	
			vetorDados.insert(vetorDados.end(), inputItem);

			cont++;					
		}
	}
	fclose(data);
	cout << "> " << cont << " samples." << endl;
}

// Separa dados em Input e Output
void LibRNA::separaDados(
	const vector<struct neuralInputVector> &vetorDados,
	int numEntradas,
	int numClasses,
	Mat &input,
	Mat &output){

	// Preeche as MATRIZ input com os valores de entrada
	for (int i = 0; i < vetorDados.size(); i++)
	{
		for (int j = 0; j < numEntradas; j++){
			input.at<float>(i, j) = vetorDados[i].entradas[j];
		}
	}
	// Preeche a MATRIZ de saída com os valores dos alvos
	for (int i = 0; i < vetorDados.size(); i++)
	{
		for (int j = 0; j < numClasses; j++){
			output.at<float>(i, j) = vetorDados[i].alvo[j];
		}
	}
}

// Separa dados em Treinamento e Teste
void LibRNA::criaConjuntos(
	Mat &input,
	Mat &output,
	int numClasses,
	int numElemPorClasse,
	Mat &vetorInputTreino,
	Mat &vetorOutputTreino,
	Mat &vetorInputTeste,
	Mat &vetorOutputTeste){

	int selectedInd;

	for (int k = 0; k < numClasses; k++){

		int *Indices = (int *)malloc(numElemPorClasse*sizeof(int));
		int *mascara = (int *)calloc(numElemPorClasse, sizeof(int));

		int conta = 0;

		// Embaralhas índices de 0 a numElemPorClasse-1
		srand(time(NULL));
		while (conta < numElemPorClasse){
			selectedInd = rand() % numElemPorClasse;
			if (mascara[selectedInd] == 0)
			{
				Indices[conta] = selectedInd;
				mascara[selectedInd] = 1;
				conta++;
			}
		}

		//for (int i = 0; i < 50; i++)
		//	printf("%d \n", Indices[i]);

		// Separa as ENTRADAS em Treino e Teste
		for (int i = 0; i < numElemPorClasse / 2; i++){
			for (int j = 0; j < input.cols; j++){
				// Pega os primeiros indices embaralhados
				vetorInputTreino.at<float>(i + k*(numElemPorClasse / 2), j) = input.at<float>(Indices[i] + k*numElemPorClasse, j);
				// Pega os seguintes índices embaralhados
				vetorInputTeste.at<float>(i + k*(numElemPorClasse / 2), j) = input.at<float>(Indices[i + numElemPorClasse / 2] + k*numElemPorClasse, j);
			}
		}
		// Semara as SAIDAS em Treino e Teste
		for (int i = 0; i < numElemPorClasse / 2; i++){
			for (int j = 0; j < output.cols; j++){
				// Pega os primeiros indices embaralhados
				vetorOutputTreino.at<float>(i + k*(numElemPorClasse / 2), j) = output.at<float>(Indices[i] + k*numElemPorClasse, j);
				// Pega os seguintes índices embaralhados							

				vetorOutputTeste.at<float>(i + k*(numElemPorClasse / 2), j) = output.at<float>(Indices[i + numElemPorClasse / 2] + k*numElemPorClasse, j);
			}
		}
		free(Indices);
		free(mascara);
	}
}

void LibRNA::normalizaEntrada(
	Mat &inputOriginal,
	Mat &inputNormalizada,
	vector<float> &coef){

	// Acha os maiores valores em cada coluna da entrada
	for (int i = 0; i < inputOriginal.cols; i++){
		float Maior = 0;
		for (int j = 0; j < inputOriginal.rows; j++){
			if (inputOriginal.at<float>(j, i) > Maior)
			{
				Maior = inputOriginal.at<float>(j, i);
			}
		}
		coef.push_back(Maior);
	}

	// Realiza a normalização
	// Acha os maiores valores em cada coluna da entrada
	for (int i = 0; i < inputOriginal.cols; i++){
		for (int j = 0; j < inputOriginal.rows; j++){
			inputNormalizada.at<float>(j, i) = inputOriginal.at<float>(j, i) / coef.at(i);
		}
	}
}


void LibRNA::calDesempenho(
	Mat vetorOutputTeste,
	Mat vetorOutputPred,
	Mat & matConfusao,
	float &percentDesempenho){

	matConfusao = Mat::zeros(vetorOutputPred.cols, vetorOutputPred.cols, CV_32FC1);

	// Coloca a maior saída para 1 e as restantes para zero
	int indVenc;
	for (int i = 0; i < vetorOutputPred.rows; i++){
		int Vencedor = -1000;
		for (int j = 0; j < vetorOutputPred.cols; j++){
			if (vetorOutputPred.at<float>(i, j) >= Vencedor){
				Vencedor = vetorOutputPred.at<float>(i, j);
				indVenc = j;
			}
		}
		for (int j = 0; j < vetorOutputPred.cols; j++)
			vetorOutputPred.at<float>(i, j) = 0;
		vetorOutputPred.at<float>(i, indVenc) = 1;
	}

	// Faz a comparação entre os vetores alvo (teste) e os preditos pela rede
	int indTeste, indPred;

	for (int i = 0; i < vetorOutputPred.rows; i++){				
		for (int j = 0; j < vetorOutputPred.cols; j++) {		
			if (vetorOutputTeste.at<float>(i, j)== 1) indTeste = j; // Só pode haver apenas um único índice 1			
			if (vetorOutputPred.at<float>(i, j) == 1) indPred = j; // Só pode haver apenas um único índice 1										
		}
		matConfusao.at<float>(indTeste, indPred)++;		
	}

	// Calcula o desempenho em porcentual de acerto
	percentDesempenho = 0;
	for (int i = 0; i < vetorOutputPred.cols; i++)
		percentDesempenho += matConfusao.at<float>(i, i);

	percentDesempenho /= vetorOutputPred.rows;

}


// Armazena nos conuntos utilizados no treinamento
void LibRNA::gravaConjuntos(
	Mat &vetorInputTreino,
	Mat &vetorOutputTreino,
	Mat &vetorInputTeste,
	Mat &vetorOutputTeste, 
	string caminho){

	ofstream out_treino, out_teste;
	out_treino.open(caminho+"\\vetorTreino.csv"); // o arquivo que será criado;

	for (size_t i = 0; i < vetorInputTreino.rows; i++)
	{
		for (size_t j = 0; j < vetorInputTreino.cols; j++){
			out_treino << vetorInputTreino.at<float>(i, j) << ";";
		}
		for (size_t j = 0; j < vetorOutputTreino.cols; j++){
			out_treino << vetorOutputTreino.at<float>(i, j);
			if (j != vetorOutputTreino.cols-1) out_treino << ";";
		}
		out_treino << endl;
	}
	out_treino.close();

	out_teste.open(caminho + "\\vetorTeste.csv");

	for (size_t i = 0; i < vetorInputTeste.rows; i++)
	{
		for (size_t j = 0; j < vetorInputTeste.cols; j++){
			out_teste << vetorInputTeste.at<float>(i, j) << ";";
		}
		for (size_t j = 0; j < vetorOutputTeste.cols; j++){
			out_teste << vetorOutputTeste.at<float>(i, j);
			if (j != vetorOutputTeste.cols - 1) out_teste << ";";
		}
		out_teste << endl;
	}
	out_teste.close();
}


// Armazena nos conuntos utilizados no treinamento
void LibRNA::carregaConjuntos(
	Mat &vetorInputTreino,
	Mat &vetorOutputTreino,
	Mat &vetorInputTeste,
	Mat &vetorOutputTeste,
	string caminho){

	// entrada.
	ifstream in_treino, in_teste; // in é uma variável.
	char u;
	
	in_treino.open(caminho + "\\vetorTreino.csv"); // abre o arquivo;			
	for (int i = 0; i < vetorInputTreino.rows; i++)
	{
		for (int j = 0; j < vetorInputTreino.cols; j++){
			in_treino >> vetorInputTreino.at<float>(i, j);
			in_treino >> u;
		}
		for (int j = 0; j < vetorOutputTreino.cols; j++){
			in_treino >> vetorOutputTreino.at<float>(i, j);
			if (j != vetorOutputTreino.cols - 1) in_treino >> u;			
		}
		//in_treino >> u;
	}
	in_treino.close();
	
	in_teste.open(caminho + "\\vetorTeste.csv"); // abre o arquivo;
	for (int i = 0; i < vetorInputTeste.rows; i++)
	{
		for (int j = 0; j < vetorInputTeste.cols; j++){
			in_teste >> vetorInputTeste.at<float>(i, j);
			in_teste >> u;
			cout << vetorInputTeste.at<float>(i, j) << " ";
		}
		for (int j = 0; j < vetorOutputTeste.cols; j++){
			in_teste >> vetorOutputTeste.at<float>(i, j);
			if (j != vetorOutputTeste.cols - 1) in_teste >> u;
			//in_teste >> u;
			cout << vetorOutputTeste.at<float>(i, j) << u;
		}
		//in_teste >> u;
		cout << endl;
	}
	in_teste.close();
}

