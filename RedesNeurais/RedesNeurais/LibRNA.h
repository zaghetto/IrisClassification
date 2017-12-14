#ifndef LIBRNA_HPP
#define LIBRNA_HPP

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>

using namespace std;
using namespace cv;

// Vetor de etradas e saídas
struct neuralInputVector
{
	//double entradas[4];
	double *entradas;
	int nEntradas;
	// Sepal length (SL)  	 	
	// Sepal width  (SW)
	// Petal length (PL)	
	// Petal width  (PW)

	/* Espécie:
	1 0 0 => Setosa
	0 1 0 => Versicolor
	0 0 1 => Virginica */
	//int alvo[3];
	int nAlvos;
	int *alvo;	
};

class LibRNA
{

public:

	// Constructor
	LibRNA();

	// Destructor
	~LibRNA();

	// Operações lógicas
	int zaghand(int a, int b);
	int zaghor(int a, int b);		
	int zaghxor(int a, int b);

	// Carrega Dados
	void carregaDados(char *nomeArquivoDados, int, int, int, vector<struct neuralInputVector> &);

	// Separa Dados em input e output
	void separaDados(const vector<struct neuralInputVector> &, int, int, Mat &, Mat &);

	// Cria conjuntos de treinamento e teste
	void criaConjuntos(Mat &, Mat &, int, int, Mat &, Mat &, Mat &, Mat &);

	// Normaliza os dados de entrada
	void normalizaEntrada(Mat &, Mat &, vector<float> &);

	// Calcula a matriz de confusão
	void calDesempenho(Mat, Mat, Mat &, float &);

	// Grava as features do conjunto de teste e de treinamento 
	// Importante, pois tais conjunto são sorteados a cada treinamento
	void gravaConjuntos(Mat &, Mat &, Mat &, Mat &, string);

	// Carrega as features do conjunto de teste e de treinamento 
	void carregaConjuntos(Mat &, Mat &, Mat &, Mat &, string);

private:

};


#endif

