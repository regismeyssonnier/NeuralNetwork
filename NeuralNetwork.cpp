// NeuralNetworkCSB.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//

#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <chrono>
#include <fstream>
using namespace std::chrono;
using namespace std;

class NeuralNetwork {
public:
    double LR;
    int size_input;
    int size_hidden;
    int size_output;

    vector<double> input, hidden, output;
    vector<double> hidden_b, output_b;
    vector<vector<double>> hidden_w, output_w;
    vector<double> cost;

    vector<vector<vector<double>>> network_w;
    vector<vector<double>> network, network_b;
    vector<double> etiquette;

    NeuralNetwork(int sz_in, int sz_hid, int sz_output, double lr):size_input(sz_in),
        size_hidden(sz_hid), size_output(sz_output), LR(lr) {

        this->input.resize(this->size_input, 0.0);
        this->hidden.resize(this->size_hidden, 0.0);
        this->output.resize(this->size_output, 0.0);
        this->cost.resize(this->size_output, 0.0);

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dhidden(-0.5, 0.5);

        for (int i = 0; i < size_input; ++i) {
            hidden_w.push_back({});
            for (int j = 0; j < size_hidden; ++j) {
                hidden_w[i].push_back(dhidden(rng) );
            }
        }
       
        for (int i = 0; i < size_hidden; ++i) {
            output_w.push_back({});
            for (int j = 0; j < size_output; ++j) {
                output_w[i].push_back(dhidden(rng) );
            }
        }

        for (int i = 0; i < size_hidden; ++i) {
            hidden_b.push_back(dhidden(rng));
        }

        for (int i = 0; i < size_output; ++i) {
            output_b.push_back(dhidden(rng));
        }

        for (int i = 0; i < size_output; ++i) {
            cout << output_b[i] << endl;
        }

    }

    NeuralNetwork(vector<int>dimension, double LR) {
        this->LR = LR;
        cerr << "init " << endl;
        
        for (int i = 0; i < dimension.size(); ++i) {
            vector<double> dim(dimension[i], 0.0);
            this->network.push_back(dim);

            if(i > 0)
                this->network_b.push_back(dim);

            if (i < dimension.size() - 1) {
                vector<vector<double>> dim2(dimension[i], vector<double>(dimension[i + 1], 0.0));
                this->network_w.push_back(dim2);
            }
                       

        }
        cerr << "init 2" << endl;
        this->cost.resize(dimension[dimension.size() - 1], 0.0);

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dhidden(-0.5, 0.5);

        for (int i = 0; i < network_w.size(); ++i) {
            for (int j = 0; j < network_w[i].size(); ++j) {
                for (int k = 0; k < network_w[i][j].size(); ++k) {
                    network_w[i][j][k] = dhidden(rng);
                }

            }

        }
        cerr << "init 3" << endl;
        for (int i = 0; i < network_b.size(); ++i) {
            for (int j = 0; j < network_b[i].size(); ++j) {
                network_b[i][j] = dhidden(rng);
            }
        }
        cerr << "init 4" << endl;

    }

    void SetEtiquette(vector<double> et) {
        this->etiquette.swap(et);
    }

    void Set_InputNet(vector<double> inp) {
        this->network[0].swap(inp);
    }

    void ForwardNN() {

        /*for (int i = 0; i < size_hidden; ++i) {
            double h = 0;
            for (int j = 0; j < size_input; ++j) {
                h += input[j] * hidden_w[j][i];
            }
            h += hidden_b[i];
            hidden[i] = Relu(h);
        }*/

        //cerr << "start f" << network.size() << endl;
        for (int i = 0; i < network.size()-2; ++i) {
           // cerr << "i " << i << endl;
            for (int j = 0; j < network[i+1].size(); ++j) {
              // cerr << "j " << j << " " << network[i + 1].size() << endl;
                double h = 0.0;
                for (int k = 0; k < network[i].size(); ++k) {
                   // cerr << "k " << k << " " << network[i].size() << endl;
                    h += network[i][k] * network_w[i][k][j];
                }
                //cerr << "endf1 " << endl;
                h += network_b[i][j];
                network[i+1][j] = this->Relu(h);
               // cerr << "endf2 " << endl;
            }

        }

        int ind = network.size() - 2;
        for (int j = 0; j < network[ind + 1].size(); ++j) {
            // cerr << "j " << j << " " << network[i + 1].size() << endl;
            double h = 0.0;
            for (int k = 0; k < network[ind].size(); ++k) {
                // cerr << "k " << k << " " << network[i].size() << endl;
                h += network[ind][k] * network_w[ind][k][j];
            }
            //cerr << "endf1 " << endl;
            h += network_b[ind][j];
            network[ind + 1][j] = this->sigmoid(h);
            // cerr << "endf2 " << endl;
        }



        //cerr << "start f2" << endl;
        for (int i = 0; i < network[network.size() - 1].size(); ++i) {
            cost[i] += network[network.size() - 1][i] - this->etiquette[i];
        }
        //cerr << "start f3" << endl;

    }

    void BackwardNN() {
        //cerr << "start Back " << endl;
        vector<vector<double>> result(network.size()-1);
        for (int i = 0; i < network[network.size() - 1].size(); ++i) {
            result[result.size()-1].push_back(cost[i] * sigmoid_derivative(network[network.size() - 1][i]));
            network_b[network_b.size() - 1][i] -= LR * result[result.size() - 1][i];
        }

        //cerr << " Back1 " << endl;

        for (int i = result.size() - 2; i >= 0; --i) {
            //cerr << i << endl;
            for (int j = 0; j < network[i+1].size(); ++j) {
               // cerr << "j " << j << " " << network[i + 1].size() << endl;
                result[i].push_back(0.0);
                for (int k = 0; k < network[i+2].size(); ++k) {
                    //cerr << "k " << k << " " << network[i + 2].size() << endl;
                    result[i][j] += network_w[i+1][j][k] * result[i+1][k] * this->dRelu(network[i+1][j]);
                }
                network_b[i][j] -= LR * result[i][j];
            }

        }

        //cerr << " Back2 " << endl;
        for (int i = network.size() - 2; i >= 0; --i) {
            double h = 0.0;
            for (int j = 0; j < network[i].size(); ++j) {
                for (int k = 0; k < network[i + 1].size(); ++k) {
                    network_w[i][j][k] -= LR * network[i][j] * result[i][k];
                }
                
            }

        }

        //cerr << " Back3 " << endl;
        cost = {};
        cost.resize(network[network.size() - 1].size(), 0.0);

        /*

        vector<double> result(size_hidden, 0.0);

        // = dotProduct(output_w, resultout);

        for (int i = 0; i < size_hidden; ++i) {
            for (int j = 0; j < size_output; ++j) {
                result[i] += output_w[i][j] * resultout[j] * this->dRelu(hidden[i]);
            }
        }

        //for (int i = 0; i < result.size(); ++i)result[i] *= this->getDTanh(hidden[i]);
        //cout << "result size " << result.size() << endl;

        for (int i = 0; i < size_hidden; ++i) {
            hidden_b[i] -= LR * result[i];
        }

        //vector<vector<double>> dw = outerProduct(hidden, resultout);
        //updateWeights(output_w, dw);
        for (int i = 0; i < size_hidden; ++i) {
            for (int j = 0; j < size_output; ++j) {
                output_w[i][j] -= LR * hidden[i] * resultout[j];

            }
        }



        //vector<vector<double>> dw2 = outerProduct(input, result);
        //updateWeights(hidden_w, dw2);
        for (int i = 0; i < size_input; ++i) {
            for (int j = 0; j < size_hidden; ++j) {
                hidden_w[i][j] -= LR * input[i] * result[j];
            }
        }
        */

        



    }


    void Set_Input(vector<double> inp) {
        input.swap(inp);
    }

    double getTanh(double v) {
        return tanh(v);
    }

    double getDTanh(double v) {
        return 1.0 - tanh(v)* tanh(v);
    }

    double dRelu(double v) {
        //return 1.0;
        if (v > 0.0)return 1.0;
        else return 0.0;
    }

    double Relu(double v) {
        //return 1.0;
        return max(0.0, v);
    }


    void Softmaxdep(const std::vector<double>& x, vector<double> &out) {
        std::vector<double> exp_x;
        double sum_exp_x = 0.0;

        // Calculer les exponentielles des éléments du vecteur
        for (const double& xi : x) {
            exp_x.push_back(std::exp(xi));
            sum_exp_x += std::exp(xi);
        }

        // Calculer le softmax
        int ind = 0;
        for (const double& exi : exp_x) {
            out[ind] = exi / sum_exp_x;
            ++ind;
        }

       
    }

    void Softmax(const std::vector<double>& x, std::vector<double>& out) {
        double max_x = *std::max_element(x.begin(), x.end()); // Trouver la valeur maximale de x

        double sum_exp_x = 0.0;

        // Calculer les exponentielles des éléments du vecteur
        for (const double& xi : x) {
            sum_exp_x += std::exp(xi - max_x); // Soustraire max_x
        }

        // Calculer le softmax
        for (size_t i = 0; i < x.size(); i++) {
            out[i] = std::exp(x[i] - max_x) / sum_exp_x; // Soustraire max_x
        }
    }

    void SoftmaxDerivative(const std::vector<double>& x, std::vector<std::vector<double>>& out) {
        int size = x.size();
        out.resize(size, std::vector<double>(size, 0.0));

        std::vector<double> softmax_values(size, 0.0);
        double sum_exp_x = 0.0;

        // Calculer les exponentielles des éléments du vecteur et le softmax
        for (const double& xi : x) {
            sum_exp_x += std::exp(xi);
        }

        for (int i = 0; i < size; i++) {
            softmax_values[i] = std::exp(x[i]) / sum_exp_x;
        }

        // Calculer la dérivée de softmax
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i == j) {
                    out[i][j] = softmax_values[i] * (1 - softmax_values[i]);
                }
                else {
                    out[i][j] = -softmax_values[i] * softmax_values[j];
                }
            }
        }
    }

    // Appliquer la fonction sigmoïde à un tableau de valeurs
    void sigmoidtd(const std::vector<double>& x, vector<double> &result) {
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = 1.0 / (1.0 + std::exp(-x[i]));
        }
        
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoid_derivative(double x) {
        return x * (1.0 - x);
    }
   
    // Appliquer la dérivée de la sigmoïde à un tableau de valeurs
    std::vector<double> sigmoid_derivativetd(const std::vector<double>& x, vector<double> &result) {
     
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = x[i] * (1.0 - x[i]);
        }
       
    }
    
    void Forward(vector<double> target) {
        for (int i = 0; i < size_hidden; ++i) {
            double h = 0;
            for (int j = 0; j < size_input; ++j) {
                h += input[j] * hidden_w[j][i];
            }
            h += hidden_b[i];
            hidden[i] = getTanh(h);
        }
        //cerr << 1 << endl;
        vector<double> outputbis(size_output, 0.0);
        for (int i = 0; i < size_output; ++i) {
            double h = 0;
            for (int j = 0; j < size_hidden; ++j) {
                h += hidden[j] * output_w[j][i];
            }
            h += output_b[i];
            outputbis[i] = h;
        }
        //cerr << 2 << endl;
        sigmoidtd(outputbis, output);
        //cerr << 3 << endl;
        for (int i = 0;i < size_output; ++i) {
            cost[i] += output[i] - target[i];
        }
        //cerr << 4 << endl;

    }

    void ForwardR(vector<double> target) {
        for (int i = 0; i < size_hidden; ++i) {
            double h = 0;
            for (int j = 0; j < size_input; ++j) {
                h += input[j] * hidden_w[j][i];
            }
            h += hidden_b[i];
            hidden[i] = Relu(h);
        }
        //cerr << 1 << endl;
        vector<double> outputbis(size_output, 0.0);
        for (int i = 0; i < size_output; ++i) {
            double h = 0;
            for (int j = 0; j < size_hidden; ++j) {
                h += hidden[j] * output_w[j][i];
            }
            h += output_b[i];
            outputbis[i] = h;
            //output[i] = h;
        }
        //cerr << 2 << endl;
        sigmoidtd(outputbis, output);
        //cerr << 3 << endl;
        for (int i = 0; i < size_output; ++i) {
            cost[i] += output[i] - target[i];
        }
        //cerr << 4 << endl;

    }

    void ForwardT(vector<double> target) {
        for (int i = 0; i < size_hidden; ++i) {
            double h = 0;
            for (int j = 0; j < size_input; ++j) {
                h += input[j] * hidden_w[j][i];
            }
            h += hidden_b[i];
            hidden[i] = getTanh(h);
        }
        //cerr << 1 << endl;
        vector<double> outputbis(size_output, 0.0);
        for (int i = 0; i < size_output; ++i) {
            double h = 0;
            for (int j = 0; j < size_hidden; ++j) {
                h += hidden[j] * output_w[j][i];
            }
            h += output_b[i];
            outputbis[i] = h;
            
        }
        //cerr << 2 << endl;
        sigmoidtd(outputbis, output);
        //cerr << 3 << endl;
        for (int i = 0; i < size_output; ++i) {
            cost[i] += output[i] - target[i];
        }
        //cerr << 4 << endl;

    }

    void ForwardS(vector<double> target) {
        for (int i = 0; i < size_hidden; ++i) {
            double h = 0;
            for (int j = 0; j < size_input; ++j) {
                h += input[j] * hidden_w[j][i];
            }
            h += hidden_b[i];
            hidden[i] = getTanh(h);
        }
        //cerr << 1 << endl;
        vector<double> outputbis(size_output, 0.0);
        for (int i = 0; i < size_output; ++i) {
            double h = 0;
            for (int j = 0; j < size_hidden; ++j) {
                h += hidden[j] * output_w[j][i];
            }
            h += output_b[i];
            outputbis[i] = h;
        }
        //cerr << 2 << endl;
        Softmax(outputbis, output);
        //cerr << 3 << endl;
        for (int i = 0; i < size_output; ++i) {
            cost[i] += output[i] - target[i];
        }
        //cerr << 4 << endl;

    }

    std::vector<std::vector<double>> outerProduct(const std::vector<double>& array1, const std::vector<double>& array2) {
        int size1 = array1.size();
        int size2 = array2.size();

        // Créer une matrice pour stocker le produit tensoriel
        std::vector<std::vector<double>> result(size1, std::vector<double>(size2, 0.0));

        // Calculer le produit tensoriel
        for (int i = 0; i < size1; i++) {
            for (int j = 0; j < size2; j++) {
                result[i][j] = array1[i] * array2[j];
            }
        }

        return result;
    }

    void updateWeights(std::vector<std::vector<double>>& weights, const std::vector<std::vector<double>>& gradient) {
        int numRows = weights.size();
        int numCols = weights[0].size();

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                weights[i][j] -= LR * gradient[i][j];
            }
        }
    }

    std::vector<double> dotProduct(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector) {
        int numRows = matrix.size();
        int numCols = matrix[0].size();

        if (numCols != vector.size()) {
            throw std::invalid_argument("Incompatible dimensions for dot product");
        }

        std::vector<double> result(numRows, 0.0);

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }

        return result;
    }

    void Backward() {

        for (int i = 0; i < size_output; ++i) {
            output_b[i] -= LR * cost[i];
        }

        vector<double> resultout;
        for (int i = 0; i < size_output; ++i) {
            resultout.push_back(cost[i] * sigmoid_derivative(output[i]));
        }
              
        vector<double> result = dotProduct(output_w, resultout);
           
        for (int i = 0; i < result.size(); ++i)result[i] *= this->getDTanh(hidden[i]);
        //cout << "result size " << result.size() << endl;
        for (int i = 0; i < size_hidden; ++i) {
            hidden_b[i] -= LR * result[i];
        }

        vector<vector<double>> dw2 = outerProduct(input, result);
        updateWeights(hidden_w, dw2);

        vector<vector<double>> dw = outerProduct(hidden, resultout);
        updateWeights(output_w, dw);

        cost = {};
        cost.resize(size_output, 0.0);

    }

    void BackwardF() {

        

        vector<double> resultout;
        for (int i = 0; i < size_output; ++i) {
            resultout.push_back(cost[i] * sigmoid_derivative(output[i]));
        }

        for (int i = 0; i < size_output; ++i) {
            output_b[i] -= LR * resultout[i];
        }

        vector<double> result(size_hidden, 0.0);
            
        // = dotProduct(output_w, resultout);

        for (int i = 0; i < size_hidden; ++i) {
            for (int j = 0; j < size_output; ++j) {
                result[i] += output_w[i][j] * resultout[j] * this->getDTanh(hidden[i]);
            }
        }

        //for (int i = 0; i < result.size(); ++i)result[i] *= this->getDTanh(hidden[i]);
        //cout << "result size " << result.size() << endl;
        
        for (int i = 0; i < size_hidden; ++i) {
            hidden_b[i] -= LR * result[i];
        }

        //vector<vector<double>> dw = outerProduct(hidden, resultout);
        //updateWeights(output_w, dw);
        for (int i = 0; i < size_hidden; ++i) {
            for (int j = 0; j < size_output; ++j) {
                output_w[i][j] -= LR * hidden[i] * resultout[j];
               
            }
        }



        //vector<vector<double>> dw2 = outerProduct(input, result);
        //updateWeights(hidden_w, dw2);
        for (int i = 0; i < size_input; ++i) {
            for (int j = 0; j < size_hidden; ++j) {
                hidden_w[i][j] -= LR * input[i] * result[j];
            }
        }
                

        cost = {};
        cost.resize(size_output, 0.0);

    }

    void BackwardFR() {



        vector<double> resultout;
        for (int i = 0; i < size_output; ++i) {
            resultout.push_back(cost[i] * sigmoid_derivative  (output[i]));
        }

        for (int i = 0; i < size_output; ++i) {
            output_b[i] -= LR * resultout[i];
        }

        vector<double> result(size_hidden, 0.0);

        // = dotProduct(output_w, resultout);

        for (int i = 0; i < size_hidden; ++i) {
            for (int j = 0; j < size_output; ++j) {
                result[i] += output_w[i][j] * resultout[j] * this->dRelu(hidden[i]);
            }
        }

        //for (int i = 0; i < result.size(); ++i)result[i] *= this->getDTanh(hidden[i]);
        //cout << "result size " << result.size() << endl;

        for (int i = 0; i < size_hidden; ++i) {
            hidden_b[i] -= LR * result[i];
        }

        //vector<vector<double>> dw = outerProduct(hidden, resultout);
        //updateWeights(output_w, dw);
        for (int i = 0; i < size_hidden; ++i) {
            for (int j = 0; j < size_output; ++j) {
                output_w[i][j] -= LR * hidden[i] * resultout[j];

            }
        }



        //vector<vector<double>> dw2 = outerProduct(input, result);
        //updateWeights(hidden_w, dw2);
        for (int i = 0; i < size_input; ++i) {
            for (int j = 0; j < size_hidden; ++j) {
                hidden_w[i][j] -= LR * input[i] * result[j];
            }
        }


        cost = {};
        cost.resize(size_output, 0.0);

    }

    void BackwardFT() {



        vector<double> resultout;
        for (int i = 0; i < size_output; ++i) {
            resultout.push_back(cost[i] * sigmoid_derivative(output[i]));
        }

        for (int i = 0; i < size_output; ++i) {
            output_b[i] -= LR * resultout[i];
        }

        vector<double> result(size_hidden, 0.0);

        // = dotProduct(output_w, resultout);

        for (int i = 0; i < size_hidden; ++i) {
            for (int j = 0; j < size_output; ++j) {
                result[i] += output_w[i][j] * resultout[j] * this->getDTanh(hidden[i]);
            }
        }

        //for (int i = 0; i < result.size(); ++i)result[i] *= this->getDTanh(hidden[i]);
        //cout << "result size " << result.size() << endl;

        for (int i = 0; i < size_hidden; ++i) {
            hidden_b[i] -= LR * result[i];
        }

        //vector<vector<double>> dw = outerProduct(hidden, resultout);
        //updateWeights(output_w, dw);
        for (int i = 0; i < size_hidden; ++i) {
            for (int j = 0; j < size_output; ++j) {
                output_w[i][j] -= LR * hidden[i] * resultout[j];

            }
        }



        //vector<vector<double>> dw2 = outerProduct(input, result);
        //updateWeights(hidden_w, dw2);
        for (int i = 0; i < size_input; ++i) {
            for (int j = 0; j < size_hidden; ++j) {
                hidden_w[i][j] -= LR * input[i] * result[j];
            }
        }


        cost = {};
        cost.resize(size_output, 0.0);

    }


    void BackwardS() {

        for (int i = 0; i < size_output; ++i) {
            output_b[i] -= LR * cost[i];
        }

       /* std::vector<double> softmax_derivative(size_output, 0.0);
        // Calcul de la dérivée de softmax pour chaque sortie
        for (int i = 0; i < size_output; ++i) {
            softmax_derivative[i] = output[i] * (1.0 - output[i]);
        }*/

        vector<double> result = dotProduct(output_w, cost);
        
        /*std::vector<double> result(size_hidden, 0.0);
        for (int i = 0; i < size_hidden; ++i) {
            for (int j = 0; j < size_output; ++j) {
                result[i] += output_w[i][j] * softmax_derivative[j];
            }
        }*/

        for (int i = 0; i < result.size(); ++i)result[i] *= this->getDTanh(hidden[i]);
        //cout << "result size " << result.size() << endl;
        for (int i = 0; i < size_hidden; ++i) {
            hidden_b[i] -= LR * result[i];
        }

        vector<vector<double>> dw2 = outerProduct(input, result);
        updateWeights(hidden_w, dw2);

        vector<vector<double>> dw = outerProduct(hidden, cost);
        updateWeights(output_w, dw);

        cost = {};
        cost.resize(size_output, 0.0);

    }

    void Training(vector<vector<double>>inp, vector<vector<double>> target, int nbi, int nb) {

        for (int i = 0; i < nb; ++i) {
            for (int j = 0; j < nbi; ++j) {
                this->Set_Input(inp[j]);
                this->Forward(target[j]);
                //if (i % (nb / 4) == 0) {
                this->BackwardF();

                if ((i + 1) % 1000 == 0) {
                    vector<double> cost_(size_output, 0);
                    for (int k = 0; k < nbi; ++k) {
                        this->Set_Input(inp[k]);
                        this->Forward(target[k]);
                        for (int l = 0; l < size_output; ++l) {
                            cost_[l] += (output[l] - target[k][l]) * (output[l] - target[k][l]);
                        }
                    }
                    for (int l = 0; l < size_output; ++l) {
                        cost_[l] /= nbi;
                    }
                    for (int l = 0; l < size_output; ++l) {
                        cout << "error" << l << ": " << cost_[l] << endl;
                    }
                }


                //}

            }
        }

    }

    void TrainingR(vector<vector<double>>inp, vector<vector<double>> target, int nbi, int nb) {
        
        vector<int>indexes;
        for (int j = 0; j < nbi; ++j) {
            indexes.push_back(j);
        }

        for (int i = 0; i < nb; ++i) {

            random_shuffle(indexes.begin(), indexes.end());

            for (int jj = 0; jj < nbi; ++jj) {
                int j = indexes[jj];
                this->Set_Input(inp[j]);
                this->ForwardR(target[j]);
                //if (i % (nb / 4) == 0) {
                    this->BackwardFR();

                if ((i + 1) % 1000 == 0) {
                    vector<double> cost_(size_output, 0);
                    for (int k = 0; k < nbi; ++k) {
                        this->Set_Input(inp[k]);
                        this->ForwardR(target[k]);
                        for (int l = 0; l < size_output; ++l) {
                            cost_[l] += (output[l] - target[k][l])* (output[l] - target[k][l]);
                        }
                    }
                    /*for (int l = 0; l < size_output; ++l) {
                        cost_[l] /= nbi;
                    }*/
                    for (int l = 0; l < size_output; ++l) {
                        cout << "error"<< l <<": " << cost_[l] << endl;
                    }
                }


                //}

            }
        }
        
    }

    void TrainingNN(vector<vector<double>>inp, vector<vector<double>> target, int nbi, int nb) {

        vector<int>indexes;
        for (int j = 0; j < nbi; ++j) {
            indexes.push_back(j);
        }

        for (int i = 0; i < nb; ++i) {

            random_shuffle(indexes.begin(), indexes.end());

            for (int jj = 0; jj < nbi; ++jj) {
                int j = indexes[jj];
                this->Set_InputNet(inp[j]);
                this->SetEtiquette(target[j]);
                this->ForwardNN();
                this->BackwardNN();

                if ((i + 1) % 1000 == 0) {
                    vector<double> cost_(network[network.size() - 1].size(), 0);
                    for (int k = 0; k < nbi; ++k) {
                        this->Set_InputNet(inp[k]);
                        this->SetEtiquette(target[k]);
                        this->ForwardNN();
                        for (int l = 0; l < network[network.size() - 1].size(); ++l) {
                            cost_[l] += (network[network.size() - 1][l] - target[k][l]) * (network[network.size() - 1][l] - target[k][l]);
                        }
                    }
                    /*for (int l = 0; l < size_output; ++l) {
                        cost_[l] /= nbi;
                    }*/
                    /*for (int l = 0; l < network[network.size() - 1].size(); ++l) {
                        cout << "output " << l << ": " << network[network.size() - 1][l] << endl;
                    }
                    for (int l = 0; l < network[network.size() - 1].size(); ++l) {
                        cout << "error" << l << ": " << cost_[l] << endl;
                    }*/
                }


                //}

            }
        }

    }


    void TrainingS(vector<vector<double>>inp, vector<vector<double>> target, int nbi, int nb) {

        for (int i = 0; i < nb; ++i) {
            for (int j = 0; j < nbi; ++j) {
                this->Set_Input(inp[j]);
                this->ForwardS(target[j]);
                //if (i % (nb / 4) == 0) {
                this->BackwardS();

                if ((i + 1) % 100000 == 0) {
                    vector<double> cost_(size_output, 0);
                    for (int k = 0; k < nbi; ++k) {
                        this->Set_Input(inp[k]);
                        this->ForwardS(target[k]);
                        for (int l = 0; l < size_output; ++l) {
                            cost_[l] += (output[l] - target[k][l]) * (output[l] - target[k][l]);
                        }
                    }
                    for (int l = 0; l < size_output; ++l) {
                        cost_[l] /= nbi;
                    }
                    for (int l = 0; l < size_output; ++l) {
                        cout << "error" << l << ": " << cost_[l] << endl;
                    }
                }


                //}

            }
        }

    }


    void TrainDataset(string fn) {

        
        int nb, nh, no;
        double a1, a2, a3, a4, a5, a6, a7, a8;
        
        for (int x = 0; x < 1; ++x) {

            ifstream f(fn);
            f >> nb >> nh >> no;
            cerr << nb << endl;
            vector<double>target(size_output);

            for (int i = 0; i < nb; ++i) {
                f >> a1 >> a2 >> a3 >> a4 >> a5 >> a6;// >> a7 >> a8;
                f >> target[0] >> target[1];
                vector<double >inp = { a1, a2, a3, a4, a5, a6 };
                this->Set_Input({ a1, a2, a3, a4, a5, a6 });
                this->ForwardR(target);
                this->BackwardFR();
                if ((i + 1) % 100000 == 0) {
                    cerr << i << endl;
                    vector<double> cost_(size_output, 0);
                    //a1 = 3.85502e-07; a2 = 1.47627e-07; a3 = 2.2316e-05; a4 = 1.04742e-05; a5 = 3.01223e-05; a6 = 7.63634e-06; a7 = 2.96141e-05; a8 = 1.03546e-05;
                    
                    this->Set_Input(inp);
                    this->ForwardR(target);
                    for (int l = 0; l < size_output; ++l) {
                        cost_[l] += (output[l] - target[l]) * (output[l] - target[l]);
                    }

                    //cerr << "target " << target[0] << " " << target[1] << endl;
                    for (int l = 0; l < size_output; ++l) {
                        cout << "error" << l << ": " << cost_[l] << endl;
                    }
                    /*for (int l = 0; l < size_output; ++l) {
                        cout << output[l] << " ";
                    }cout << endl;*/
                }


            }


            f.close();

        }


    }

    void InitByDump(vector<vector<double>> hw,
        vector<vector<double>> ow,
        vector<double> ob,
        vector<double> hb
    ) {

        this->hidden_w.swap(hw);
        this->output_w.swap(ow);
        this->output_b.swap(ob);
        this->hidden_b.swap(hb);

    }

    void DumpCpp() {

        cout << "/*hidden weight: */" << endl;
        cout << "{";
        for (int i = 0; i < size_input; ++i) {
            cout << "{";
            for (int j = 0; j < size_hidden; ++j) {
                cout << hidden_w[i][j];
                if (j < size_hidden - 1)cout << ",";
            }
            cout << "}";
            if (i < size_input - 1)cout << ",";
        }
        cout << "},\n";

        cout << "/*output weight: */" << endl;
        cout << "{";
        for (int i = 0; i < size_hidden; ++i) {
            cout << "{";
            for (int j = 0; j < size_output; ++j) {
                cout << output_w[i][j];
                if (j < size_output - 1)cout << ",";
            }
            cout << "}";
            if (i < size_hidden - 1)cout << ",";
        }
        cout << "},\n";

        cout << "/*output bias : */" << endl;
        cout << "{";
        for (int i = 0; i < size_output; ++i) {
            cout << output_b[i];
            if (i < size_output - 1)cout << ",";
        }
        cout << "},\n";

        cout << "/*hidden bias : */" << endl;
        cout << "{";
        for (int i = 0; i < size_hidden; ++i) {
            cout << hidden_b[i];
            if (i < size_hidden - 1)cout << ",";
        }
        cout << "}";


    }



    void DisplayOutput() {

        /*cout << "hidden weight: " << endl;
        for (int i = 0; i < size_input; ++i) {
            for (int j = 0; j < size_hidden; ++j) {
                cout << hidden_w[i][j] << endl;
            }
        }

        cout << "output weight: " << endl;
        for (int i = 0; i < size_hidden; ++i) {
            for (int j = 0; j < size_output; ++j) {
                cout << output_w[i][j] << endl;
            }
        }*/

        cout << "output : " << endl;
        for (int i = 0; i < size_output; ++i) {
            cout << i << " " << output[i] << endl;
        }
    }

    void DisplayOutputNN() {

        /*for (int i = 1; i < network_w.size(); ++i) {
            cerr << "weight layer " << i << endl;
            for (int j = 0; j < network_w[i].size(); ++j) {
                for (int k = 0; k < network_w[i][j].size(); ++k) {
                    cerr << network_w[i][j][k] << endl;
                }
            }
        }

        for (int j = 0; j < network_b.size(); ++j) {
            cerr << "bias layer " << j << endl;
            for (int k = 0; k < network_b[j].size(); ++k) {
                cerr << network_b[j][k] << endl;
            }
        }*/

        cout << "output : " << endl;
        for (int i = 0; i < network[network.size()-1].size(); ++i) {
            cout << i << " " << network[network.size() - 1][i] << endl;
        }
    }




};

int main()
{
    vector<int> dimension = { 2, 8, 2 };
    NeuralNetwork nn = NeuralNetwork(dimension, 0.1);

    vector<vector<double>> input = {
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 },
    };

    vector<vector<double>> output = {
        { 1, 0 },
        { 0, 1 },
        { 0, 1 },
        { 1, 0 },
    };

    auto start = high_resolution_clock::now();

    //nn.TrainDataset("dataset.txt");
    //nn.DisplayOutput();
    //nn.DumpCpp();
    //nn.InitByDump();

    //nn.DumpCpp();
    nn.TrainingNN(input, output, 4, 10000);
    
    nn.Set_InputNet({ 0, 1 });
    nn.ForwardNN();
    nn.DisplayOutputNN();
    nn.Set_InputNet({ 0, 0 });
    nn.ForwardNN();
    nn.DisplayOutputNN();
    nn.Set_InputNet({ 1, 1 });
    nn.ForwardNN();
    nn.DisplayOutputNN();
    nn.Set_InputNet({ 1, 0 });
    nn.ForwardNN();
    nn.DisplayOutputNN();

    

    /*vector<double>fwd = { 0,0 };
    nn.Set_Input({ 0, 1 });
    nn.ForwardR(fwd);
    nn.DisplayOutput();
    nn.Set_Input({ 1, 0 });
    nn.ForwardR(fwd);
    nn.DisplayOutput();
    nn.Set_Input({ 0, 0 });
    nn.ForwardR(fwd);
    nn.DisplayOutput();
    nn.Set_Input({ 1, 1 });
    nn.ForwardR(fwd);
    nn.DisplayOutput();*/

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << "ms " << endl;
        
}

// Exécuter le programme : Ctrl+F5 ou menu Déboguer > Exécuter sans débogage
// Déboguer le programme : F5 ou menu Déboguer > Démarrer le débogage

// Astuces pour bien démarrer : 
//   1. Utilisez la fenêtre Explorateur de solutions pour ajouter des fichiers et les gérer.
//   2. Utilisez la fenêtre Team Explorer pour vous connecter au contrôle de code source.
//   3. Utilisez la fenêtre Sortie pour voir la sortie de la génération et d'autres messages.
//   4. Utilisez la fenêtre Liste d'erreurs pour voir les erreurs.
//   5. Accédez à Projet > Ajouter un nouvel élément pour créer des fichiers de code, ou à Projet > Ajouter un élément existant pour ajouter des fichiers de code existants au projet.
//   6. Pour rouvrir ce projet plus tard, accédez à Fichier > Ouvrir > Projet et sélectionnez le fichier .sln.
