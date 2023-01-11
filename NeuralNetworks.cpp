// NeuralNetworks.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
using namespace std;



struct NN1 {

    vector<double> input, hidden, bias, hiddeno, biaso, Eb, Ebo;
    vector<vector<double>>weight, weighto, Ew, Ewo;
    vector<double> inputb, hiddenb, biasb, hiddenob, biasob;
    vector<vector<double>>weightb, weightob;
    int SZI, SZH, SZO, SZW, SZWO;
    double LR = 0.0009, beta = 0.9, LRB = 0.00009, LRT = 0.00000001;
    NN1(int inp, int hl, int out) {
        SZI = inp;
        SZH = hl;
        SZW = SZI * SZH;
        SZWO = out * hl;
        SZO = out;

        hidden.resize(SZH);
        hiddeno.resize(SZO);
        Eb.resize(SZH, 0);
        Ebo.resize(SZO, 0);
       
        srand(time(NULL));

        for (int i = 0; i < SZI; ++i) {
            weight.push_back({});
            Ew.push_back({});
            for (int j = 0; j < SZH; ++j) {
                weight[i].push_back((( (double)(rand() % 40) + 1.0) - 0.0) / 100.0 );
                Ew[i].push_back(0);
            }
        }

        for (int i = 0; i < SZH; ++i) {
            weighto.push_back({});
            Ewo.push_back({});
            for (int j = 0; j < SZO; ++j) {
                weighto[i].push_back((((double)(rand() % 40) + 1.0) - 0.0) / 100.0 );
                Ewo[i].push_back(0);
            }
        }
        biaso.push_back((((double)(rand() % 40) + 1.0) - 0.0) / 100.0 );

        for (int i = 0; i < SZH; ++i) {
            bias.push_back((((double)(rand() % 40) + 1.0) - 0.0) / 100.0 );
            //biaso[i] = ((rand() % 4) + 1) - 2;
        }

    }

    void build(vector<double> inp) {
        input.swap(inp);
    }

    void Copy(vector<double> s, vector<double> d) {
        d.resize(s.size());
        for (int i = 0; i < s.size(); ++i) {
            d[i] = s[i];
        }
    }

    void Copy2(vector<vector<double>> s, vector<vector<double>> d) {
        d.resize(s.size());
        for (int i = 0; i < s.size(); ++i) {
            d[i].resize(s[i].size());
            for (int j = 0; j < s[i].size(); ++i) {
                d[i][j] = s[i][j];
            }
            
        }
    }

    void Swap() {
        //Copy(input, inputb);
        Copy(hidden, hiddenb);
        Copy(hiddeno, hiddenob);
        Copy2(weight, weightb);
        Copy2(weighto, weightob);
        Copy(bias, biasb);
        Copy(biaso, biasob);
    }

    void Swap2() {
        //Copy(inputb, input);
        Copy(hiddenb, hidden);
        Copy(hiddenob, hiddeno);
        Copy2(weightb, weight);
        Copy2(weightob, weighto);
        Copy(biasb, bias);
        Copy(biasob, biaso);
    }

    void Compute() {
       // cout << "bug 1 " << endl;
        //hidden
        for (int i = 0; i < SZH; ++i) {
            int h = 0;
            for (int j = 0; j < SZI; ++j) {
                h += input[j] * weight[j][i];
            }
            h += bias[i];
            if (h < 0)h = 0;
            hidden[i] = h;
            
        }
        //cout << "bug 2 " << endl;
        
        //out
        for (int i = 0; i < SZO; ++i) {
            int h = 0;
            for (int j = 0; j < SZH; ++j) {
                h += hidden[j] * weighto[j][i];
            }
            h += biaso[0];
            if (h < 0)h = 0;
            hiddeno[i] = h;

        }

        //cout << "bug 3 " << endl;
    }

    double dRelu(double v) {
        //return 1.0;
        if (v > 0.0)return 1.0;
        else return 0.0;
    }
    

    void Backpropagation1(int target) {

        int cost = 2 * (hiddeno[0] - target );
        //cout << "bug b1 " << endl;
        for (int j = 0; j < SZH; ++j) {
            for (int i = 0; i < SZO; ++i) {
                weighto[j][i] -= LRT * cost * hidden[j];
            }
        }
        
        biaso[0] -= LRT * cost;
        //cout << "bug b2 " << endl;
        
        for (int j = 0; j < SZI; ++j) {
            for (int i = 0; i < SZH; ++i) {
                for (int jo = 0; jo < SZH; ++jo) {
                    for (int io = 0; io < SZO; ++io) {
                        weight[j][i] -= LRT * cost * weighto[jo][io] * dRelu(hidden[i]) * input[j];
                    }
                }
            }
        }
        //cout << "bug b3 " << endl;
        for (int jo = 0; jo < SZH; ++jo) {
            for (int io = 0; io < SZO; ++io) {

                for (int i = 0; i < SZH; ++i) {
                    bias[i] -= LRT * cost * weighto[jo][io] * dRelu(hidden[i]) ;
                }
                
            }
        }
        //cout << "bug b4 " << endl;


    }

    void BackpropagationRMSprop(int target) {

        int cost = 2 * (hiddeno[0] - target);
        //cout << "bug b1 " << endl;
        for (int j = 0; j < SZH; ++j) {
           
            for (int i = 0; i < SZO; ++i) {
                double dw = LR * cost * hidden[j];
                Ewo[j][i] = beta * Ewo[j][i] + (1 - beta) * dw * dw;
                //cout << "Ewo :" << Ewo[j][i] << " " << (LR / sqrt(Ewo[j][i])) * dw;
                double sqe = sqrt(Ewo[j][i]), lre;
                if (sqe == 0)lre = 0;
                else lre = LR / sqe;
                weighto[j][i] -= lre * dw;
            }//cout << endl;
            
        }

        double db = LRB * cost;
        Ebo[0] = beta * Ebo[0] + (1 - beta) * db;
        biaso[0] -= Ebo[0];
        //cout << "bug b2 " << endl;

        for (int j = 0; j < SZI; ++j) {
            for (int i = 0; i < SZH; ++i) {
                for (int jo = 0; jo < SZH; ++jo) {
                    for (int io = 0; io < SZO; ++io) {
                        double dw = LR * cost * weighto[jo][io] * dRelu(hidden[i]) * input[j];
                        Ew[j][i] = beta * Ew[j][i] + (1 - beta) * dw * dw;
                        //cout << "Ew :" << Ew[j][i] << " " << (LR / sqrt(Ew[j][i])) * dw;
                        double sqe = sqrt(Ew[j][i]), lre;
                        if (sqe == 0)lre = 0;
                        else lre = LR / sqe;
                        weight[j][i] -= lre * dw;
                    }
                }
            }
        }
        //cout << "bug b3 " << endl;
        for (int jo = 0; jo < SZH; ++jo) {
            for (int io = 0; io < SZO; ++io) {

                for (int i = 0; i < SZH; ++i) {
                    double db = LRB * cost * weighto[jo][io] * dRelu(hidden[i]);
                    Eb[i] = beta * Eb[i] + (1 - beta) * db;
                    bias[i] -= Eb[i];
                }

            }
        }
        //cout << "bug b4 " << endl;


    }

    double get_result() {
        return hiddeno[0];
    }

    void dump() {
        cout << "weight" << endl;
        for (int i = 0; i < SZI; ++i) {
            cout << i << ": ";
            for (int j = 0; j < SZH; ++j) {
                cout << weight[i][j] << " ";
            }cout << endl;
        }

        cout << "biash" << endl;
        for (int i = 0; i < SZH; ++i) {
            cout << bias[i] << endl;
        }

        cout << "biaso" << endl << biaso[0] << endl;
    }

    void Train(vector<double> input, int target, int nb) {
        build(input);
        for (int i = 0; i < nb; ++i) {
            Compute();
            Backpropagation1(target);

        }

    }

    void TrainRMSprop(vector<double> input, int target, int nb) {
        build(input);
        for (int i = 0; i < nb; ++i) {
            Compute();
            BackpropagationRMSprop(target);

        }

    }


};


int main()
{
    std::cout << "Hello World!\n";
    vector<double> a = { 1, 1 };
   

    NN1 nn(3, 4, 1);
    /*nn.dump();
    nn.TrainRMSprop({1000, 0}, 1000, 100);
    nn.dump();
    nn.TrainRMSprop({ 10, 500 }, 510, 100);
    nn.dump();
    nn.TrainRMSprop({230, 500}, 730, 100);
    nn.dump();
    nn.TrainRMSprop({ 100, 100 }, 200, 100);
    nn.dump();
    nn.TrainRMSprop({ 235, 457 }, 692, 100);
    nn.dump();
    nn.TrainRMSprop({ 500, 200 }, 700, 100);
    nn.dump();*/
    
    /*nn.Train({5000, 0}, 100, 100);
    nn.Train({ 5000, 180 }, 100, 100);
    nn.Train({ 5000, -180 }, 100, 100);
    nn.Train({ 0, 180 }, 75, 100);
    nn.Train({ 0, -180 }, 75, 100);
    nn.Train({ 0, 0 }, 75, 100);
    nn.Train({ 2500, 20 }, 100, 100);
    nn.Train({ 2500, -20 }, 100, 100);
    nn.Train({ 1000, 40 }, 100, 100);
    nn.Train({ 1000, -40 }, 100, 100);
    nn.Train({ 666, 20 }, 100, 100);
    nn.Train({ 666, -20 }, 100, 100);
    nn.dump(); */

    /*nn.dump();
    nn.TrainRMSprop({ 1000, 180, 100 }, 70, 10);
    nn.TrainRMSprop({ 1000, 110, 100 }, 70, 10);
    nn.TrainRMSprop({ 1000, 60, 100 }, 50, 10);
    nn.TrainRMSprop({ 1000, 25, 100 }, 25, 10);
   // nn.TrainRMSprop({ 1000, 25, 50 }, 100, 10);
    nn.TrainRMSprop({ 1000, -180, 100 }, 70, 10);
    nn.TrainRMSprop({ 1000, -110, 100 }, 70, 10);
    nn.TrainRMSprop({ 1000, -60, 100 }, 50, 10);
    nn.TrainRMSprop({ 1000, -25, 100 }, 25, 10);
   // nn.TrainRMSprop({ 1000, -25, 50 }, 50, 10);
    nn.TrainRMSprop({5000, 180, 100}, 100, 10);
    nn.TrainRMSprop({ 5000, 110, 100 }, 100, 10);
    nn.TrainRMSprop({ 5000, 60, 100 }, 100, 10);
    nn.TrainRMSprop({ 5000, 25, 100 }, 100, 10);
    nn.TrainRMSprop({ 5000, -180, 100 }, 100, 10);
    nn.TrainRMSprop({ 5000, -110, 100 }, 100, 10);
    nn.TrainRMSprop({ 5000, -60, 100 }, 100, 10);
    nn.TrainRMSprop({ 5000, -25, 100 }, 100, 10);
    nn.dump()*/


    /*nn.dump();
    nn.TrainRMSprop({ 1000, 180, 25 }, 50, 10);
    nn.TrainRMSprop({ 1000, 180, -25 }, 50, 10);
    nn.TrainRMSprop({ 1000, -180, 25 }, 50, 10);
    nn.TrainRMSprop({ 1000, -180, -25 }, 50, 10);
    nn.TrainRMSprop({ 1000, 110, 25 }, 50, 10);
    nn.TrainRMSprop({ 1000, 110, -25 }, 50, 10);
    nn.TrainRMSprop({ 1000, -110, 25 }, 50, 10);
    nn.TrainRMSprop({ 1000, -110, -25 }, 50, 10);
    nn.TrainRMSprop({ 1000, 60, 25 }, 50, 10);
    nn.TrainRMSprop({ 1000, 60, -25 }, 50, 10);
    nn.TrainRMSprop({ 1000, -60, 25 }, 50, 10);
    nn.TrainRMSprop({ 1000, -60, -25 }, 50, 10);
    nn.TrainRMSprop({ 1000, 25, 25 }, 50, 10);
    nn.TrainRMSprop({ 1000, 25, -25 }, 50, 10);
    nn.TrainRMSprop({ 1000, -25, 25 }, 50, 10);
    nn.TrainRMSprop({ 1000, -25, -25 }, 50, 10);
    
    nn.TrainRMSprop({ 5000, 180, 25 }, 200, 10);
    nn.TrainRMSprop({ 5000, 180, -25 }, 200, 10);
    nn.TrainRMSprop({ 5000, -180, 25 }, 200, 10);
    nn.TrainRMSprop({ 5000, -180, -25 }, 200, 10);
    nn.TrainRMSprop({ 5000, 110, 25 }, 200, 10);
    nn.TrainRMSprop({ 5000, 110, -25 }, 200, 10);
    nn.TrainRMSprop({ 5000, -110, 25 }, 200, 10);
    nn.TrainRMSprop({ 5000, -110, -25 }, 200, 10);
    nn.TrainRMSprop({ 5000, 60, 25 }, 200, 10);
    nn.TrainRMSprop({ 5000, 60, -25 }, 200, 10);
    nn.TrainRMSprop({ 5000, -60, 25 }, 200, 10);
    nn.TrainRMSprop({ 5000, -60, -25 }, 200, 10);
    nn.TrainRMSprop({ 5000, 25, 25 }, 200, 10);
    nn.TrainRMSprop({ 5000, 25, -25 }, 200, 10);
    nn.TrainRMSprop({ 5000, -25, 25 }, 200, 10);
    nn.TrainRMSprop({ 5000, -25, -25 }, 200, 10);
    nn.dump();*/

    nn.dump();

    double d[] = { 1000, 5000 };
    double v[] = { 50, 120 };

    for (int s = 0; s < 2; ++s) {
        for (double i = -180; i <= 180; i += 20) {
            for (double j = -180; j <= 180; j += 20) {
                nn.TrainRMSprop({ d[s], i, j }, v[s], 10);
            }
        }
    }

    nn.dump();

    nn.build({ 2000, 120, 25 });//240
    nn.Compute();
    cout << "res 1 " << nn.get_result() << endl;

    nn.build({ 5000, 60, 60});//834
    nn.Compute();
    cout << "res 2 " << nn.get_result() << endl;
    
    nn.build({ 7563, 88, 46 });//303
    nn.Compute();
    cout << "res 3 " << nn.get_result() << endl;
}
