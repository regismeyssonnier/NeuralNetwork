program Reinforcement_Learning;
{$H+}{$mode objfpc}  
uses sysutils, crt, math;

const
     dim : array[0..2] of Integer= (65, 18, 8);
     //dim : array[0..2] of Integer= (2, 10, 2);
     input: array[0..4- 1, 0..2 - 1] of Double = (
     (0, 0),
     (0, 1),
     (1, 0),
     (1, 1)
     );

  output: array[0..4- 1, 0..2 - 1] of Double = (
     (1, 0),
     (0, 1),
     (0, 1),
     (1, 0)
     );

  INITIAL_EPSILON = 0.9; // Probabilité initiale d'exploration
  MIN_EPSILON = 0.01;    // Probabilité minimale d'exploration
  EPSILON_DECAY = 0.995; // Facteur de diminution d'epsilon

  beta1 = 0.9;
  beta2 = 0.999;
  lambda = 0.1;


    {PDoubleArray = ^TDoubleArray;
    TDoubleArray = array[0..0] of Double;
    PDouble2DArray = ^TDouble2DArray;
    TDouble2DArray = array[0..0, 0..0] of Double;
    PDouble3DArray = ^TDouble3DArray;
    TDouble3DArray = array[0..0, 0..0, 0..0] of Double;}

//Allocate mem
{procedure AllocateArray1D(var arr: PDoubleArray; size: Integer);
begin
  GetMem(arr, size * SizeOf(Double));
end;

procedure AllocateArray2D(var arr: PDouble2DArray; rows, cols: Integer);
begin
  GetMem(arr, rows * cols * SizeOf(Double));
end;

procedure AllocateArray3D(var arr: PDouble3DArray; x, y, z: Integer);
begin
  GetMem(arr, x * y * z * SizeOf(Double));
end;}

// end Allocate mem

Type
    MadKnight = class
    public
    g : array[0..63] of double;
    np1, np2, np3, np : Integer;
    numa : array[0..3] of Integer;
    pmove: array[0..100] of Integer;
    pind_move : Integer;
    myp : Integer;


    constructor Create();
    function Init_Player(numplayer : Integer; var indices : array of Integer; sz : Integer):Integer;
    function GetBestMove(_g: array of double; numpl: Integer): Integer;
    function GetPossibleMove(numpl: Integer;var nb : Integer): Integer;
    function step(action: Integer; var done : Integer): Double;
    procedure Reset(var f: TextFile);
    end;

constructor MadKnight.Create();
var
   i : Integer;
begin
     for i := 0 to 63 do
     begin
          g[i] := 0;
     end;
     myp := 0;
     np := 0;
     np1 := 0;
     np2 := 0;
     np3 := 0;
end;

function MadKnight.Init_Player(numplayer : Integer; var indices : array of Integer; sz : Integer):Integer;
var
   ind, nind, i : Integer;

begin
     randomize;
     ind := Random( sz);
     g[indices[ind]] := numplayer;
     nind := indices[ind];
     for i:= ind to sz-1 do
     begin
          indices[i] := indices[i+1];
     end;

     Result := nind;

end;

procedure MadKnight.Reset(var f: TextFile);
var
   indices : array[0..63] of Integer;
   i, indi, indc : Integer;
begin
     //Writeln(f, 'before init');
      //flush(f);

     np := 1;

     for i := 0 to 63 do
     begin
          g[i] := 0;
          indices[i] := i;
     end;

     indc := 2;
     indi := 0;

     if myp = 0 then
     begin
         np1 := Init_Player(5, indices, 64);
         numa[indi] := 5;

     end
     else
     begin
         np1 := Init_Player(2, indices, 64);
         numa[indi] := 2;
         indc := indc + 1;
     end;

     indi := indi + 1;

     if myp = 1 then
     begin
         np2 := Init_Player(5, indices, 63);
         numa[indi] := 5;

     end
     else
     begin
         np2 := Init_Player(2, indices, 63);
         numa[indi] := 2;
         indc := indc + 1;
     end;

     indi := indi + 1;

     if myp = 2 then
     begin
         np3 := Init_Player(5, indices, 62);
         numa[indi] := 5;

     end
     else
     begin
         np3 := Init_Player(2, indices, 62);
         numa[indi] := 2;
         indc := indc + 1;
     end;
      //Writeln(f, 'after init');
      //flush(f);



end;

function MadKnight.GetBestMove(_g: array of double; numpl: Integer): Integer;
var
    x, y, i, xx, yy, res, count: Integer;
    dir: array[0..7, 0..1] of Integer;
begin
    x := numpl mod 8;
    y := numpl div 8;

    dir[0,0] := -2; dir[0,1] := 1;
    dir[1,0] := -2; dir[1,1] := -1;
    dir[2,0] := -1; dir[2,1] := -2;
    dir[3,0] := 1;  dir[3,1] := -2;
    dir[4,0] := 2;  dir[4,1] := -1;
    dir[5,0] := 2;  dir[5,1] := 1;
    dir[6,0] := 1;  dir[6,1] := 2;
    dir[7,0] := -1; dir[7,1] := 2;

    count := 0;
    for i := 0 to 7 do
    begin
        xx := x + dir[i,0];
        yy := y + dir[i,1];

        res := i;

        if (xx < 0) or (xx >= 8) or (yy < 0) or (yy >= 8) then
            res := -1
        else if _g[(yy * 8) + xx] <> 0 then
            res := -1;

        if res <> -1 then
            count := count + 1;
    end;

    GetBestMove := count;
end;

function MadKnight.GetPossibleMove(numpl: Integer;var nb : Integer): Integer;
var
    move: array[0..100] of Integer;
    ind_move : Integer;
    x, y, xx, yy, i, res, n: Integer;
    dir: array[0..7, 0..1] of Integer;
begin
    ind_move := 0;
    pind_move := 0;

    x := numpl mod 8;
    y := numpl div 8;

    dir[0,0] := -2; dir[0,1] := 1;
    dir[1,0] := -2; dir[1,1] := -1;
    dir[2,0] := -1; dir[2,1] := -2;
    dir[3,0] := 1;  dir[3,1] := -2;
    dir[4,0] := 2;  dir[4,1] := -1;
    dir[5,0] := 2;  dir[5,1] := 1;
    dir[6,0] := 1;  dir[6,1] := 2;
    dir[7,0] := -1; dir[7,1] := 2;

    for i := 0 to 7 do
    begin
        xx := x + dir[i,0];
        yy := y + dir[i,1];

        res := i;

        if (xx < 0) or (xx >= 8) or (yy < 0) or (yy >= 8) then
            res := -1
        else if g[(yy * 8) + xx] <> 0 then
            res := -1;

        if res <> -1 then
        begin
            move[ind_move] := i;
            ind_move := ind_move + 1;
            pmove[pind_move] := i;
            pind_move := pind_move + 1;
            nb := nb + 1;

        end;
    end;

    if ind_move = 0 then
        GetPossibleMove := -1
    else
    begin
        Randomize;
        n := move[Random(ind_move)];
        GetPossibleMove := ((y + dir[n,1]) * 8) + (x + dir[n,0]);
    end;
end;


function MadKnight.step(action: Integer; var done : Integer): Double;
var
    reward: Double;
    numpl, _numpl, x, y, ind_action, res, i, xx, yy, rest, indi, mv, mv2: Integer;
    max_score, max_score_zarma, opp_score: Double;
    dir: array[0..7, 0..1] of Integer;
    nb, mnb : Integer;
    move: array[0..100] of Integer;
    ind_move : Integer;

begin
    reward := 0;
    done := 0;

    dir[0,0] := -2; dir[0,1] := 1;
    dir[1,0] := -2; dir[1,1] := -1;
    dir[2,0] := -1; dir[2,1] := -2;
    dir[3,0] := 1;  dir[3,1] := -2;
    dir[4,0] := 2;  dir[4,1] := -1;
    dir[5,0] := 2;  dir[5,1] := 1;
    dir[6,0] := 1;  dir[6,1] := 2;
    dir[7,0] := -1; dir[7,1] := 2;

    numpl := 0;
    if np = 1 then
        numpl := np1
    else if np = 2 then
        numpl := np2
    else if np = 3 then
        numpl := np3;

    x := numpl mod 8;
    y := numpl div 8;

    ind_action := -100000;
    max_score := -1e38;
    max_score_zarma := -1e38;
    res := -1;
    

        xx := x + dir[action,0];
        yy := y + dir[action,1];

        res := 0;
        if (xx < 0) or (xx >= 8) or (yy < 0) or (yy >= 8) then
            res := -1
        else if g[(yy * 8) + xx] <> 0 then
            res := -1;


    if res <> -1 then
    begin
        g[numpl] := 1;
        numpl := ((y + dir[action,1]) * 8) + (x + dir[action,0]);
        opp_score := 0;
        nb := 0;
        GetPossibleMove(numpl, nb);

        for i := 0 to pind_move-1 do
        begin
             move[i] := pmove[i];
        end;
        ind_move := pind_move;



        
        if np = 1 then
        begin
            np1 := numpl;

            g[numpl] := numa[np-1];



        end
        else if np = 2 then
        begin
            np2 := numpl;

            g[numpl] := numa[np-1];

        end
        else if np = 3 then
        begin
            np3 := numpl;

            g[numpl] := numa[np-1];


        end;

        mnb := nb;
        if (myp = 0) and (np = 1) then
        begin
             nb := 0;
             GetPossibleMove(np2, nb);
             for i := 0 to pind_move-1 do
             begin
                  x := np2 mod 8;
                  y := np2 div 8;
                  mv := ((y + dir[pmove[i],1]) * 8) + (x + dir[pmove[i],0]);

                       if mv = numpl then
                       begin
                            mnb := mnb + 1;
                       end;

             end;

             GetPossibleMove(np3, nb);
             for i := 0 to pind_move-1 do
             begin
                  x := np3 mod 8;
                  y := np3 div 8;
                  mv := ((y + dir[pmove[i],1]) * 8) + (x + dir[pmove[i],0]);

                       if mv = numpl then
                       begin
                            mnb := mnb + 1;
                       end;

             end;



        end
        else if (myp = 1) and (np = 2) then
        begin
             nb := 0;
             GetPossibleMove(np1, nb);
             for i := 0 to pind_move-1 do
             begin
                  x := np1 mod 8;
                  y := np1 div 8;
                  mv := ((y + dir[pmove[i],1]) * 8) + (x + dir[pmove[i],0]);

                       if mv = numpl then
                       begin
                            mnb := mnb + 1;
                       end;

             end;

             GetPossibleMove(np3, nb);
             for i := 0 to pind_move-1 do
             begin
                  x := np3 mod 8;
                  y := np3 div 8;
                  mv := ((y + dir[pmove[i],1]) * 8) + (x + dir[pmove[i],0]);

                       if mv = numpl then
                       begin
                            mnb := mnb + 1;
                       end;

             end;

        end
        else if (myp = 2) and (np = 3) then
        begin
             nb := 0;
             GetPossibleMove(np1, nb);
             for i := 0 to pind_move-1 do
             begin
                  x := np1 mod 8;
                  y := np1 div 8;
                  mv := ((y + dir[pmove[i],1]) * 8) + (x + dir[pmove[i],0]);

                       if mv = numpl then
                       begin
                            mnb := mnb + 1;
                       end;

             end;

             GetPossibleMove(np2, nb);
             for i := 0 to pind_move-1 do
             begin
                  x := np2 mod 8;
                  y := np2 div 8;
                  mv := ((y + dir[pmove[i],1]) * 8) + (x + dir[pmove[i],0]);

                       if mv = numpl then
                       begin
                            mnb := mnb + 1;
                       end;

             end;
        end;

        x := numpl mod 8;
        y := numpl div 8;
        reward := float(mnb) + (abs(x - 3.5) + abs(y - 3.5));//(float(nb) - opp_score*0.5) / 8.0;
        //Write(reward);
        //reward := reward - (1.0 - (max_score / max_score_zarma)) * 0.1;
    end
    else
    begin
        if np = 1 then
        begin
             if myp = 0 then
             begin
                  reward := -1;
                  done := 1;
             end
             else
             begin
                  reward := 1;
                  done := 0;
             end;

             np1 := -1
        end
        else if np = 2 then
        begin
             if myp = 1 then
             begin
                  reward := -1;
                  done := 1;
             end
             else
             begin
                  reward := 1;
                  done := 0;
             end;

             np2 := -1
        end
        else if np = 3 then
        begin
             if myp = 2 then
             begin
                  reward := -1;
                  done := 1;
             end
             else
             begin
                  reward := 1;
                  done := 0;
             end;
             np3 := -1;
        end;


    end;

    if done = 0 then
    begin
        if (np = 1) and (myp = np-1) and (np2 = -1) and (np3 = -1) then
        begin
            reward := 1;
            done := 1;
        end;

        if (np = 2) and (myp = np-1) and (np1 = -1) and (np3 = -1) then
        begin
            reward := 1;
            done := 1;
        end;

        if (np = 3) and (myp = np-1) and (np1 = -1) and (np2 = -1) then
        begin
            reward := 1;
            done := 1;
        end;

        if (np1 = -1) and (np2 = -1) and (np3 = -1) then
        begin
            reward := -1;
            done := 1;
        end;
    end;

    np := np + 1;
    if np = 4 then
        np := 1;

    if (np1 = -1) and (np = 1) then
        np := 2;
    
    if (np2 = -1) and (np = 2) then
        np := 3;

    if (np3 = -1) and (np = 3) then
        np := 1;

    step := reward;
end;



Type
    NN = class
    public
    LR : double;
    epsilon,epsilon2,  epoch: Double;
    cost: array[0..512] of double;

    Mad : MadKnight;
  
    network_w, bnetwork_w, m_w, v_w: array[0..10, 0..512, 0..512]of double;
    network, network_b, bnetwork_b, m_b, v_b, results: array[0..10, 0..512]of double;
    etiquette: array[0..512] of double;
    dimnn : array[0..20] of Integer;
    maxdim :Integer;

    procedure ForwardNN();
    procedure BackwardNN();
    procedure BackwardNNA();
    procedure PredictNN();
    procedure SaveWeight();
    procedure LoadSaveWeight();
    procedure SaveBestNN(fn: string);
    procedure SaveBestNNN(fn: string);
    constructor Create(dim : array of Integer; sz : Integer; _lr : Double);
    procedure SetInput(inp : array of double);
    procedure SetEtiquette(et : array of double);
    function GetTanhD(v : double):double;
    function GetReluD(v : double):double;
    function Relu(v : double):double;
    function SelectAction(state: array of Double): Integer;
    function SelectActionMD(state: array of Double): Integer;
    procedure UpdateEpsilon();
    function ArgMax():Integer;
    procedure TrainXOR();
    procedure TrainMAD();
    procedure Free_mem();
    end;

function RandomNormal(mean, stddev: Double): Double;
begin
  Result := mean + stddev * Sqrt(-2 * Ln(Random)) * Cos(2 * Pi * Random);
end;

function UniformRandom(min, max: Double): Double;
begin
  Result := min + (max - min) * Random; // Générer un nombre uniformément distribué
end;


constructor NN.Create(dim : array of Integer; sz : Integer; _lr : Double);
var
   i, j, k, dmax, maxi, ni, no : Integer;
   sq : double;
   outputFile: TextFile;

begin
  dmax := -1;
  maxi := -2000000;
  for i := 0 to sz-1 do
  begin
       dimnn[i] := dim[i];
       WriteLN(inttostr(dimnn[i]));
       if dim[i] > maxi then
       begin
            maxi := dim[i];
       end;
  end;

  epsilon := INITIAL_EPSILON;
  epsilon2 := 0.00000001;
  LR := _lr;
  epoch := 0;

  WriteLN(inttostr(dimnn[0]));

  maxdim := sz;

  {AllocateArray1D(etiquette, dim[sz-1]);
  AllocateArray1D(cost, dim[sz-1]);

  AllocateArray2D(results, sz-1, maxi);
  AllocateArray2D(network, sz, maxi);
  AllocateArray2D(network_b, sz-1, maxi);

  // Allocation pour les tableaux 3D
  AllocateArray3D(network_w, sz-1, maxi, maxi);  }


  Randomize;

  AssignFile(outputFile, 'C:\Users\regis\Documents\Pascal\RL\network_weights_initial.txt'); // Nom du fichier
  Rewrite(outputFile); // Création ou écriture dans le fichier
  for i := 0 to maxdim-2 do
     begin
          for j := 0 to dimnn[i]-1 do
          begin
               for k:=0 to dimnn[i+1]-1 do
               begin
                 //if i = 0 then
                 //begin
                      ni := dimnn[i];
                 //end
                 //else
                 //begin
                 //     ni := dimnn[i-1];
                 //end;
                 no := dimnn[i+1];
                 m_w[i, j, k] := 0;
                 v_w[i, j, k] := 0;
                 network_w[i, j, k] := (Random - 0.5) * sqrt(2.0 / float(ni + no));
                 WriteLN(outputfile, i,' ', j, ' ', k, ' ' , network_w[i, j, k]);
                 flush(outputfile);
            end;
       end;
  end;
  CloseFile(outputFile);

  for i := 0 to sz - 2 do
  begin
       for j := 0 to maxi-1 do
       begin
            m_b[i, j] := 0;
            v_b[i, j] := 0;
            network_b[i, j] := UniformRandom(-0.5, 0.5);
            //WriteLn(network_b^[i, j]);
       end;

  end;

  for i := 0 to sz - 1 do
  begin
       for j := 0 to maxi-1 do
       begin
            network[i, j] := 0;
       end

  end;



end;

procedure NN.SetInput(inp : array of double);
var
   i : Integer;
begin
     //WriteLN(inttostr(dimnn[0]-1));
     for i := 0 to dimnn[0]-1 do
     begin
          network[0, i] := inp[i];
          //Writeln(i, ' ', network^[0, i]);

     end;

end;

procedure NN.SetEtiquette(et : array of double);
var
   i : Integer;
begin
     for i := 0 to dimnn[maxdim-1]-1 do
     begin
          etiquette[i] := et[i];

     end;


end;

procedure NN.Free_mem();
begin
     {FreeMem(network);
     FreeMem(network_b);
     FreeMem(network_w);
     FreeMem(results);
     FreeMem(cost);
     FreeMem(etiquette);    }
end;

function NN.GetTanhD(v : double):double;
begin
    Result := 1.0 - Tanh(v) * Tanh(v);
end;

function NN.GetReluD(v : double):double;
begin
     if v < 0.0 then
        Result := 0.0
     else
        Result := 1.0;
end;

function NN.Relu(v : double):double;
begin
  if v < 0.0 then
    Result := 0.0
  else
    Result := v;
end;

function NN.ArgMax():Integer;
var
   i, ind : Integer;
   maxi : double;
begin
     ind := -1;
     maxi := -2000000;
     for i := 0 to dimnn[maxdim-1]-1 do
     begin
          if network[maxdim-1, i] > maxi then
          begin
               maxi := network[maxdim-1, i];
               ind := i;
          end;
     end;

     Result := ind;
end;

function NN.SelectActionMD(state: array of Double): Integer;
var
    rand_val: Double;
    best_action, nb: Integer;
begin
    // Génère une valeur aléatoire entre 0 et 1
    {rand_val := Random;
    nb := 0;
    mad.GetPossibleMove(mad.np, nb);
    if (rand_val < epsilon) and (nb > 0) then
    begin
        // Choisir une action aléatoire (exploration)

        SelectActionMD := mad.pmove[Random(mad.pind_move)];  // Remplacez '2' par le nombre d'actions possibles (ex: 2 pour XOR)
    end
    else
    begin  }
        // Exploiter le réseau de neurones pour choisir la meilleure action
        SetInput(state);
        PredictNN();
        best_action := ArgMax(); // Implémentez ArgMax pour choisir l'action ayant le meilleur résultat du réseau
        SelectActionMD := best_action;
    //end;
end;

function NN.SelectAction(state: array of Double): Integer;
var
    rand_val: Double;
    best_action: Integer;
begin
    // Génère une valeur aléatoire entre 0 et 1
    {rand_val := Random;

    if rand_val < epsilon then
    begin
        // Choisir une action aléatoire (exploration)
        SelectAction := Random(2);  // Remplacez '2' par le nombre d'actions possibles (ex: 2 pour XOR)
    end
    else
    begin}
        // Exploiter le réseau de neurones pour choisir la meilleure action
        SetInput(state);
        PredictNN();
        best_action := ArgMax(); // Implémentez ArgMax pour choisir l'action ayant le meilleur résultat du réseau
        SelectAction := best_action;
    //end;
end;

procedure NN.UpdateEpsilon();
begin
    // Réduire epsilon pour favoriser l'exploitation au fur et à mesure que le modèle apprend
    if epsilon > MIN_EPSILON then
    begin
        epsilon := epsilon * EPSILON_DECAY;
    end;
end;

procedure NN.PredictNN();
var
   i, j, k, ind : Integer;
   h : double;
begin
     for i := 0 to maxdim-2 do
     begin
          for j := 0 to dimnn[i+1]-1 do
          begin
               h := 0.0;
               for k := 0 to dimnn[i]-1 do
               begin
                    h := h + network[i, k] * network_w[i, k, j];
               end;
               h := h + network_b[i, j];
               //WriteLn('avant Activation: network[', i+1, ',', j, '] = ', network^[i+1, j]);
               network[i+1, j] := TanH(h);
                //WriteLn('after Activation: network[', i+1, ',', j, '] = ', network^[i+1, j]);

          end;

     end;

     ind := maxdim-2;
     for j:=0 to dimnn[ind+1]-1 do
     begin
         h := 0.0;
         for k := 0 to dimnn[ind]-1 do
         begin
              h := h + network[ind, k] * network_w[ind, k, j];
         end;

         h:= h + network_b[ind, j];
         network[ind+1, j] := h;

     end;

end;


procedure NN.ForwardNN();
var
   i, j, k, ind : Integer;
   h : double;
begin
     for i := 0 to maxdim-2 do
     begin
          for j := 0 to dimnn[i+1]-1 do
          begin
               h := 0.0;
               for k := 0 to dimnn[i]-1 do
               begin
                    h := h + network[i, k] * network_w[i, k, j];
               end;
               h := h + network_b[i, j];
               //WriteLn('avant Activation: network[', i+1, ',', j, '] = ', network^[i+1, j]);
               network[i+1, j] := TanH(h);
                //WriteLn('after Activation: network[', i+1, ',', j, '] = ', network^[i+1, j]);

          end;

     end;

     ind := maxdim-2;
     for j:=0 to dimnn[ind+1]-1 do
     begin
         h := 0.0;
         for k := 0 to dimnn[ind]-1 do
         begin
              h := h + network[ind, k] * network_w[ind, k, j];
         end;

         h:= h + network_b[ind, j];
         network[ind+1, j] := h;

     end;

     for i := 0 to dimnn[maxdim-1]-1 do
     begin
          cost[i] := cost[i] + 2.0 * (network[maxdim-1, i] - etiquette[i]) / float(dimnn[maxdim-1]);
     end;


end;

procedure NN.BackwardNN();
var
   i, j, k : Integer;
   ind_result : Integer;
   h : double;
begin
     for i := 0 to maxdim - 1 do
     begin
          for j := 0 to dimnn[i+1]-1 do
          begin
               results[i, j] := 0;
          end;
     end;

     for i := 0 to dimnn[maxdim-1]-1 do
     begin
          results[maxdim-2, i] := cost[i];
          //WriteLn('result=' + floattostr(results[maxdim-1, i]));
          network_b[maxdim-2, i] := network_b[maxdim-2, i] - LR * results[maxdim-2, i];
     end;


     for i := maxdim-3 downto 0 do
     begin
          //WriteLn('result=' + inttostr(i) );
          for j := 0 to dimnn[i+1]-1 do
          begin
               for k := 0 to dimnn[i+2]-1 do
               begin
                    results[i, j] := results[i, j] + network_w[i+1,j, k] * results[i+1, k] * GetTanhD(network[i+1, j]);
                    //WriteLn('result=' + floattostr(results[i, j]));
               end;

               network_b[i, j] := network_b[i, j] - LR * results[i, j];
          end;




     end;

     for i := maxdim-2 downto 0 do
     begin
          h := 0;
          for j := 0 to dimnn[i]-1 do
          begin
               for k:=0 to dimnn[i+1]-1 do
               begin
                    //WriteLN('Before Update: network_w[0, 0, 0] = ', network_w[i, j, k]);
                    network_w[i, j, k] := network_w[i, j, k] - LR * network[i, j] * results[i, k];
                    //WriteLN('After Update: network_w[0, 0, 0] = ', network_w[i, j, k]);

               end;
          end;

     end;

     for i := 0 to dimnn[maxdim-1]-1 do
     begin
          cost[i] := 0.0;
     end;

end;


procedure NN.BackwardNNA();
var
   i, j, k : Integer;
   ind_result : Integer;
   h, gradient_w, m_hat_w, v_hat_w, gradient_b, m_hat_b, v_hat_b : double;
begin
     for i := 0 to maxdim - 2 do
     begin
          for j := 0 to dimnn[i+1]-1 do
          begin
               results[i, j] := 0;
          end;
     end;

     for i := 0 to dimnn[maxdim-1]-1 do
     begin
          results[maxdim-2, i] := cost[i];
          //WriteLn('result=' + floattostr(results[maxdim-1, i]));
          network_b[maxdim-2, i] := network_b[maxdim-2, i] - LR * results[maxdim-2, i];
     end;

     for i := maxdim-3 downto 0 do
     begin
          //WriteLn('result=' + inttostr(i) );
          for j := 0 to dimnn[i+1]-1 do
          begin
               for k := 0 to dimnn[i+2]-1 do
               begin
                    results[i, j] := results[i, j] + network_w[i+1,j, k] * results[i+1, k] * GetTanhD(network[i+1, j]);
                    //WriteLn('result=' + floattostr(results[i, j]));
               end;

               network_b[i, j] := network_b[i, j] - LR * results[i, j];
          end;




     end;

     for i := maxdim-2 downto 0 do
     begin
          for j := 0 to dimnn[i]-1 do
          begin
               for k := 0 to dimnn[i+1]-1 do
               begin
                    //Writeln(i, ' ',j, ' ',k);
                    gradient_w := lambda * (network[i,j] * results[i,k]);

                    m_w[i,j,k] := beta1 * m_w[i,j,k] + (1.0 - beta1) * gradient_w;
                    v_w[i,j,k] := beta2 * v_w[i,j,k] + (1.0 - beta2) * (gradient_w*gradient_w);

                    m_hat_w := m_w[i,j,k] / (1.0 - Power(beta1, epoch + 1.0));
                    v_hat_w := v_w[i,j,k] / (1.0 - Power(beta2, epoch + 1.0));

                    network_w[i,j,k] := network_w[i,j,k] - LR * m_hat_w / (sqrt(v_hat_w) + epsilon2);


               end;
          end;
     end;


     for i := maxdim-2 downto 0 do
     begin
          for j := 0 to dimnn[i+1]-1 do
          begin

                gradient_b := results[i,j];

                m_b[i,j] := beta1 * m_b[i,j] + (1 - beta1) * gradient_b;
                v_b[i,j] := beta2 * v_b[i,j] + (1 - beta2) * (gradient_b*gradient_b);

                m_hat_b := m_b[i,j] / (1.0 - Power(beta1, epoch + 1.0));
                v_hat_b := v_b[i,j] / (1.0 - Power(beta2, epoch + 1.0));

                network_b[i,j] := network_b[i,j] - LR * m_hat_b / (sqrt(v_hat_b) + epsilon2);

          end;

     end;



     for i := 0 to dimnn[maxdim-1]-1 do
     begin
          cost[i] := 0.0;
     end;



end;

procedure NN.SaveWeight();
var
   i, j, k: Integer;
begin
     for i := 0 to maxdim-2 do
     begin
          for j := 0 to dimnn[i]-1 do
          begin

               for k := 0 to dimnn[i+1]-1 do
               begin
                    bnetwork_w[i][j][k] := network_w[i][j][k];
               end;
          end;
     end;

     for j := 0 to maxdim-2 do
     begin
          for k := 0 to dim[j+1]-1 do
          begin
               bnetwork_b[j][k] := network_b[j][k];
          end;
     end;
end;

procedure NN.LoadSaveWeight();
var
   i, j, k: Integer;
begin
     for i := 0 to maxdim-2 do
     begin
          for j := 0 to dimnn[i]-1 do
          begin

               for k := 0 to dimnn[i+1]-1 do
               begin
                    network_w[i][j][k] := bnetwork_w[i][j][k];
               end;
          end;
     end;

     for j := 0 to maxdim-2 do
     begin
          for k := 0 to dim[j+1]-1 do
          begin
               network_b[j][k] := bnetwork_b[j][k];
          end;
     end;
end;

procedure NN.SaveBestNN(fn: string);
var
    f: TextFile;
    i, j, k: Integer;

begin
    AssignFile(f, fn);
    Rewrite(f);
    DecimalSeparator := '.';

    Write(f, '{');
    for i := 0 to maxdim-2 do
    begin
        Write(f, '/* weight layer ', i, '*/', #13#10);
        Write(f, '{');
        for j := 0 to dimnn[i]-1 do
        begin
            Write(f, '{');
            for k := 0 to dimnn[i+1]-1 do
            begin
                Write(f, FloatToStrf(bnetwork_w[i][j][k], ffFixed, 7, 6));
                if k < dimnn[i+1]-1 then
                    Write(f, ',');
            end;
            Write(f, '}');
            if j < dimnn[i]-1 then
                Write(f, ',');
        end;
        Write(f, '}');
        if i < maxdim-2 then
            Write(f, ',' + #13#10);
    end;
    Write(f, '},' + #13#10 + '{');

    for j := 0 to maxdim-2 do
    begin
        Write(f, '/*layer bias : */', #13#10);
        Write(f, '{');
        for k := 0 to dim[j+1]-1 do
        begin
            Write(f, FloatToStrf(bnetwork_b[j][k], ffFixed, 7, 6));
            if k < dim[j+1]-1 then
                Write(f, ',');
        end;
        Write(f, '}');
        if j < maxdim-2 then
            Write(f, ',' + #13#10);
    end;
    Write(f, '}' + #13#10);

    CloseFile(f);
end;

procedure NN.SaveBestNNN(fn: string);
var
    f: TextFile;
    i, j, k: Integer;

begin
    AssignFile(f, fn);
    Rewrite(f);
    DecimalSeparator := '.';

    Write(f, '{');
    for i := 0 to maxdim-2 do
    begin
        Write(f, '/* weight layer ', i, '*/', #13#10);
        Write(f, '{');
        for j := 0 to dimnn[i]-1 do
        begin
            Write(f, '{');
            for k := 0 to dimnn[i+1]-1 do
            begin
                Write(f, FloatToStrf(network_w[i][j][k], ffFixed, 7, 6));
                if k < dimnn[i+1]-1 then
                    Write(f, ',');
            end;
            Write(f, '}');
            if j < dimnn[i]-1 then
                Write(f, ',');
        end;
        Write(f, '}');
        if i < maxdim-2 then
            Write(f, ',' + #13#10);
    end;
    Write(f, '},' + #13#10 + '{');

    for j := 0 to maxdim-2 do
    begin
        Write(f, '/*layer bias : */', #13#10);
        Write(f, '{');
        for k := 0 to dim[j+1]-1 do
        begin
            Write(f, FloatToStrf(network_b[j][k], ffFixed, 7, 6));
            if k < dim[j+1]-1 then
                Write(f, ',');
        end;
        Write(f, '}');
        if j < maxdim-2 then
            Write(f, ',' + #13#10);
    end;
    Write(f, '}' + #13#10);

    CloseFile(f);
end;


procedure NN.TrainXOR();
var
   state: array[0..2-1] of Double;
   action, correct_action, i, j,l, fr,  episode, total_rewardm: Integer;
   total_reward, reward, maxr: Double;
begin
   epsilon := INITIAL_EPSILON; // Initialiser epsilon pour la stratégie e-greedy
   maxr := -2000000000;
   total_rewardm := 0;
   for episode := 0 to 100000 do // Nombre d'épisodes d'entraînement
   begin

      total_reward := 0;
      for i := 0 to 4 - 1 do // Boucle sur les exemples XOR
      begin
         // Charger l'input XOR
         state[0] := input[i, 0];
         state[1] := input[i, 1];
         
         // Sélection de l'action (0 ou 1) selon la stratégie e-greedy
         action := SelectAction(state);

         // Obtenir l'action correcte attendue
         if output[i, action] = 1 then
            reward := 1 // Récompense maximale si l'action est correcte
         else
            reward := -1; // Pénalité si l'action est incorrecte

         total_reward := total_reward + reward;
         total_rewardm := total_rewardm + trunc(reward);

         // Mise à jour des coûts pour le backward
         for j := 0 to dimnn[maxdim-1]-1 do
         begin
            if j = action then
            begin
                 fr := ArgMax();
                 cost[j] :=  network[maxdim-1, j] + (-reward + 0.95 *  network[maxdim-1, fr]-network[maxdim-1, j])   ;
            end
            else
            begin
                 cost[j] :=  0.0;
            end;

         end;

         if float(total_rewardm) > maxr then
         begin
           //WriteLn( 'rew ', total_reward);
           maxr := float(total_rewardm);
           //SaveWeight();
         end;

         // Rétropropagation de l'erreur via BackwardNN
         BackwardNNA();
      end;




      // Mise à jour du taux d'exploration
      UpdateEpsilon();

      if ((episode mod 100) = 0) then
         // Affichage de la récompense totale pour cet épisode
         WriteLn('Episode ', episode, ' total=', total_rewardm, ' Max Reward: ', FloatToStr(maxr));
   end;
   WriteLn('Episode ', episode, ' Total Reward: ', FloatToStr(maxr));

   //LoadSaveWeight();
   for j := 0 to 3 do
     begin
          SetInput(input[j]);
          PredictNN();
          for l := 0 to dimnn[0]-1 do
          begin
               WriteLn('input' + inttostr(l) + '=' + floattostr(input[j, l]));
          end;
          //WriteLn('output=' + floattostr(nnz.network[nnz.maxdim-1, 0]));
          for l := 0 to dimnn[maxdim-1]-1 do
          begin
               WriteLn('output' + inttostr(l) + '=' + floattostr(network[maxdim-1, l]));

          end;

          Writeln('Response=' + inttostr(ArgMax()));
     end;


end;

procedure NN.TrainMad();
var
   state: array[0..64] of Double;
   action, correct_action, i, j,l, fr,  episode, ind_state, done, x: Integer;
   total_reward, reward, maxr, meanr, dv: Double;
   f: TextFile;
   is_our_turn : boolean;


begin
   epsilon := INITIAL_EPSILON; // Initialiser epsilon pour la stratégie e-greedy
   maxr := -2000000000;
   meanr := 0;
   dv := 1;
   mad := MadKnight.Create();
   SaveWeight();

   AssignFile(f, 'C:\Users\regis\Documents\Pascal\RL\output.txt');
    Rewrite(f);


   for episode := 0 to 100000 do // Nombre d'épisodes d'entraînement
   begin
      // WriteLn(f, 'episode=' + inttostr(episode));
       //flush(f);

      for x:= 0 to 2 do
      begin
        total_reward := 0;
      //WriteLn(f, 'myp=' + inttostr(mad.myp));
      // flush(f);
      //start begin
      mad.Reset(f);
      mad.myp := x;

      done := 0;
      while done = 0 do // Boucle sur les exemples XOR
      begin

          if ((mad.myp+1) = mad.np) then
          begin
               is_our_turn := true;
          end
          else
          begin
               is_our_turn := false;

          end;

         ind_state := 0;
         if (mad.np = 1) and (mad.np1 <> -1) then
         begin
              state[ind_state] := mad.np1 / 63.0;
              ind_state := ind_state + 1;

         end
         else if (mad.np = 2) and (mad.np2 <> -1) then
         begin
              state[ind_state] := mad.np2 / 63.0;
              ind_state := ind_state + 1;
         end
         else if (mad.np = 3) and (mad.np3 <> -1) then
         begin
              state[ind_state] := mad.np3 / 63.0;
              ind_state := ind_state + 1;

         end
         else
         begin

              state[ind_state] := 0;
              ind_state := ind_state + 1;
         end;

         for i := 0 to 63 do
         begin
              state[ind_state] := mad.g[i] / 5.0;
              ind_state := ind_state + 1;
         end;
         
         // Sélection de l'action (0 ou 1) selon la stratégie e-greedy
         action := SelectActionMd(state);

        // WriteLn(f, 'action=' + inttostr(action));
      // flush(f);

         //WriteLn('action=' + inttostr(action));

         reward := mad.step(action, done);

         //WriteLn(f, 'done=' + inttostr(done));
       //flush(f);
         //WriteLn(reward);
         if is_our_turn then
         begin
              total_reward := total_reward + reward;


         // Mise à jour des coûts pour le backward
         for j := 0 to dimnn[maxdim-1]-1 do
         begin
            if j = action then
            begin
                 fr := ArgMax();
                 cost[j] :=  network[maxdim-1, j] + (-reward + 0.95 *  network[maxdim-1, fr]-network[maxdim-1, j])   ;
            end
            else
            begin
                 cost[j] :=  0.0;
            end;

         end;

         // Rétropropagation de l'erreur via BackwardNN

              BackwardNNA();
         end; //end is_our_turn

         //WriteLn(f, 'backward=' + inttostr(done));
       //flush(f);
      end;  // end done
        //WriteLn(f, 'end episode=' + inttostr(episode));



      meanr := (meanr + total_reward);

      if total_reward > maxr then
      begin
           maxr := total_reward;
           SaveWeight();
      end;

      //end
      end;

      if ((episode mod 1000) = 0) then
         WriteLn('Episode ', episode, ' Total Reward: ', FloatToStr(total_reward), '/', FloatToStr(maxr), ' mean=', FloatToStr(meanr/ float(dv)), ' total_rew :', meanr);

         dv :=dv + 1;
      // Mise à jour du taux d'exploration
      UpdateEpsilon();

      //if ((episode mod 100) = 0) then
         // Affichage de la récompense totale pour cet épisode
         //WriteLn('Episode ', episode, ' Total Reward: ', FloatToStr(total_reward));
   end;
   WriteLn('Episode ', episode, ' Max Reward: ', FloatToStr(maxr));
   CloseFile(f);


end;

Type
    RL = class
    public

    constructor Create;
    end;


constructor RL.Create;
begin

end;

procedure Training();
var
   nnz : NN;
   i, j, k, l : Integer;
   error : double;
   outputFile: TextFile;
begin
     nnz := nn.Create(dim, 3, 0.001);

     //nnz.SetInput(input[1]);
     //nnz.network[0, 1] := 3.0;
     //nnz.SetEtiquette(output[1]);
     //nnz.ForwardNN();
     {for k := 0 to nnz.maxdim - 1 do
     begin
          for l := 0 to nnz.dimnn[k] - 1 do
          begin

               WriteLn(k, ' ', l, ' : ', floattostr(nnz.network[k,l]));

          end;
     end;}


     for i := 0 to 100000 do
     begin
          for j := 0 to 3 do
          begin
               nnz.SetInput(input[j]);
               nnz.SetEtiquette(output[j]);
               nnz.ForwardNN();
               nnz.BackwardNN();

          end;

          if ((i mod 100) = 0) then
          begin
               error := 0;
               for j := 0 to 3 do
               begin
                    nnz.SetInput(input[j]);
                    nnz.ForwardNN();


                    //WriteLn('output=' + floattostr(nnz.network[nnz.maxdim-1, 0]));
                    for l := 0 to nnz.dimnn[nnz.maxdim-1]-1 do
                    begin
                         //WriteLn('output=' + inttostr(l) + ' ' + floattostr(nnz.network[nnz.maxdim-1, l]));
                         error := error + (nnz.network[nnz.maxdim-1, l] - output[j, l]) * (nnz.network[nnz.maxdim-1, l] - output[j, l]);
                         //WriteLN('Error= ' + floattostr(error));
                    end;

                    error := error / 4.0;

               end;
               for j := 0 to nnz.dimnn[nnz.maxdim-1]-1 do
               begin
                    nnz.cost[j] := 0.0;
               end;

               WriteLN('Error= ' + floattostr(error));



          end;

     end;

     for j := 0 to 3 do
     begin
          nnz.SetInput(input[j]);
          nnz.ForwardNN();
          for l := 0 to nnz.dimnn[0]-1 do
          begin
               WriteLn('input' + inttostr(l) + '=' + floattostr(input[j, l]));
          end;
          //WriteLn('output=' + floattostr(nnz.network[nnz.maxdim-1, 0]));
          for l := 0 to nnz.dimnn[nnz.maxdim-1]-1 do
          begin
               WriteLn('output' + inttostr(l) + '=' + floattostr(nnz.network[nnz.maxdim-1, l]));

          end;
     end;

      AssignFile(outputFile, 'C:\Users\regis\Documents\Pascal\RL\network_weights.txt'); // Nom du fichier
  Rewrite(outputFile); // Création ou écriture dans le fichier

  // Boucle pour parcourir le réseau de neurones
  for i := 0 to 1 do
  begin
    for j := 0 to 9 do
    begin
      for k := 0 to 1 do
      begin
        // Écrire dans le fichier
        WriteLn(outputFile, i, ' ', j, ' ', k, ' ', nnz.network_w[i, j, k]);
      end;
    end;
  end;

  // Fermer le fichier
  CloseFile(outputFile);





     nnz.Free_mem();

end;



var
   nnz : NN;
   i : Integer;

begin
     Randomize;
     //Training();

      //nnz := nn.Create(dim, 3, 0.01);
      //nnz.TrainXor();

      nnz := nn.Create(dim, 3, 0.001);
      nnz.TrainMad();
      nnz.SaveBestNN('C:\Users\regis\Documents\Pascal\RL\best_w.txt');
      nnz.SaveBestNNN('C:\Users\regis\Documents\Pascal\RL\best_wn.txt');
                        
     readkey;

end.
