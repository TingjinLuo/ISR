function T = TransLabelR(gnd)

% TransLabelR function makes the integer-valued labels represented by binary
% coded labels, such as the k-th class is [0,..0,1(k),0...0]
% gnd: gnd(i) is a integer-valued labels,Num_S*1
% T: Binary-coded Labels,Num_S*C
% note: {0:unlabeled data}

% gnd = gnd -min(gnd)+1;% From 1 to C
Num_S = length(gnd);
C_Intv = unique(gnd);
Num_C = max(C_Intv);

T =zeros(Num_S,Num_C);

for i=1:Num_S
    T(i,gnd(i))=1;
end
