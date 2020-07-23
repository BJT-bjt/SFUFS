function [W,objWAW] = L20_function1(A,m)
[d,~] = size(A);
W = orth(rand(d,m));
fea_num = m;
    for iter = 1:10
        %update P
        pinvAW = pinv(W'*A*W);
        P = A*W*pinvAW*W'*A;
        diagP = diag(P);
        %update W
        [value,index] = sort(diagP,'descend');
        indexW = index(1:fea_num);
        indexO = index(fea_num+1:end);
        M = A*W;
        MP = M([indexW], :);
        OMP = orth(MP);
        W([indexW],:) = OMP;
        W([indexO],:) =0;
        objWAW(iter) = trace(W'*A*W);
    end
    plot(objWAW);
end