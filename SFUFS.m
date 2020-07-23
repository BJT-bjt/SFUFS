function [W, S, F, ob] = SFUFS(X, A0, lambda,  pdim, h)
%SFUFS 此处显示有关此函数的摘要
%   c: projection dimension
%   h: number of selected features

[dim,num] = size(X);
NIters =1;
maxIter = 500;
%initialize W
W = rand(dim, pdim);
W = orth(W);
F = rand(num, pdim);
distX = L2_distance_1(X,X);
[distX1, idx] = sort(distX,2);
issymmetric=1;

distX = L2_distance_1(X,X);
[distX1, idx] = sort(distX,2);

r = 0.1;
A0 = A0-diag(diag(A0));
A10 = (A0+A0')/2;
D10 = diag(sum(A10));
L0 = D10 - A10;
L = L0;
for it = 1:NIters
    
        %% compute S
    distf = L2_distance_1(F',F');
    S1 = zeros(num);
    for i=1:num
        idxa0 = 1:num;
        a0 = A0(i,:);
        ai = a0(idxa0);
        dfi = distf(i,idxa0);
%         ad = -(dfi)/(4*r);
        ad = ai-(r*dfi)/(4);
        S1(i,idxa0) = EProjSimplex_new(ad);
    end;
    S = S1;
    S = (S+S')/2;
    Ds = diag(sum(S));
    Ls = Ds - S;
    L = Ls;
    
%     F_old = F;
    %compute F
    I = eye(size(L));
    invL = inv((L + lambda*I));
    
%     [F, temp, ev]=eig1(L, pdim, 0);
    AA = X*(I-lambda*invL)*X';
    eva = norm(AA,2);
    [v,e] = eig(AA);
    eva2 = max(diag(e));
    eta = eva;
    etae(it) = eta;
    A = sparse(eta*eye(size(AA))-AA);
    %% compute W
    iter=1;
    for iter = 1:maxIter
        %update P
        pinvAW = sparse(pinv(W'*A*W));
        P = sparse(A*W*pinvAW*W'*A);
        diagP = diag(P);
        %update W
        [value,index] = sort(diagP,'descend');
        indexW = index(1:h);
        indexO = index(h+1:end);
        M = A*W;
        MP = M([indexW], :);
        OMP = orth(MP);
        W([indexW],:) = OMP;
        W([indexO],:) =0;
        obj(iter) = trace(W'*A*W);
        disp(['when iteration = ',num2str(iter), ', the objective function value=', num2str(obj(iter))])
    end
    F = lambda*invL*X'*W;
    obj2(it) = norm(S-A0,'fro')^2+r*trace(F'*L*F)+lambda*norm(X'*W-F,'fro')^2;
%     disp(['when iteration = ',num2str(it), ', eva=', num2str(eva)])
%     disp(['when iteration = ',num2str(it), ', obj2=', num2str(obj2(it))])
%     disp(['when iteration = ',num2str(it), ', eva2=', num2str(eva2)])
    
    
%     [~, temp, ev]=eig1(L, pdim, 0);
%     evs(:,it+1) = ev;

%     fn1 = sum(ev(1:pdim));
%     fn2 = sum(ev(1:pdim+1));
%     if fn1 > 0.00000000001
%         r = 2*r;
%     elseif fn2 < 0.00000000001
%         r = r/2;  %F = F_old;
%     else
%         break;
%     end;
%     ir(it) = r;
%     St{it} = S;

end
    ob.obj = obj;
    ob.obj2=obj2;
%     ob.eta = eva;
%     ob.ir = ir;
%     ob.St=St;


end

