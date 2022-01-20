function [Uleft, Vright, Yset] = integrateODE(params)

% This code implements the 2S-POD-DEIM algorithm to integrate a
% semi-implicit ODE, where the reduced model is integrated
% by a matrix-oriented first-order exponential Euler scheme. All parameters
% are defined in the driver.

% Related Article:

% G. Kirsten and Valeria Simoncini.
% A matrix-oriented POD-DEIM algorithm applied to nonlinear differential matrix equations.
% pp. 1-25, June 2020. Revised May 2021. arXiv: 2006.13289   [math.NA].

%We provide this code without any guarantee that the code is bulletproof as input
%parameters are not checked for correctness.
%Finally, if the user encounters any problems with the code, the author can be
%contacted via e-mail.

%% Outputs %%

% Uleft -- Left basis vectors 
% Vright -- Right basis vectors 
% Yset -- Tensor of low dimensional solutions so that U(t_i) \approx Uleft*Yset(:,:,i)*Vright'.

%% %%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%
nsmax = params.nmax;
kaps = params.kaps;
tau = params.tau;
tf = params.tf;
nn = params.nn + 1;
ode = params.ode;

Utilde = []; Vtilde = []; ss = [];
FUtilde = []; FVtilde = []; Fss = [];

A = ode.A; X0 = ode.X0;
F = ode.F; G = ode.G;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


nsnaps = nsmax/4;
h = tf/nsnaps;
kappa= kaps;


tauh = tau/sqrt(nsmax); % This is the tolerance used for the basis truncation
error2tau = tau;

timer = 0;
i = 1;

countsnaps = 0;
Fcountsnaps = 0;
totalnodes = 0;
Ftotalnodes = 0;


%%% Preliminaries %%%

[V1,D1] = eig(full(A)); %eigenvalue decomposition of symmetric A
dd1 = diag(D1);
V1i = inv(V1);

[V2,D2] = eig(full(A'));
dd2 = diag(D2);
V2i = inv(V2);

fullint = tic; % Initiate timers
svdtime = 0;

L = eye(nn);
L1 = eye(nn);

%% Start the procedure

for phase =1:3 % The phases for the refinement procedure
    
    solution=1; errr = 1;     included = 1;
    nonlinear = 1; Ferrr=1;     Fincluded = 1;
    
    
    countsnaps=0;
    
    if phase==1
        
        nsnaps = nsmax/4; h = tf/nsnaps;
        kappa= kaps;
        timer = 0;
        phasetimer =1;
        X(:,:,1) = X0;
        FF(:,:,1) = F(timer,X(:,:,1));
        
        % We also calculate X(h) since we are using a second order SI Euler
        
        for ii = 1:nn
            for jj = 1:nn
                L1(ii,jj) = 1/((1-h*dd1(ii)) - (h*dd2(jj)));
            end
        end
        
        rhs = X(:,:,1) + h*FF(:,:,1) + h*G;
        
        
        
        Xnew1 = V1*( L1.*(V1i*(rhs)*V2) )*V2i;
        
        
        X(:,:,2) = Xnew1;
        FF(:,:,2) = F(timer+h,X(:,:,2));
        
        % Form the matrix L used for the quick solving of th LYAP eq.
        
        for ii = 1:nn
            for jj = 1:nn
                L(ii,jj) = 1/((3-2*h*dd1(ii)) - (2*h*dd2(jj)));
            end
        end
        
    elseif phase==2
        
        kappa= kaps;
        timer = h/2;
        phasetimer = 1;
        
        % Here we need to determine X(h/2) from X(0) as initial condition;
        
        rhs = X0 + timer*F(X0,0) + timer*G;
        X(:,:,1) = V1*( L1.*(V1i*(rhs)*V2) )*V2i;
        FF(:,:,1) = F(timer,X(:,:,1));
        
        % We also calculate X(h/2 + h) since we are using a second order SI Euler
        
        rhs = X(:,:,1) + h*FF(:,:,1) + h*G;
        X(:,:,2) = V1*( L1.*(V1i*(rhs)*V2) )*V2i;
        FF(:,:,2) = F(timer+h,X(:,:,2));
        
        % The stepsize remains the same as is phase 1, so L is the same
        
    elseif phase==3
        
        nsnaps = nsmax/2; h = tf/nsnaps;
        timer = h/2;
        phasetimer = 1;
        kappa= kaps;
        
        % Here we need to determine X(h/4) from X(0) as initial condition;
        
        for ii = 1:nn
            for jj = 1:nn
                L1(ii,jj) = 1/((1-h*dd1(ii)) - (h*dd2(jj)));
            end
        end
        rhs = X0 + timer*F(X0,0) + timer*G;
        X(:,:,1) = V1*( L1.*(V1i*(rhs)*V2) )*V2i;
        FF(:,:,1) = F(timer,X(:,:,1));
        
        % We also calculate X(h/4 + h) since we are using a second order SI Euler
        
        rhs = X(:,:,1) + h*FF(:,:,1) + h*G;
        X(:,:,2) = V1*( L1.*(V1i*(rhs)*V2) )*V2i;
        FF(:,:,2) = F(timer+h,X(:,:,2));
        
        % Form the matrix L used for the quick solving of th LYAP eq.
        clear L1
        
        for ii = 1:nn
            for jj = 1:nn
                L(ii,jj) = 1/((3-2*h*dd1(ii)) - (2*h*dd2(jj)));
            end
        end
        
    end
    
    fprintf('Starting the phase %d with  %d snapshots....\n\n\n\n',phase,nsnaps)
    fprintf('Snapshot no. Snapshot sol. included?  Nonlinear snap included?\n')
    
    while timer < tf
        
        if phasetimer < 3 %% This includes the two starting matrices into the space
            
            Xnew = X(:,:,phasetimer);
            Fnew = FF(:,:,phasetimer);
            
        else
            
            if phasetimer == 3 %Clear the tensor
                Xold = X(:,:,1); Fold = FF(:,:,1);
                clear X; clear FF;
            end
            
            
            cc = L.*( V1i*(4*Xnew - Xold + 2*h*(2*Fnew - Fold + G))*V2 ) ;
            
            Xold = Xnew;
            Xnew = (V1*cc)*V2i;
            
            Fold = Fnew;
            Fnew = F(timer,Xnew);
            
            
        end
        
        %% Test whether the snapshot needs to be included
        if timer > 0
            
            included = 0;Fincluded = 0;
            
            %%% SNAPSHOT SOLUTION CHECK %%%
            Xapprox = Uleft*(Uleft'*Xnew*Vright)*Vright';
            errr = norm(Xnew - Xapprox,'fro')/norm(Xnew,'fro');
            err(i)=errr; solution = 0;
            if errr > tau, solution = 1; included = 1; countsnaps = countsnaps+1; end
            
            %%% NONLINEAR SNAPSHOT CHECK %%%
            Fapprox = Dleft*(Dleft'*Fnew*Dright)*Dright';
            Ferrr = norm(Fnew - Fapprox,'fro')/norm(Fnew,'fro');
            Ferr(i)=Ferrr; nonlinear = 0;
            if Ferrr > tau, nonlinear = 1; Fincluded = 1; Fcountsnaps = Fcountsnaps+1; end
            
        end
        
        
        %% Preprocess the snapshots to update the bases when selected
        
        if solution || nonlinear % If either of the snapshots are not yet well approximated in the current space we add them.
            
            svdtimer = tic;
            
            if solution %%% PREPROCESS SNAPSHOT SOLUTION %%%%
                
                [utilde,stilde,vtilde]=svds(Xnew,kappa,'largest','Tolerance',1e-4);
                
                stildep=diag(stilde)';
                ntilde=sum(stildep/stildep(1)>tauh);
                stildep=stildep(1:ntilde);
            end
            
            if nonlinear %%% PREPROCESS NONLINEAR SNAPSHOT %%%%
                
                [Futilde,Fstilde,Fvtilde]=svds(Fnew,kappa,'largest','Tolerance',1e-4);
                
                Fstildep=diag(Fstilde)';
                Fntilde=sum(Fstildep/Fstildep(1)>tauh);
                Fstildep=Fstildep(1:Fntilde);
            end
            
            
            %% Update the bases if the snapshot is selected
            
            if solution
                %%% SNAPSHOT SOLUTION %%%%
                
                Utilde = [Utilde, utilde(:,1:ntilde)];
                Vtilde = [Vtilde, vtilde(:,1:ntilde)];
                
                ss = [ss, stildep];
                
                if length(ss) > kappa %% When the collected singular triplets become more than kmax, truncate and keep the top kmax
                    [sdecrease,id] = sort(ss,'descend');
                    ss = sdecrease(1:kappa);
                    Utilde = Utilde(:,id(1:kappa));
                    Vtilde = Vtilde(:,id(1:kappa));
                end
                
                % Compress and truncate the vectors to form the matrices Uleft and Vright, to check the error for the next snapshot.
                [uUtot,sUtot,~] = svd(Utilde*diag(sqrt(ss)),0);  sUtot=diag(sUtot);
                [uVtot,sVtot,~] = svd(Vtilde*diag(sqrt(ss)),0);  sVtot=diag(sVtot);
                
                p1=length(sUtot)-sum((sqrt(cumsum(sUtot(end:-1:1).^2))/norm(sUtot))<tauh);
                p2=length(sVtot)-sum((sqrt(cumsum(sVtot(end:-1:1).^2))/norm(sVtot))<tauh);
                
                
                
                Uleft=uUtot; % These are the POD Bases
                Vright=uVtot;
                
            end
            
            
            if nonlinear
                %%% NONLINEAR SNAPSHOT %%%%
                
                FUtilde = [FUtilde, Futilde(:,1:Fntilde)];
                FVtilde = [FVtilde, Fvtilde(:,1:Fntilde)];
                
                Fss = [Fss, Fstildep];
                
                if length(Fss) > kappa %% When the collected singular triplets become more than kmax, truncate and keep the top kmax
                    [Fsdecrease,Fid] = sort(Fss,'descend');
                    Fss = Fsdecrease(1:kappa);
                    FUtilde = FUtilde(:,Fid(1:kappa));
                    FVtilde = FVtilde(:,Fid(1:kappa));
                end
                
                % Compress and truncate the vectors to form the matrices Uleft and Vright, to check the error for the next snapshot.
                
                [FuUtot,FsUtot,~] = svd(FUtilde*diag(sqrt(Fss)),0);  FsUtot=diag(FsUtot);
                [FuVtot,FsVtot,~] = svd(FVtilde*diag(sqrt(Fss)),0);  FsVtot=diag(FsVtot);
                
                Fp1=length(FsUtot)-sum((cumsum(FsUtot(end:-1:1))/sum(FsUtot))<tauh);
                Fp2=length(FsVtot)-sum((cumsum(FsVtot(end:-1:1))/sum(FsVtot))<tauh);
                
                
                Dleft=FuUtot; % These are the bases for DEIM
                Dright=FuVtot;
                
                
                
                
            end
            
            svdtime = svdtime + toc(svdtimer); % Timer for basis construction
            
        end   % if solution || nonlinear
        
        
        
        
        err2(i) = errr;
        timer = timer+h;
        phasetimer = phasetimer + 1;
        
        %%% SNAPSHOT SOLUTION %%%
        if included && timer > 0 %%% If the snapshot was included, check the error after inclusion (Just experimental).
            Xapprox = Uleft*(Uleft'*Xnew*Vright)*Vright';
            err2(i) = norm(Xnew - Xapprox,'fro')/norm(Xnew,'fro');
        end
        
        %%% NONLINEAR SNAPSHOT %%%
        
        Ferr2(i) = Ferrr;
        if Fincluded && timer > 0 %%% If the snapshot was included, check the error after inclusion (Just experimental).
            Fapprox = Dleft*(Dleft'*Fnew*Dright)*Dright';
            Ferr2(i) = norm(Fnew - Fapprox,'fro')/norm(Fnew,'fro');
        end
        
        
        
        
        % disp([i-1, included, Fincluded])
        q = i-1;
        fprintf('Snapshot no. %d... Snapshot sol. included? %d ... Nonlinear snap included? %d\n',q,included, Fincluded)
        i=i+1;
        
    end  % end of while timer
    
    totalnodes = totalnodes+min(countsnaps,nsnaps);
    Ftotalnodes = Ftotalnodes+min(Fcountsnaps,nsnaps);
    errorafter2 = sum(err)/length(err);
    Ferrorafter2 = sum(Ferr)/length(Ferr);
    
    err2=0;
    Ferr2 = 0;
    err=0;
    Ferr = 0;
    
    if  phase == 2 && errorafter2 < error2tau && Ferrorafter2 < error2tau
        
        totalnodes2 = totalnodes;
        fprintf('Convergence happened after phase %d already....\n\n\n\n',phase)
        
        break
    end
    
end   % end of phase
%

fprintf('\n Basis size k1,k2, p1, p2: %d %d %d %d \n',p1,p2,Fp1,Fp2)


Uleft=uUtot(:,1:p1); % These are the POD Bases
Vright=uVtot(:,1:p2);

Dleft=FuUtot(:,1:Fp1); % These are the bases for DEIM
Dright=FuVtot(:,1:Fp2);


Vright = Uleft;
Dright = Dleft;

%% Integrate reduced model and compare solution to full model %%

% Form the DEIM interolation and oblique projector

deimtimer = tic;
[~,~,II]=qr(Dleft','vector');  II=II(1:Fp1)';
[~,~,JJ]=qr(Dright','vector'); JJ=JJ(1:Fp2)';
deimtime = toc(deimtimer);
Uapprox=Dleft/Dleft(II,:);
Vapprox=Dright/Dright(JJ,:);

% Form the ROM by projection

Ak = Uleft'*A*Uleft;       Bk = Vright'*A'*Vright;


Gk = Uleft'*G*Vright;      Y0 = Uleft'*X0*Vright;
Deimleft = Uleft'*Uapprox; Deimright = Vright'*Vapprox;

[V1small,D1small] = eig(full(Ak)); %eigenvalue decomposition of symmetric A
dd1small = diag(D1small);
V1ismall = inv(V1small);

[V2small,D2small] = eig(full(Bk));
dd2small = diag(D2small);
V2ismall = inv(V2small);

fprintf('\n Computing average error....')
errh = [];

nt = 300; % Number of timesteps checked online

Yset = zeros(p1,p2,nt-1); %Initialize tensor to store the small solutions
Yset(:,:,1) = Y0;

h = tf/nt;
timer = 0;



Abig = sparse( kron(Bk', eye(size(Ak))) + kron(eye(size(Bk)), Ak) );
PP = speye(size(Uleft,1));

PR = PP(:,JJ);
PL = PP(:,II);



ntref = 4000;
hvec = tf/ntref;


Dkron = kron(Deimright, Deimleft);
Pkron = kron(PR'*Vright,PL'*Uleft);


odefun = @(t,y) Abig*y +  Dkron*F(t,Pkron*y);
deimfun = @(t,y) Dkron*F(t,Pkron*y);


tspan = 0:hvec:tf;

rat = h/hvec;




%%%%%%%%%% FULL MODEL %%%%%%%%%%%%%

ht = h;

Q1plusbig = V1; I_Q1plusbig = V1i; E1plusbig = D1;
QBplusbig = V2; I_QBplusbig = V2i; EBplusbig = D2;




for ii = 1:nn
    for jj = 1:nn
        LL1big(ii,jj) = 1/((h*dd1(ii)) + (h*dd2(jj)));
    end
end

EAB1big = exp(ht*diag(E1plusbig))*exp(ht*diag(EBplusbig).');


Xnew = X0;

Fnewbig = F(timer,Xnew);

U1eigbig = I_Q1plusbig*X0*QBplusbig;




%%%%%%%%%% SMALL MODEL %%%%%%%%%%%%

exptime = 0;
ht = h;

Q1plus = V1small; I_Q1plus = V1ismall; E1plus = D1small;
QBplus = V2small; I_QBplus = V2ismall; EBplus = D2small;



for ii = 1:p1
    for jj = 1:p2
        LL1(ii,jj) = 1/((h*dd1small(ii)) + (h*dd2small(jj)));
    end
end

EAB1 = exp(ht*diag(E1plus))*exp(ht*diag(EBplus).');

i=1;

err(i)=  norm(X0 - Uleft*Y0*Vright','fro')/norm(Xnew,'fro');

i = i+1;

Ynew = Y0;

Fnewsmall = Deimleft*F(timer,Uleft(II,:)*Ynew*Vright(JJ,:)')*Deimright';
timer = timer+h;

U1eig = I_Q1plus*Y0*QBplus;


while timer < tf-h
    
    %%%%%% BIG MODEL %%%%%%
    
    G1big = Fnewbig;
    ProjG1big = I_Q1plusbig*G1big*QBplusbig;
    ErhsUbig = EAB1big.*ProjG1big-ProjG1big;
    Ubig = EAB1big.*(U1eigbig)+ h*(LL1big.*ErhsUbig);
    U1eigbig = Ubig;
    Ubig = Q1plusbig*Ubig*I_QBplusbig;
    Xnew = Ubig;
    
    
    
    Fnewbig = F(timer,Xnew);
    
    %%%%%% SMALL MODEL %%%%%%
    exptimer = tic;
    
    G1 = Fnewsmall;
    ProjG1 = I_Q1plus*G1*QBplus;
    ErhsU = EAB1.*ProjG1-ProjG1;
    U = EAB1.*(U1eig)+ h*(LL1.*ErhsU);
    U1eig = U;
    U = Q1plus*U*I_QBplus;
    Ynew = U;
    Yset(:,:,i) = real(Ynew);
    
    Fnewsmall = Deimleft*F(timer,Uleft(II,:)*Ynew*Vright(JJ,:)')*Deimright';
    exptime = exptime+toc(exptimer);
    
    err(i)=  norm(Xnew - Uleft*Ynew*Vright','fro')/norm(Xnew,'fro');
    
    
    timer = timer+h;
    i = i+1;
    
    
end


romtime = exptime;





fprintf('\n')
fprintf('\n ROM integrated with %s at %d timesteps in %d seconds\n','Exponential Euler',nt,romtime)
fprintf('\n Final Average error %d, bases built by %d solution nodes and %d nonlinear nodes \n',sum(err(1:end))/nt, totalnodes, Fcountsnaps)
fprintf('\n Bases costructed with %s in %d seconds \n','dynamic',svdtime)
fprintf('Done.\n')

end
