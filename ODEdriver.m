                    %% 2S-POD-DEIM %%

clear all
clc

% This code is used to call the 2S-POD-DEIM algorithm to integrate a
% convection-diffusion-reaction ODE, where the reduced model is integrated
% by a matrix-oriented first-order exponential Euler scheme.

% Related Article:

% G. Kirsten and Valeria Simoncini. 
% A matrix-oriented POD-DEIM algorithm applied to nonlinear differential matrix equations. 
% pp. 1-25, June 2020. Revised May 2021. arXiv: 2006.13289   [math.NA]. 

% Related Example: Example 3, page 16.

%We provide this code without any guarantee that the code is bulletproof as input 
%parameters are not checked for correctness.
%Finally, if the user encounters any problems with the code, the author can be
%contacted via e-mail.



%% Here we define the parameters

nn =499;        % nn is the problem dimension, that is the solution has size nn+1 x nn+1
nsmax = 400;    % The maximum number of snapshots available in the dynamic procedure
kappa = 70;     % The maximum admissable dimension of the reduced basis
tau = 1e-4;     % The tolerance for truncating the basis and dynamic snapshot selection
tf = 0.3;       % The end of the timespan T = [0, tf]


%% Here we define the coefficient matrices and nonlinear function %%


sigma = 0.05;
BC = zeros(nn); BC(1,1) = 2; BC(1,2) = -1/2; BC(end,end-1) = -1/2; BC(end,end) = 2;
BC = (2/3)*BC;

e = ones(nn, 1);
A_1D = spdiags([e, -2*e, e], -1:1, nn, nn);
A_1D = A_1D + BC; % Neumann

x = linspace(0,1,nn)';
y = linspace(0,1,nn);

[X,Y] = meshgrid(x,y);

dx = x(2) - x(1);
B = (1/(2*dx))*(spdiags([-1*ones(nn,1), ones(nn,1)],[-1,1], nn, nn));
B(1,2) = 0;
B(end-1,end) = 0;

ode.A = (1/dx^2)*sigma*(A_1D) + B;


ode.X0 = 0.3 + 256*(X.*(1-X).*Y.*(1-Y)).^2;

ode.F = @(t,U) U.*(U-0.5).*(1-U);
ode.G = zeros(nn);
ode.B = B;






%%


params.nn = nn-1;
params.nmax = nsmax;
params.kaps = kappa;
params.tau = tau;
params.tf = tf;
params.ode = ode;

[Uleft, Vright, Yset] = integrateODE(params);









