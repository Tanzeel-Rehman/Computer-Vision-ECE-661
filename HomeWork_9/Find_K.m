function K = Find_K(b)
w11 = b(1);
w12 = b(2);
w22 = b(3);
w13 = b(4);
w23 = b(5);
w33 = b(6);
% Find the Intrinsic parameters
x0 = (w12* w13- w11* w23)/(w11 * w22 - w12^2);
lambda = w33-(w13^2 + x0 *(w12 * w13 - w11 * w23))/w11;
alphaX = sqrt(lambda / w11);
alphaY = sqrt(lambda * w11/(w11*w22 - w12^2));
s = - w12 * alphaX^2 * alphaY/lambda;
y0 = s*x0/alphaY - (w13 * alphaX^2/lambda);
K = [alphaX, s, x0; 0, alphaY, y0; 0, 0, 1];

 end