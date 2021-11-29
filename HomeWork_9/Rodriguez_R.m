function R =Rodriguez_R(omega)
wx = [0 -omega(3) omega(2); omega(3) 0 -omega(1); -omega(2) omega(1) 0];
phi = norm(omega);
R = eye(3)+ sin(phi)/phi * wx + (1-cos(phi))/phi * wx^2;
end


    