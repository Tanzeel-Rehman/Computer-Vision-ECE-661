function intersection = IntersectLines(lineA, lineB)
%% Code converted from HW1
%Inputs:
%   lineA and lineB: a structure containing the XY coordinates of two points
%          that can define the lines using crossproduct of points 
%Output
%   intersection: An XY representation of the Intersection point
%%
%First from the two pints in HC 
lineA = cross([lineA.point1,1], [lineA.point2, 1]);
%Second line from the two points in HC
lineB = cross([lineB.point1, 1],[lineB.point2, 1]); 
intersection = cross(lineA, lineB);
% Change the HC to XY coordinates 
intersection = double([intersection(1)/intersection(end) intersection(2)/intersection(end)]);
end
