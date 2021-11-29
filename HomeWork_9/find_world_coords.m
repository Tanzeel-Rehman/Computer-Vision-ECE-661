function coords_world = find_world_coords(board_size) 
%Intialize the world coordinates
coords_world = zeros(80,2); 
%Generate World Coordinates
for i = 1:8
    for j = 1:10
        coords_world((i-1)* 10 + j,:)=[(i-1)*board_size, (j-1)*board_size];
    end
end
end