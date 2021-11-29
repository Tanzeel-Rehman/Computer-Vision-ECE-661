function sorted_corners = FindCorners(lines,gray_img)
%Function for finding and sorting the corners
%Divide the lines in horizontal vs vertical lines
mask = abs([lines.theta])<=40;
vertical_lines = lines(mask);
horizontal_lines = lines(~mask);
%Initialize an array of corners
corners = zeros(length(vertical_lines)*length(horizontal_lines),2);

for i= 1:length(vertical_lines)
    for j = 1:length(horizontal_lines)  %For every vertical lines check all horizontal lines
        %Find the intersection between every pair of horizontal and
        %vertical lines
        intersection = IntersectLines(vertical_lines(i), horizontal_lines(j));
        %Check if the intersection is within image bounds
        if(intersection(1)>1 && intersection(2)>1 && intersection(1)<size(gray_img,2) && intersection(2)<size(gray_img,1))
            %Append the nrew corners point in the list
            corners ((i-1)*10+j,:) = intersection;
        end
    end
end

%Find the minimum corners along eight vertical lines
x = zeros(length(vertical_lines),1);
for i=1:length(vertical_lines)
x(i)=min(corners((i-1)*10+1:i*10));
end
[~, ind] = sort(x,'ascend');

%Sort the eight vertical lines and their corresponding horizontal corners 
xs10 = zeros(length(vertical_lines)*length(horizontal_lines),2);
for i=1:8
    xs10((i-1)*10+1:i*10,:)=corners((ind(i)-1)*10+1:(ind(i)-1)*10+10,:);
end

%Finally sort the 10 corners for every vertical lines based on the y values
sorted_corners = zeros(80,2);
for i=1:8
    Imagecorners_temp = xs10((i-1)*10+1:i*10,:);
    sorted_corners((i-1)*10+1:i*10,:) = sortrows(Imagecorners_temp,2,'ascend');
end
end