function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 
%population and profit.


figure; 

plot(x, y, 'rx', 'MarkerSize', 10); 
ylabel('Profit in $10,000s'); 
xlabel('Population of City in 10,000s'); 

end
