close all;

order = [4, 3, 5, 6, 2, 1];

for i = 1:6
    method = order(i);
    
    Time = out{method}.Time;
    RMSE = out{method}.RMSE;
    obj  = out{method}.obj/max(out{method}.obj);

    figure(1);
    hold on;
    plot(Time, obj);
    figure(2);
    hold on;
    plot(Time, RMSE)
end

figure(1);
ylabel('normalized object');
xlabel('time(sec)');

figure(2);
ylabel('testing RMSE');
xlabel('time(sec)');