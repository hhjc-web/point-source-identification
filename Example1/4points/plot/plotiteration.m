load('loss.mat')

figure(1)
p = loglog(iteration, solution_error);
grid on
xlim([0,12000])
p.LineWidth = 1.2;
xlabel('i') 
ylabel('Relative error') 
ax = gca;
ax.FontSize=12;
print('error-1','-depsc')

figure(2)
p = loglog(iteration, solution_loss);
grid on
p.LineWidth = 1.2;
ylim([0.001,100])
xlabel('i') 
ylabel('Loss') 
ax = gca;
ax.FontSize=12;
print('loss-1','-depsc')

figure(3)
p1=plot(iteration,point_1(:,1),'r-*','MarkerSize',5);
hold on
p2=plot(iteration,point_1(:,2),'k-*','MarkerSize',5);
p3=plot(iteration,point_1(:,3),'b-*','MarkerSize',5);
grid on
xlabel('i') 
ylabel('Value') 
legend({'c','x_1','x_2'},'Location','northwest','FontSize',12)
hold off
ax = gca;
ax.FontSize=12;
print('point4-1','-depsc')

figure(4)
p1=plot(iteration,point_2(:,1),'r-*','MarkerSize',5);
hold on
p2=plot(iteration,point_2(:,2),'k-*','MarkerSize',5);
p3=plot(iteration,point_2(:,3),'b-*','MarkerSize',5);
grid on
xlabel('i') 
ylabel('Value') 
legend({'c','x_1','x_2'},'Location','northwest','FontSize',12)
hold off
ax = gca;
ax.FontSize=12;
print('point3-1','-depsc')

figure(5)
p1=plot(iteration,point_3(:,1),'r-*','MarkerSize',5);
hold on
p2=plot(iteration,point_3(:,2),'k-*','MarkerSize',5);
p3=plot(iteration,point_3(:,3),'b-*','MarkerSize',5);
grid on
xlabel('i') 
ylabel('Value') 
legend({'c','x_1','x_2'},'Location','northwest','FontSize',12)
hold off
ax = gca;
ax.FontSize=12;
print('point2-1','-depsc')

figure(6)
p1=plot(iteration,point_4(:,1),'r-*','MarkerSize',5);
hold on
p2=plot(iteration,point_4(:,2),'k-*','MarkerSize',5);
p3=plot(iteration,point_4(:,3),'b-*','MarkerSize',5);
grid on
xlabel('i') 
ylabel('Value') 
legend({'c','x_1','x_2'},'FontSize',12)
hold off
ax = gca;
ax.FontSize=12;
print('point1-1','-depsc')