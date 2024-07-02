k = 1:10:800;

load('loss0.mat')
solution_error0 = solution_error(k);
solution_loss0 = solution_loss(k);
load('loss2.mat')
solution_error2 = solution_error(k);
solution_loss2 = solution_loss(k);
load('loss5.mat')
solution_error5 = solution_error(k);
solution_loss5 = solution_loss(k);
load('loss10.mat')
solution_error10 = solution_error(k);
solution_loss10 = solution_loss(k);

iteration = iteration(k);

figure(1)
p1 = plot(iteration, solution_error0, 'r-');
hold on
p2 = plot(iteration, solution_error2, 'b-');
p3 = plot(iteration, solution_error5, 'c-');
p4 = plot(iteration, solution_error10, 'm-');
grid on
p1.LineWidth = 1.2;
p2.LineWidth = 1.2;
p3.LineWidth = 1.2;
p4.LineWidth = 1.2;
ylim([-0.01,0.5])
xlabel('Iteration') 
ylabel('Relative error')
legend({'\delta=0%','\delta=2%','\delta=5%','\delta=10%'},'Location','northeast','FontSize',12)
hold off
ax = gca;
ax.FontSize=12;
print('error-2','-depsc')

figure(2)
p1 = plot(iteration, solution_loss0, 'r-');
hold on
p2 = plot(iteration, solution_loss2, 'b-');
p3 = plot(iteration, solution_loss5, 'c-');
p4 = plot(iteration, solution_loss10, 'm-');
grid on
p1.LineWidth = 1.2;
p2.LineWidth = 1.2;
p3.LineWidth = 1.2;
p4.LineWidth = 1.2;
ylim([-0.5,20])
xlabel('Iteration') 
ylabel('Loss')
legend({'\delta=0%','\delta=2%','\delta=5%','\delta=10%'},'Location','northeast','FontSize',12)
hold off
ax = gca;
ax.FontSize=12;
print('loss-2','-depsc')

% figure(3)
% p1=plot(iteration,point_1(:,1),'r-*','MarkerSize',5);
% hold on
% p2=plot(iteration,point_1(:,2),'k-*','MarkerSize',5);
% p3=plot(iteration,point_1(:,3),'b-*','MarkerSize',5);
% grid on
% xlabel('Iteration') 
% ylabel('Value') 
% legend({'c','x_1','x_2'},'Location','northwest','FontSize',12)
% hold off
% ax = gca;
% ax.FontSize=12;
% print('point4-1','-depsc')
% 
% figure(4)
% p1=plot(iteration,point_2(:,1),'r-*','MarkerSize',5);
% hold on
% p2=plot(iteration,point_2(:,2),'k-*','MarkerSize',5);
% p3=plot(iteration,point_2(:,3),'b-*','MarkerSize',5);
% grid on
% xlabel('Iteration') 
% ylabel('Value') 
% legend({'c','x_1','x_2'},'Location','northwest','FontSize',12)
% hold off
% ax = gca;
% ax.FontSize=12;
% print('point3-1','-depsc')
% 
% figure(5)
% p1=plot(iteration,point_3(:,1),'r-*','MarkerSize',5);
% hold on
% p2=plot(iteration,point_3(:,2),'k-*','MarkerSize',5);
% p3=plot(iteration,point_3(:,3),'b-*','MarkerSize',5);
% grid on
% xlabel('Iteration') 
% ylabel('Value') 
% legend({'c','x_1','x_2'},'Location','northwest','FontSize',12)
% hold off
% ax = gca;
% ax.FontSize=12;
% print('point2-1','-depsc')
% 
% figure(6)
% p1=plot(iteration,point_4(:,1),'r-*','MarkerSize',5);
% hold on
% p2=plot(iteration,point_4(:,2),'k-*','MarkerSize',5);
% p3=plot(iteration,point_4(:,3),'b-*','MarkerSize',5);
% grid on
% xlabel('Iteration') 
% ylabel('Value') 
% legend({'c','x_1','x_2'},'FontSize',12)
% hold off
% ax = gca;
% ax.FontSize=12;
% print('point1-1','-depsc')