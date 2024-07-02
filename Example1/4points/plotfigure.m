load('solution.mat')

nx=500;
ny=500;

X = -1:2/nx:1;
Y = -1:2/ny:1;

solution = optimal_solution(:);
pred = pred_solution(:);
solution = reshape(solution, [nx+1, ny+1]);
solution = flipud(solution);
pred = reshape(pred, [nx+1, ny+1]);
pred = flipud(pred);

figure(1)
imagesc(X, Y, solution)
ax = gca;
ax.YDir = 'normal';
ax.FontSize=15;
colorbar();
caxis([-1 4]);
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
print('true_solution-1','-depsc')

figure(2)
imagesc(X, Y, pred)
ax = gca;
ax.YDir = 'normal';
ax.FontSize=15;
colorbar();
caxis([-1 4]);
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
print('pred_solution-1','-depsc')

figure(3)
imagesc(X, Y, abs(pred - solution))
ax = gca;
ax.YDir = 'normal';
ax.FontSize=15;
colorbar();
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
print('error_solution-1','-depsc')
