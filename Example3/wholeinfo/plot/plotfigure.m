load('solution.mat')

nx=200;
ny=200;

X = -1:2/nx:1;
Y = -1:2/ny:1;

solution = optimal_solution(:);
pred = pred_solution(:);
solution = reshape(solution, [nx+1, ny+1]);
solution = transpose(solution);
pred = reshape(pred, [nx+1, ny+1]);
pred = transpose(pred);

figure(1)
imagesc(X, Y, real(solution))
ax = gca;
ax.YDir = 'normal';
ax.FontSize=15;
colorbar();
caxis([-0.6 1]);
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
print('true_solution-3-r','-depsc')

figure(2)
imagesc(X, Y, real(pred))
ax = gca;
ax.YDir = 'normal';
ax.FontSize=15;
colorbar();
caxis([-0.6 1]);
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
print('pred_solution-3-r','-depsc')

figure(3)
imagesc(X, Y, abs(real(pred) - real(solution)))
ax = gca;
ax.YDir = 'normal';
ax.FontSize=15;
colorbar();
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
print('error_solution-3-r','-depsc')

figure(4)
imagesc(X, Y, imag(solution))
ax = gca;
ax.YDir = 'normal';
ax.FontSize=15;
colorbar();
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
print('true_solution-3-i','-depsc')

figure(5)
imagesc(X, Y, imag(pred))
ax = gca;
ax.YDir = 'normal';
ax.FontSize=15;
colorbar();
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
print('pred_solution-3-i','-depsc')

figure(6)
imagesc(X, Y, abs(imag(pred) - imag(solution)))
ax = gca;
ax.YDir = 'normal';
ax.FontSize=15;
colorbar();
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
print('error_solution-3-i','-depsc')
