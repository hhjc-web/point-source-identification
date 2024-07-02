function FEM
model=createpde;
R1=[3,4,-1,1,1,-1,-1,-1,1,1]'; % describe domain
gm=R1;
g=decsg(gm);
geometryFromEdges(model,g);
n=10;
c=10;

% errorL2=zeros(n,1); errorH1=zeros(n,1); AveH=zeros(n,1);
Aveh0=[0.2,0.15,0.1,0.07,0.05,0.03,0.02,0.015,0.012,0.01];

for k=n:n
    mesh=generateMesh(model,'GeometricOrder','linear','Hmax',Aveh0(k));
    [p,e,t]=meshToPet(mesh);
    pdemesh(p,e,t);
    G=t;
    Np=length(p);
    A=zeros(Np,Np);
    b=zeros(Np,1);
    [w,xi]=GaussianQuadraturePW(6);
    Nxi=length(w);
    VecN=xi;
    VecNx=ones(Nxi,3);
    VecNy=ones(Nxi,3);
    for i=1:length(t)
        K=zeros(3,3);
        L=zeros(3,1);
        x1=p(1,G(1,i));y1=p(2,G(1,i));
        x2=p(1,G(2,i));y2=p(2,G(2,i));
        x3=p(1,G(3,i));y3=p(2,G(3,i));
        S=det([ones(1,3);[x1,x2,x3];[y1,y2,y3]])/2;
        DVecx=[y2-y3,y3-y1,y1-y2];
        DVecy=-[x2-x3,x3-x1,x1-x2];
        tempx=xi*[x1;x2;x3]; tempy=xi*[y1;y2;y3];        
        kappa=tempx.^2+tempy.^2+1;
        Phi1=-1/(4*pi)*log((tempx-1/3).^2+(tempy-1/3).^2);
        Phi2=-1/(4*pi)*log((tempx+1/3).^2+(tempy+1/3).^2);
        VecF1=-c./kappa.*(4*Phi1-(tempx.*(tempx - 1/3)+tempy.*(tempy - 1/3))./(pi*((tempx-1/3).^2 + (tempy-1/3).^2)))+c./kappa.^2.*(4*tempx.^2+4*tempy.^2).*Phi1 - c./kappa.*(4*Phi2-(tempx.*(tempx + 1/3)+tempy.*(tempy + 1/3))./(pi*((tempx+1/3).^2 + (tempy+1/3).^2)))+c./kappa.^2.*(4*tempx.^2+4*tempy.^2).*Phi2;
%         VecF1=-8*tempx.^2-8*tempy.^2-4;
        for j1=1:3
            for j2=1:3
                temp=kappa.*(VecNx(:,j1).*VecNx(:,j2)*DVecx(j1)*DVecx(j2)/(4*S^2)+VecNy(:,j1).*VecNy(:,j2)*DVecy(j1)*DVecy(j2)/(4*S^2));
                K(j1,j2)=S*w*temp;
            end
            temp=VecF1.*VecN(:,j1);
            L(j1)=S*w*temp;
        end
        for j1=1:3
            for j2=1:3
                A(G(j1,i),G(j2,i))=A(G(j1,i),G(j2,i))+K(j1,j2);
            end
            b(G(j1,i),1)=b(G(j1,i),1)+L(j1);
        end
    end
    eta=GaussianQPoints(0,1,4);
    NE=[1-eta;eta];
    for i=1:length(e)
        x1=p(1,e(1,i)); y1=p(2,e(1,i));
        x2=p(1,e(2,i)); y2=p(2,e(2,i));
        hxy=sqrt((x1-x2)^2+(y1-y2)^2);
        tempx=x1+(x2-x1)*eta;
        tempy=y1+(y2-y1)*eta;
        kappa=tempx.^2+tempy.^2+1;
        Phi1=-1/(4*pi)*log((tempx-1/3).^2+(tempy-1/3).^2);
        Phi2=-1/(4*pi)*log((tempx+1/3).^2+(tempy+1/3).^2);
        
        if(e(5,i)==1)
            VecF=-2*c/8 - c/(2*pi)*(tempy-1/3)./((tempx-1/3).^2+(tempy-1/3).^2) - 2*c*tempy.*Phi1./kappa - c/(2*pi)*(tempy+1/3)./((tempx+1/3).^2+(tempy+1/3).^2) - 2*c*tempy.*Phi2./kappa;
%             VecF=-2*tempy.*kappa;
            L=zeros(2,1);
            for j1=1:2
                temp=VecF.*NE(j1,:)*hxy;
                L(j1)=GaussianQuadrature(0,1,temp,4);
            end
            for j1=1:2
                b(e(j1,i),1)=b(e(j1,i),1)+L(j1);
            end
        elseif(e(5,i)==2)
            VecF=-2*c/8 + c/(2*pi)*(tempx-1/3)./((tempx-1/3).^2+(tempy-1/3).^2) + 2*c*tempx.*Phi1./kappa + c/(2*pi)*(tempx+1/3)./((tempx+1/3).^2+(tempy+1/3).^2) + 2*c*tempx.*Phi2./kappa;
%             VecF=2*tempx.*kappa;
            L=zeros(2,1);
            for j1=1:2
                temp=VecF.*NE(j1,:)*hxy;
                L(j1)=GaussianQuadrature(0,1,temp,4);
            end
            for j1=1:2
                b(e(j1,i),1)=b(e(j1,i),1)+L(j1);
            end
        elseif(e(5,i)==3)
            VecF=-2*c/8 + c/(2*pi)*(tempy-1/3)./((tempx-1/3).^2+(tempy-1/3).^2) + 2*c*tempy.*Phi1./kappa + c/(2*pi)*(tempy+1/3)./((tempx+1/3).^2+(tempy+1/3).^2) + 2*c*tempy.*Phi2./kappa;
%             VecF=2*tempy.*kappa;
            L=zeros(2,1);
            for j1=1:2
                temp=VecF.*NE(j1,:)*hxy;
                L(j1)=GaussianQuadrature(0,1,temp,4);
            end
            for j1=1:2
                b(e(j1,i),1)=b(e(j1,i),1)+L(j1);
            end
        elseif(e(5,i)==4)
            VecF=-2*c/8 - c/(2*pi)*(tempx-1/3)./((tempx-1/3).^2+(tempy-1/3).^2) - 2*c*tempx.*Phi1./kappa - c/(2*pi)*(tempx+1/3)./((tempx+1/3).^2+(tempy+1/3).^2) - 2*c*tempx.*Phi2./kappa;
%             VecF=-2*tempx.*kappa;
            L=zeros(2,1);
            for j1=1:2
                temp=VecF.*NE(j1,:)*hxy;
                L(j1)=GaussianQuadrature(0,1,temp,4);
            end
            for j1=1:2
                b(e(j1,i),1)=b(e(j1,i),1)+L(j1);
            end
        end
    end
    
    VecD=find(abs(p(1,:)-1)<1e-10 & abs(p(2,:)-1)<1e-10);
    A(VecD,:)=0; A(VecD,VecD)=1; 
    b(VecD)=c/(4*pi)*log(8/9)/3+c/(4*pi)*log(32/9)/3;
   
% for i=1:length(VecD)
%         p1=e(1,VecD(i)); p2=e(2,VecD(i));
%         A(p1,:)=0; A(p2,:)=0;
%         A(p1,p1)=1;A(p2,p2)=1;
%         b(p1)=0;b(p2)=0;
%     end
    
    v=A\b;
    x=transpose(p);
    u=v-c/(4*pi)*log((x(:,1)-1/3).^2+(x(:,2)-1/3).^2)./(x(:,1).^2+x(:,2).^2+1)-c/(4*pi)*log((x(:,1)+1/3).^2+(x(:,2)+1/3).^2)./(x(:,1).^2+x(:,2).^2+1);
    x_bound=x(e(5,:)==1 | e(5,:)==2 | e(5,:)==3 | e(5,:)==4,:);
    v_bound=v(e(5,:)==1 | e(5,:)==2 | e(5,:)==3 | e(5,:)==4);
    for i=1:length(e)
        t=10;
        eta_extra=transpose(1/t:1/t:(1-1/t));
        v_extra=(1-eta_extra)*v(e(1,i))+eta_extra*v(e(2,i));
        x_extra=(1-eta_extra)*x(e(1,i),:)+eta_extra*x(e(2,i),:);
        x_bound = [x_bound;x_extra];
        v_bound = [v_bound;v_extra];
    end
    u_bound=v_bound-c/(4*pi)*log((x_bound(:,1)-1/3).^2+(x_bound(:,2)-1/3).^2)./(x_bound(:,1).^2+x_bound(:,2).^2+1)-c/(4*pi)*log((x_bound(:,1)+1/3).^2+(x_bound(:,2)+1/3).^2)./(x_bound(:,1).^2+x_bound(:,2).^2+1);
    save("data.mat",'x','v','u','x_bound','u_bound');
    if(k==n)
%         pdemesh(p,e,t,u);
        figure(1)
        pdeplot(model,"XYData",u);
%         print('true_solution-4','-depsc')
        load 'solution.mat' pred_solution
        figure(2)
        pdeplot(model,"XYData",pred_solution);
%         print('pred_solution-4','-depsc')
        figure(3)
        pdeplot(model,"XYData",abs(u-pred_solution));
%         print('error_solution-4','-depsc')
    end
%     for i=1:length(t)
%         x1=p(1,G(1,i)); y1=p(2,G(1,i));
%         x2=p(1,G(2,i)); y2=p(2,G(2,i));
%         x3=p(1,G(3,i)); y3=p(2,G(3,i));
%         S=det([ones(1,3);[x1,x2,x3];[y1,y2,y3]])/2;
%         DVecx=[y2-y3,y3-y1,y1-y2];
%         DVecy=-[x2-x3,x3-x1,x1-x2];
%         tempx=xi*[x1;x2;x3]; tempy=xi*[y1;y2;y3];
%         utrue=tempx.^2+tempy.^2;
%         dxu=2*tempx;
%         dyu=2*tempy;
%         
%         tempu=zeros(Nxi,1);
%         tempdxu=zeros(Nxi,1);
%         tempdyu=zeros(Nxi,1);
%         for j=1:3
%             tempu=tempu+u(G(j,i))*VecN(:,j);
%             tempdxu=tempdxu+u(G(j,i))*VecNx(:,j)*DVecx(j)/(2*S);
%             tempdyu=tempdyu+u(G(j,i))*VecNy(:,j)*DVecy(j)/(2*S);
%         end
%         tempu=tempu-utrue;
%         tempdxu=tempdxu-dxu;
%         tempdyu=tempdyu-dyu;
%         errorL2(k)=errorL2(k)+S*w*tempu.^2;
%         errorH1(k)=errorH1(k)+S*w*(tempdxu.^2+tempdyu.^2)+S*w*tempu.^2;
%     end
%     errorL2(k)=sqrt(errorL2(k));
%     errorH1(k)=sqrt(errorH1(k));
%     AveH(k)=sqrt(1/length(t));
% end
% 
% figure
% loglog(AveH,errorL2,'--','DisplayName','errorL2');
% hold on
% loglog(AveH,AveH.^2,'--','DisplayName','O(h^2)');
% legend
% hold off
% 
% figure
% loglog(AveH,errorH1,'--','DisplayName','errorH1');
% hold on
% loglog(AveH,AveH,'--','DisplayName','O(h)');
% legend
% hold off
% 
% alpha1=(log(errorL2(n))-log(errorL2(n-1)))/(log(AveH(n))-log(AveH(n-1)));
% alpha2=(log(errorH1(n))-log(errorH1(n-1)))/(log(AveH(n))-log(AveH(n-1)));

end