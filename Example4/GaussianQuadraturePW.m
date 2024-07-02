function [w,xi]=GaussianQuadraturePW(typeA)
if(typeA==6)
    xi=zeros(7,3);
    w=zeros(1,7);
    xi(1,:)=[1/3,1/3,1/3];
    temp=[0.736712498968435,0.237932366472434,0.025355134551932];
    xi(2,:)=temp; xi(3,:)=[temp(1),temp(3),temp(2)];
    xi(4,:)=[temp(2),temp(1),temp(3)];
    xi(5,:)=[temp(2),temp(3),temp(1)];
    xi(6,:)=[temp(3),temp(1),temp(2)];
    xi(7,:)=[temp(3),temp(2),temp(1)];
    w=[0.375,0.104166666666667*ones(1,6)];
end