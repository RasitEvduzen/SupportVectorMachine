function dy = ridg(t,y,P)


p1=P(1,1);
p2=P(2,1);
p3=P(3,1);

dy = zeros(3,1);
dy(1) = p1 * (y(2)- y(1));
dy(2) = p2*y(1) - y(2) - y(1)*y(3);
dy(3)=  y(1)*y(2) - p3* y(3);
