close all
sys = ss([1 1; 0 1], [0 ;1], [1 0], [0], -1);
dist = bimodal_gaussian;

k = ss(dist.DRInC.a, dist.DRInC.b, dist.DRInC.c, [], -1);
cldrinc = ss([sys.A, sys.B * k.C; k.B * sys.C * sys.A, k.A], ...
             [0*sys.C' ; -k.B], [sys.C, 0*k.B'], [], -1);
kdrinc = k;

k = ss(dist.Robust.a, dist.Robust.b, dist.Robust.c, [], -1);
clrob = ss([sys.A, sys.B * k.C; k.B * sys.C * sys.A, k.A], ...
           [0*sys.C' ; -k.B], [sys.C, 0*k.B'], [], -1);
krob = k;

k = ss(dist.LQG.a, dist.LQG.b, dist.LQG.c, [], -1);
cllqg = ss([sys.A, sys.B * k.C; k.B * sys.C * sys.A, k.A], ...
           [0*sys.C' ; -k.B], [sys.C, 0*k.B'], [], -1);
klqg = k;

k = ss(dist.DRLQG.a, dist.DRLQG.b, dist.DRLQG.c, [], -1);
cldrlqg = ss([sys.A, sys.B * k.C; k.B * sys.C * sys.A, k.A], ...
             [0*sys.C' ; -k.B], [sys.C, 0*k.B'], [], -1);
kdrlqg = k;


max(abs(eig(cldrinc.A)))
max(abs(eig(clrob.A)))
max(abs(eig(cllqg.A)))
max(abs(eig(cldrlqg.A)))

figure;
hold on;
bode(kdrinc);
bode(krob);
bode(klqg);
bode(kdrlqg);
bode(sys);
legend("DRInC","Robust","LQG","DRLQG","System");
title("Component frequency response");

figure;
hold on;
plot(irf(ssm(kdrinc.a, kdrinc.b, kdrinc.c)));
plot(irf(ssm(krob.a, krob.b, krob.c)));
plot(irf(ssm(klqg.a, klqg.b, klqg.c)));
plot(irf(ssm(kdrlqg.a, kdrlqg.b, kdrlqg.c)));
plot(irf(ssm(sys.a, sys.b, sys.c)));
legend("DRInC","Robust","LQG","DRLQG","System");
title("Component impulse response");

figure;
hold on;
bode(cldrinc);
bode(clrob);
bode(cllqg);
bode(cldrlqg);
legend("DRInC","Robust","LQG","DRLQG");
title("Closed loop frequency response");

figure;
hold on;
plot(irf(ssm(cldrinc.a, cldrinc.b, cldrinc.c)));
plot(irf(ssm(clrob.a, clrob.b, clrob.c)));
plot(irf(ssm(cllqg.a, cllqg.b, cllqg.c)));
plot(irf(ssm(cldrlqg.a, cldrlqg.b, cldrlqg.c)));
legend("DRInC","Robust","LQG","DRLQG");
title("Closed loop impulse response");
