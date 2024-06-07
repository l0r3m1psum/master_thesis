% https://staff.fnwi.uva.nl/a.j.p.heck/Courses/mptut.pdf
% https://www.ntg.nl/literatuur/hobby/mpman.pdf
% https://ctan.math.illinois.edu/info/drawing-with-metapost/Drawing-with-Metapost.pdf

prologues := 3;

beginfig(1);
    path universe, intersection, subset, inneruniverse;
    pair bwd_point;

    width = 8cm; height = 4cm;
    universe = fullcircle xscaled width yscaled height;
    inneruniverse = universe scaled .7;
    intersection = fullcircle xscaled (width/4) yscaled height;
    label("FWD", point +.4(length inneruniverse) of inneruniverse);
    bwd_point = point -.9(length inneruniverse) of inneruniverse;
    label("BWD", bwd_point);
    label("MIX", center universe);
    subset = fullcircle xscaled (width/4) yscaled (height/4)
        shifted (bwd_point - (0,ypart bwd_point));
    label("backprop", center subset);
    draw universe; draw intersection; draw subset;
endfig;

% https://vigna.di.unimi.it/metagraph/
input metagraph

beginfig(2);
    noderadius := 14;

    node.x0(btex $x^{[-1]}$ etex);
    node.x1(btex $x^{[0]}$ etex);
    node.x2(btex $x^{[1]}$ etex);
    node.x3(btex $x^{[2]}$ etex);
    node.x4(btex $x^{[3]}$ etex);
    node.x5(btex $x^{[4]}$ etex);
    node.x6(btex $x^{[5]}$ etex);

    xstep := 2.5cm;
    ystep := 1.5cm;
    x1c = x0c - (0,2ystep);
    x2c = x0c + (1xstep,0);
    x3c = (xpart x2c, ypart .5(x0c+x1c));
    x4c = (2xstep, ypart x1c);
    x5c = (2xstep, ypart x0c);
    x6c = (3xstep, ypart .5(x0c+x1c));
    
    drawboxed(x0, x1, x2, x3, x4, x5, x6);
    drawarrow larc(x2)(--)(x0)("");
    drawarrow larc(x3)(--)(x0)("");
    drawarrow larc(x3)(--)(x1)("");
    drawarrow larc(x5)(--)(x2)("");
    drawarrow larc(x5)(--)(x3)("");
    drawarrow larc(x4)(--)(x1)("");
    drawarrow larc(x6)(--)(x4)("");
    drawarrow larc(x6)(--)(x5)("");
endfig;

beginfig(3);
    defaultfont := "cmss8";
    defaultscale := .9;
    dotlabeldiam := 2dotlabeldiam;
    xstep := 3cm;
    ystep := 1cm;

    def mylabel (expr label, point, pos)(suffix loc) =
        dotlabel.loc(label, point);
        pos = point;
    enddef;

    def mydrawarrow (expr path) =
        % FIXME: I have to remove a fized lenth from the subpath
        drawarrow subpath (.1,.9) of path;
    enddef;

    mylabel("Device",                (0,ystep),       z0)(lft);
    mylabel("Library",               (xstep,2ystep),  z1)(top);
    mylabel("CommandQueue",          (xstep,ystep),   z2)(top);
    mylabel("Buffer",                (1.5xstep,0),      z3)(top);
    mylabel("Function",              (2xstep,2ystep), z4)(top);
    mylabel("CommandBuffer",         (2xstep,ystep),  z5)(top);
    mylabel("ComputePipelineState",  (3xstep,2ystep), z6)(top);
    mylabel("ComputeCommandEncoder", (3xstep,ystep),  z7)(top);

    mydrawarrow(z0--z1);
    mydrawarrow(z1--z4);
    mydrawarrow(z4--z6);
    mydrawarrow(z6--z7);

    mydrawarrow(z0--z2);
    mydrawarrow(z2--z5);
    mydrawarrow(z5--z7);

    mydrawarrow(z0--z3);
    mydrawarrow(z3--z7);
endfig;

beginfig(4);
    node.x0(btex $x$ etex);
    node.x1(btex $\bf x$ etex);
    node.x2(btex $X$ etex);

    ystep := 1.5cm;
    x1c = x0c - (0,ystep);
    x2c = x1c - (0,ystep);
    
    drawboxed(x0, x1, x2);
    drawarrow larc(x0)(--)(x1)("");
    drawarrow larc(x1)(--)(x2)("");
endfig;

end;