same(A, A).

foo(A, B, A, D, D) :-
    a(A),
    b(B).

a(a).
b(b).


bar(foo(A, x), A, A).

p(_, g(_), f(_, _, _)).

chained(l(A, B, f(C, A), D)) :-
    chainedA(A),
    chainedC(A, D),
    chainedD(D),
    chainedB(b(B, C)).

chainedA(a(x, _)).
chainedB(b(C, c(C))).
chainedC(X, X).
chainedD(a(_, y)).


also_chained(A, C, E) :-
    also_chainedA(A, a(b(C), d(e)), a(b(c), d(E))).

also_chainedA(A, A, A).

deep_chain(A, a(b(c(d(e(f(g(A))))))), a(b(c(d(e(f(g(A)))))))).


other_test(X) :-
    also_chainedA(X, a(b(c(d(e(a, _))))), a(b(c(d(e(_, b)))))).

abc(a).
abc(b).
abc(c).

